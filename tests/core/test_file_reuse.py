"""Prepared runtime agrees with uncached global queries and original draws."""
from tests.support.readout_target import ZERO_TARGET
from fractions import Fraction
from random import Random
import unittest

from magna import _native, compile_circuit, parse_circuit, run, ControllerRegistry
from magna.file_api import NoiseDraw, _NoiseSchedule
from magna.file_api import ResourceLimit
from tests.support import file_hooks as hooks
from tests.support import native_testing


class FileReuseTests(unittest.TestCase):
    def test_uint64_denominator_boundary_and_reproducibility(self):
        denominator = (1 << 64)-1
        categories = (((denominator//2+7, denominator),), ((1, denominator),),
                      ((denominator-1, denominator),))
        plan = native_testing.load().make_file_noise(categories)
        actual = [row[0] for row in native_testing.load().draw_file_noise(plan, list(range(128)))]
        self.assertTrue(all(row[1:]==(-1, 0) for row in actual))
        self.assertTrue(40 < sum(row[0]==0 for row in actual) < 88)
        self.assertEqual(actual, [native_testing.load().sample_file_categories(categories, seed) for seed in range(128)])

    def test_prepared_noise_weights_and_batch_partition(self):
        big = 10**100
        categories = [[], [(0, 1)], [(1, 1), (0, 3)], [(1, 15000)]*15,
                      [(1, 10), (1, 5)], [(big-1, big), (1, big)], [(1, big), (1, 3)]]
        factors = [((2, 3), (4, 7))]*len(categories)
        plan = native_testing.load().make_file_noise(categories, factors)
        seeds = list(range(128))
        actual = native_testing.load().draw_file_noise(plan, seeds)
        # Algorithms may change the seeded stream across versions. Preserve
        # exact likelihoods, deterministic channels and per-seed batch identity.
        for choices, weight in actual:
            self.assertEqual(choices[:3], (-1, -1, 0))
            expected = Fraction(1)
            for choice, factor in zip(choices, factors): expected *= Fraction(*factor[choice>=0])
            self.assertEqual(Fraction(*weight), expected)
        self.assertEqual(actual, [native_testing.load().sample_file_categories(categories, seed, factors) for seed in seeds])
        self.assertEqual(actual, native_testing.load().draw_file_noise(plan, seeds[:47])+native_testing.load().draw_file_noise(plan, seeds[47:]))
        categories[3].clear()
        self.assertEqual(native_testing.load().draw_file_noise(plan, seeds), actual)
        for category in (((1, 0),), ((-1, 2),), ((1, -2),), ((3, 4), (1, 2))):
            with self.assertRaises(ValueError): native_testing.load().make_file_noise((category,))

    def test_no_noise_does_not_advance_rng_and_batch_matches_single(self):
        empty = _NoiseSchedule(parse_circuit('H 0'))
        rng = Random(12); state = rng.getstate()
        self.assertEqual(hooks.draw_many(empty, rng, 7), [NoiseDraw({}, ())]*7)
        self.assertEqual(rng.getstate(), state)
        schedule = _NoiseSchedule(parse_circuit('DEPOLARIZE2(1/10) 0 1\nM(1/5) 0'),
                                  {'method': 'importance', 'proposal_scale': '2'})
        rng = Random(7)
        expected = [hooks.draw(schedule, rng) for _ in range(70)]
        self.assertEqual(hooks.draw_many(schedule, Random(7), 70), expected)

    def test_fixed_preparation_is_reused_for_new_original_faults(self):
        circuit = parse_circuit('H 0\nT 0\nDEPOLARIZE1(1/10) 0\nM(1/10) 0\nCX rec[-1] 1')
        compiled = compile_circuit(circuit, task={'outputs': [1], 'ideal_state': ZERO_TARGET})
        rng = Random(4)
        for _ in range(20): compiled.sample(rng)
        self.assertEqual(compiled.compilations, 2)

    def test_prepared_scalar_queries_equal_uncached_queries_including_report_flips(self):
        circuit = parse_circuit('H 0\nT 0\nCX 0 1\nDEPOLARIZE2(1/4) 0 1\nM(1/3) 1\nCX rec[-1] 0\n'
                                'DETECTOR rec[-1]\nPOSTSELECT det[0]')
        task = {'outputs': [0], 'observables': [[1, 0], [1, 1]]}
        compiled = compile_circuit(circuit, task=task, ideal_state=parse_circuit('H 0'))
        noises = hooks.draw_many(compiled.noise_schedule, Random(8), 40)
        prepared = [compiled.sample(Random(1), noise=n) for n in noises]
        # Budgeted calls deliberately retain per-query checks and bypass bundles.
        uncached = compile_circuit(circuit, task=task, ideal_state=parse_circuit('H 0'), options={'max_queries': 10000})
        direct = [uncached.sample(Random(1), noise=n) for n in noises]
        self.assertEqual(prepared, direct)

    def test_internal_batch_preserves_samples_weights_and_target_recovery(self):
        physical = parse_circuit('H 0\nT 0\nDEPOLARIZE1(1/7) 0\nH 1\nM(1/5) 1\nCX rec[-1] 0')
        recovery = parse_circuit('H 1\nM 1\nCX rec[-1] 0')
        kwargs = dict(task={'outputs': [0], 'observables': [[1, 0]]}, recovery=recovery,
                      ideal_state=parse_circuit('H 0\nT 0'),
                      sampling={'method': 'importance', 'proposal_scale': '2'})
        batch = compile_circuit(physical, **kwargs)
        single = compile_circuit(physical, **kwargs)
        rng = Random(12)
        expected = [single.sample(rng) for _ in range(70)]
        self.assertEqual(list(hooks.samples(batch, Random(12), 70)), expected)
        self.assertEqual(batch.queries, single.queries)
        self.assertEqual(batch.compilations, 2)
        self.assertLessEqual(len(batch.plans), 4)

    def test_external_callbacks_guarded_noise_and_eviction_remain_dynamic(self):
        registry = ControllerRegistry(); calls = []
        def echo(bits, config):
            calls.append(bits); return bits
        registry.register('echo', echo, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CX ctrl[0] 1\nX_ERROR(1/2) ctrl[0] 1\nM 1', controllers=registry)
        kwargs = dict(task={'outputs': [1], 'ideal_state': ZERO_TARGET})
        cached = compile_circuit(circuit, **kwargs)
        evicted = compile_circuit(circuit, options={'max_plans': 1, 'max_queries': 10000}, **kwargs)
        actual = list(hooks.samples(cached, Random(5), 30))
        self.assertEqual(len(calls), 30)
        expected = list(hooks.samples(evicted, Random(5), 30))
        self.assertEqual(actual, expected)
        self.assertEqual(len(calls), 60)
        self.assertEqual(set(calls), {(0,), (1,)})
        self.assertLessEqual(len(cached.plans), 4)
        self.assertLessEqual(len(evicted.plans), 1)

    def test_equivalent_resolved_histories_share_only_preparation(self):
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        c = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CX ctrl[0] 1\nCZ rec[-1] 1\nZ_ERROR(1/4) 1', controllers=registry)
        compiled = compile_circuit(c, task={'outputs': [1]})
        target, = c.analysis.external_instructions[0].outputs
        empty = NoiseDraw({}, ())
        unresolved = hooks.engine(compiled, {}, empty)
        zero = hooks.engine(compiled, {target: 0}, empty)
        one = hooks.engine(compiled, {target: 1}, empty)
        self.assertIs(zero, unresolved)
        self.assertIsNot(one, zero)
        self.assertIs(one.engine, zero)
        self.assertEqual(compiled.compilations, 1)
        for engine, expected in ((zero, 1), (one, -1)):
            self.assertEqual(engine.query(engine.pack(), z=1).total().real_decimal(), expected)
            for bit in (0, 1):
                row, = engine.rows((('m0',),))
                self.assertAlmostEqual(engine.query(engine.pack(), event_rows=(row,), outcomes=(bit,)).total().real_decimal(), .5)
        # A new public budget is part of both cache identities.
        compiled.options['max_path_bits'] = 0
        with self.assertRaises(ResourceLimit): hooks.engine(compiled, {target: 0}, empty)

    def test_native_budget_identity_and_weak_structure_eviction(self):
        import gc
        import weakref
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        c = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\nCX ctrl[1] 1\nCS ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(c, task={'outputs': [1]}, options={'max_plans': 1})
        target, = c.analysis.external_instructions[0].outputs
        empty = NoiseDraw({}, ())
        zero = hooks.engine(compiled, {target: 0}, empty)
        zero_ref = weakref.ref(zero)
        compiled.options['max_path_bits'] = 8192
        one = hooks.engine(compiled, {target: 1}, empty)
        self.assertIsNot(one, zero)
        self.assertEqual(zero.query(zero.pack(), z=1).total().real_decimal(), 1)
        self.assertEqual(one.query(one.pack(), z=1).total().real_decimal(), -1)
        self.assertEqual(len(compiled._plan_structures), 1)
        del zero; gc.collect()
        self.assertIsNone(zero_ref())
        compiled.options.pop('max_path_bits')
        again = hooks.engine(compiled, {target: 0}, empty)
        self.assertEqual(again.query(again.pack(), z=1).total().real_decimal(), 1)
        self.assertEqual(compiled.compilations, 3)

    def test_batch_failure_keeps_completed_prefix_and_query_budget_contract(self):
        def failing_circuit():
            registry = ControllerRegistry(); calls = []
            def callback(bits, config):
                calls.append(bits)
                if len(calls) == 4: raise ValueError('declared callback failure')
                return bits
            registry.register('fail', callback, version='1')
            return parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[fail] det[-1] -> ctrl[0]\nCX ctrl[0] 1', controllers=registry)
        batch = run(failing_circuit(), shots=70, seed=8)
        direct = run(failing_circuit(), shots=70, seed=8, options={'max_queries': 10000})
        for key in ('status', 'completed', 'failed', 'not_attempted', 'failures', 'partial_statistics'):
            self.assertEqual(batch[key], direct[key])
        self.assertGreater(batch['completed'], 0)
        limited = run(parse_circuit('H 0\nM 0'), shots=3, options={'max_queries': 2})
        self.assertEqual((limited['completed'], limited['failed'], limited['not_attempted']), (2, 1, 0))


if __name__ == '__main__': unittest.main()
