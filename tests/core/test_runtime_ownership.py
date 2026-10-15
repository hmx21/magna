"""Protocol-independent structural ownership and per-invocation assignment checks."""
from concurrent.futures import ThreadPoolExecutor
from random import Random
import gc
import unittest
from magna import _native, compile_circuit, parse_circuit, ControllerRegistry
from magna.file_api import NoiseDraw
from tests.support import file_hooks as hooks
from tests.support import native_testing
from tests.support.operation_hooks import total_plans


class RuntimeOwnershipTests(unittest.TestCase):
    def test_execution_and_noise_schedule_have_independent_owner_lifetimes(self):
        circuit = parse_circuit('H 0\nZ_ERROR(1/7) 0\nM 0')
        compiled = compile_circuit(circuit, sampling={'method': 'importance', 'proposal_scale': '2'})
        execution, schedule = compiled._execution, compiled.noise_schedule
        expected = [execution.sample(Random(seed)) for seed in range(12)]
        expected_noise = hooks.draw_many(schedule, Random(391), 12)
        del compiled, circuit;gc.collect()
        self.assertEqual([execution.sample(Random(seed)) for seed in range(12)], expected)
        del schedule;gc.collect()
        self.assertEqual([execution.sample(Random(seed)) for seed in range(12)], expected)
        # Reversing destruction order keeps the public schedule independently usable.
        compiled = compile_circuit(parse_circuit('H 0\nZ_ERROR(1/7) 0\nM 0'),
                                   sampling={'method': 'importance', 'proposal_scale': '2'})
        schedule = compiled.noise_schedule
        del compiled;gc.collect()
        actual_noise = hooks.draw_many(schedule, Random(391), 12)
        self.assertEqual([(d.gates, d.report_flips, d.likelihood_ratio) for d in actual_noise],
                         [(d.gates, d.report_flips, d.likelihood_ratio) for d in expected_noise])

    def test_structure_reuse_changed_rhs_faults_effects_and_uncached(self):
        c = compile_circuit(parse_circuit('H 0\nCX 0 1\nM(1/3) 1'))
        e = hooks.engine(c, {}, NoiseDraw({}, ()))
        events = e.rows((('m0',),))
        for flip in ((), ('m0',)):
            packed = e.pack(report_flips=flip)
            for bit in (0, 1):
                queries = [(events, (bit,), x, z) for x, z in ((0, 0), (0, 1))]
                actual, _, _ = e.totals([packed], queries)
                self.assertEqual(actual[0], [e.query(packed, event_rows=a, outcomes=b, x=x, z=z).total()
                                             for a, b, x, z in queries])
        self.assertEqual(total_plans(e), 1)
        e.totals([e.pack()], [(events, (0,), 1, 0)])
        self.assertEqual(total_plans(e), 2)
        for queries in ([(events, (), 0, 0)], [(events, (2,), 0, 0)]):
            with self.assertRaises(ValueError): e.totals([e.pack()], queries)

    def test_scalar_totals_are_thread_safe_and_reject_bad_outcomes(self):
        c = compile_circuit(parse_circuit('H 0\nM 0'))
        e = hooks.engine(c, {}, NoiseDraw({}, ()))
        events, packed = e.rows((('m0',),)), e.pack()
        def call(bit): return e.totals([packed], [(events, (bit,), 0, 0)])[0][0][0]
        expected = [call(0), call(1)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            self.assertEqual(list(pool.map(call, [0, 1]*12)), expected*12)
        for outcomes in ((), (2,), (0, 1)):
            with self.assertRaises(ValueError): e.totals([packed], [(events, outcomes, 0, 0)])

    def test_cache_bound_and_changed_event_structure(self):
        c = compile_circuit(parse_circuit('H 0\nM 0'))
        e = hooks.engine(c, {}, NoiseDraw({}, ()))
        for n in range(40):
            e.totals([e.pack()], [((0,)*n, (0,)*n, 0, 0)])
        self.assertEqual(total_plans(e), 32)
        e.totals([e.pack()], [((0,), (0,), 0, 0)])
        for malformed in ([((False,), (0,), 0, 0)], [((0,), (0,), False, 0)]):
            with self.assertRaises(ValueError): e.totals([e.pack()], malformed)

    def test_sparse_noise_equals_dense_categories_and_isolation(self):
        from magna.file_api import _NoiseSchedule
        c = parse_circuit('DEPOLARIZE2(1/3) 0 1\nCORRELATED_ERROR(1/5) X0 Z1\n'
                          'ELSE_CORRELATED_ERROR(1/7) Y0\nM(1/4) 0 1')
        schedule = _NoiseSchedule(c, {'method': 'importance', 'proposal_scale': '3/2'})
        rng = Random(45); seeds = [rng.getrandbits(64) for _ in range(100)]
        dense = native_testing.load().draw_file_noise(schedule._native_plan, seeds)
        table = schedule.actions
        actual = hooks.draw_many(schedule, Random(45), 100)
        for draw, (choices, weight) in zip(actual, dense):
            gates = {}; reports = []
            for group, choice in zip(table, choices):
                if choice < 0: continue
                oid, value = group[choice]
                if oid == 'report': reports.append(value)
                else: gates[oid] = gates.get(oid, ()) + value
            self.assertEqual((draw.gates, draw.report_flips, draw.likelihood_ratio),
                             (gates, tuple(reports), weight))
        for group in table:
            for oid, value in group:
                if oid != 'report' and value: value[0]['qubits'].clear()
        self.assertEqual(hooks.draw_many(schedule, Random(45), 100), actual)
        # Direct native table snapshots input and validates action/channel shape.
        actions = [[['a', [['Y', 0]]]]]
        cap = native_testing.load().make_file_noise([[(1, 1)]], None, actions)
        actions[0][0][1].clear()
        self.assertEqual(native_testing.load().draw_file_noise_actions(cap, [7])[0][0], (('a', (('Y', 0),)),))
        for bad in ([], [[('a', [('Q', 0)])]], [[('a', [('X', -1)])]]):
            with self.assertRaises(ValueError): native_testing.load().make_file_noise([[(1, 1)]], None, bad)

    def test_reentrant_callbacks_and_exception_leave_fresh_control_values(self):
        registry = ControllerRegistry(); holder = {}; calls = []; nested = []; active = False; fail = False
        def callback(bits, config):
            nonlocal active
            calls.append(bits)
            if fail: raise RuntimeError('intentional decoder failure')
            if not active and holder.get('nested', False):
                active = True
                try: nested.append(holder['compiled'].sample(Random(19)))
                finally: active = False
            return bits
        registry.register('echo', callback, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
                                'CX ctrl[0] 1\nM 1', controllers=registry)
        compiled = compile_circuit(circuit, task={'sample_records': [0, 1]})
        holder.update(compiled=compiled, nested=True)
        rng = Random(6); actual = compiled.sample(rng); state = rng.getstate()
        holder['nested'] = False
        reference_rng = Random(6)
        self.assertEqual(actual, compiled.sample(reference_rng))
        self.assertEqual(state, reference_rng.getstate())
        self.assertEqual(nested[0], compiled.sample(Random(19)))
        fail = True
        with self.assertRaisesRegex(ValueError, 'intentional decoder failure'): compiled.sample(Random(6))
        fail = False
        self.assertEqual(actual, compiled.sample(Random(6)))
        self.assertEqual(len(calls), 6)


if __name__ == '__main__': unittest.main()
