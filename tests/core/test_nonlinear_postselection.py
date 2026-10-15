"""Nonlinear final events checked against independent finite truth tables."""
from tests.support.readout_target import ZERO_TARGET
from fractions import Fraction
from itertools import product
from random import Random
import unittest

import numpy as np

from magna import ControllerRegistry, compile_circuit, enumerate_errors, parse_circuit, run
from magna.circuit_file import CircuitSyntaxError
from magna.file_api import NoiseDraw
from tests.support import file_hooks as hooks


class NonlinearPostselectionTests(unittest.TestCase):
    def registry(self):
        registry = ControllerRegistry()
        registry.register('and', lambda bits, config: (bits[0] & bits[1],), version='1')
        registry.register('echo', lambda bits, config: bits, version='1')
        return registry

    def exact(self, text, task=None, **kwargs):
        result = enumerate_errors(parse_circuit(text, controllers=self.registry()),
                                  task=task, max_weight=0, **kwargs)
        self.assertEqual(result['status'], 'complete', result)
        return result['totals']

    def test_noncommuting_readout_truth_table_and_target(self):
        # The two successive measurements are Z then X on the same wire.
        # Their outcomes a,b are independent fair bits; output is |a>.
        text = ('H 0\nM 0\nH 0\nM 0\nCX rec[-2] 1\n'
                'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]')
        task = {'outputs': [1], 'observables': [[0, 1]]}
        totals = self.exact(text, task, ideal_state=parse_circuit(''))
        accepted = [(a, b) for a, b in product((0, 1), repeat=2) if not (a & b)]
        self.assertEqual(totals['acceptance'], len(accepted)/4)
        self.assertEqual(totals['error'], sum(a for a, b in accepted)/4)
        self.assertEqual(totals['moments'], [sum(1-2*a for a, b in accepted)/4])
        self.assertAlmostEqual(totals['conditional_error_within_enumerated_sector'], 1/3)

    def test_mixed_affine_selector_integrates_unobserved_record(self):
        text = ('H 0 1 2\nM 0 1 2\nCX rec[-1] 3\n'
                'DETECTOR rec[-3]\nDETECTOR rec[-2]\n'
                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\nPOSTSELECT ctrl[1]')
        circuit = parse_circuit(text, controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [3]}, ideal_state=parse_circuit(''))
        for seed in range(8):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw['acceptance'], .5)
            self.assertEqual(draw['sampled_parities'], 2)
        totals = self.exact(text, {'outputs': [3]}, ideal_state=parse_circuit(''))
        self.assertEqual(totals['acceptance'], .5)
        self.assertEqual(totals['error'], .125)
        self.assertEqual(totals['conditional_error_within_enumerated_sector'], .25)

    def test_dense_noncommuting_magic_instrument_with_report_noise(self):
        text = ('H 0\nT 0\nMX 0\nS 0\nT 0\nM(1/3) 0\nH 0\nT 0\nMY 0\n'
                'DETECTOR rec[-3]\nDETECTOR rec[-1]\n'
                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                'CONTROL[XOR] ctrl[0] rec[-2] -> ctrl[1]\n'
                'CONTROL[NOT] ctrl[1] -> ctrl[2]\nPOSTSELECT ctrl[2]')
        # Independent fixed-input branch amplitudes P_y T H P_z T S P_x T H|0>.
        h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        x = np.array([[0, 1], [1, 0]]); y = np.array([[0, -1j], [1j, 0]])
        z = np.diag([1, -1]); s = np.diag([1, 1j]); t = np.diag([1, np.exp(1j*np.pi/4)])
        def projector(p, bit): return (np.eye(2)+(1-2*bit)*p)/2
        expected_acceptance = expected_error = 0.
        for a, b, c, flip in product((0, 1), repeat=4):
            state = (projector(y, c) @ t @ h @ projector(z, b) @ t @ s @
                     projector(x, a) @ t @ h @ np.array([1, 0]))
            mass = np.vdot(state, state).real * (1/3 if flip else 2/3)
            if (a & c) ^ b ^ flip:
                expected_acceptance += mass
                expected_error += c*mass
        result = enumerate_errors(parse_circuit(text, controllers=self.registry()), max_weight=1,
                                  ideal_state=parse_circuit('H 0\nS 0'))
        self.assertEqual(result['status'], 'complete', result)
        self.assertAlmostEqual(result['totals']['acceptance'], expected_acceptance, places=12)
        self.assertAlmostEqual(result['totals']['error'], expected_error, places=12)

    def test_full_history_final_readout_and_explicit_state(self):
        calls = []
        registry = ControllerRegistry()
        def decode(bits, config):
            calls.append(bits)
            a, b, final, prior = bits
            return ((a & b) ^ final ^ prior,)
        registry.register('decode', decode, version='1')
        text = ('CONTROL[XOR](1) -> ctrl[0]\nX 0\nMR 0\nDETECTOR rec[-1]\n'
                'X 0\nMR 0\nDETECTOR rec[-1]\nX 0\nM(1) 0\n'
                'DETECTOR rec[-1]\n'
                'CONTROL[decode] det[0] det[1] det[-1] ctrl[0] -> ctrl[0]\n'
                'POSTSELECT ctrl[0]')
        result = run(parse_circuit(text, controllers=registry), shots=3, task={'outputs': []})
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['estimates']['acceptance'], 1)
        self.assertEqual(calls, [(1, 1, 0, 1)]*3)

    def test_report_flips_select_reported_bits_and_preserve_true_output(self):
        text = ('X 0 1\nM(1) 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]')
        circuit = parse_circuit(text, controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [0], 'sample_records': [0, 1]},
                                   ideal_state=parse_circuit(''))
        draw = compiled.sample(Random(1))
        self.assertEqual(draw['sampled_records'], [0, 1])
        self.assertEqual((draw['acceptance'], draw['error']), (1, 1))
        self.assertEqual(compiled.noise_schedule.category_count, 1)
        mixed = text.replace('POSTSELECT ctrl[0]',
            'CONTROL[XOR] ctrl[0] rec[-2] -> ctrl[1]\nPOSTSELECT ctrl[1]')
        self.assertEqual(run(parse_circuit(mixed, controllers=self.registry()))['estimates']['acceptance'], 1)

    def test_versions_overwrites_repeat_and_explicit_prior_decisions(self):
        text = ('CONTROL[XOR](0) -> ctrl[0]\nREPEAT 2 {\nR 0\nH 0\nM 0\n'
                'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]\n}\n'
                'CONTROL[XOR](1) -> ctrl[0]')
        circuit = parse_circuit(text, controllers=self.registry())
        selected = [i.targets[0] for i in circuit.instructions if i.name == 'POSTSELECT']
        self.assertEqual([t.version for t in selected], [1, 2])
        self.assertEqual(self.exact(text)['acceptance'], .25)
        prior = ('H 0 1\nM 0 1\nDETECTOR rec[-2]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
                 'DETECTOR rec[-1]\nCONTROL[and] ctrl[0] det[-1] -> ctrl[0]\n'
                 'CONTROL[NOT] ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]\n'
                 'CONTROL[XOR](0) -> ctrl[0]')
        self.assertEqual(self.exact(prior)['acceptance'], .25)

    def test_joint_duplicate_conflicting_detector_and_task_conditions(self):
        prefix = ('H 0 1\nM 0 1\nDETECTOR rec[-2]\n'
                  'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                  'CONTROL[echo] det[-2] det[-1] -> ctrl[0] ctrl[1]\n')
        for select, task, expected in (
            ('POSTSELECT ctrl[0] ctrl[1]', {}, .25),
            ('CONTROL[NOT] ctrl[0] -> ctrl[2]\nCONTROL[NOT] ctrl[1] -> ctrl[3]\n'
             'POSTSELECT ctrl[2] ctrl[3]', {}, .25),
            ('POSTSELECT ctrl[0] ctrl[0]', {}, .5),
            ('POSTSELECT ctrl[0]\nCONTROL[NOT] ctrl[0] -> ctrl[2]\nPOSTSELECT ctrl[2]', {}, 0),
            ('POSTSELECT ctrl[1] det[0]', {}, .25),
            ('POSTSELECT ctrl[0]\nPOSTSELECT !det[0]', {}, 0),
            ('POSTSELECT ctrl[0]\nDETECTOR rec[-2] rec[-1]\nPOSTSELECT !det[3]', {}, .25),
        ):
            with self.subTest(select=select, task=task):
                self.assertEqual(self.exact(prefix+select, task)['acceptance'], expected)

    def test_exact_symbol_cancellation_keeps_affine_path(self):
        text = ('H 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                'CONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\n'
                'CONTROL[XOR] ctrl[0] ctrl[1] -> ctrl[2]\nPOSTSELECT ctrl[2]')
        circuit = parse_circuit(text, controllers=self.registry())
        compiled = compile_circuit(circuit)
        self.assertFalse(hooks.feedback_template(compiled).selections)
        self.assertEqual(compiled.task['postselection'], (('m1',),))
        self.assertEqual(self.exact(text)['acceptance'], .5)
        affine = compile_circuit(parse_circuit('H 0\nM 0\nCONTROL[XOR](1) rec[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]'))
        self.assertFalse(hooks.feedback_template(affine).selections)
        self.assertEqual(affine.sample(Random(1))['acceptance'], .5)
        # Identical opaque calls remain distinct versioned symbols.
        independent = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[1]\nCONTROL[XOR] ctrl[0] ctrl[1] -> ctrl[2]\n'
            'POSTSELECT ctrl[2]', controllers=self.registry())
        self.assertEqual(len(hooks.feedback_template(compile_circuit(independent)).selections), 1)

    def test_fixed_template_assignments_and_private_boundaries(self):
        circuit = parse_circuit('H 0 1 2\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
            'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nCH ctrl[0] 2\nCONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\nPOSTSELECT ctrl[1]',
            controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [2]}, ideal_state=parse_circuit('H 0'))
        term, = circuit.analysis.external_instructions[0].outputs
        noise = NoiseDraw({}, ())
        baseline = hooks.engine(compiled, {}, noise)
        feedback_wire = next(iter(hooks.feedback_template(compiled).virtual.values()))[0]
        selection_wire, _, record, _ = next(iter(hooks.feedback_template(compiled).selections.values()))
        self.assertNotEqual(feedback_wire, selection_wire)
        self.assertEqual(circuit.num_records, 2)
        self.assertEqual(circuit.num_qubits, 3)
        self.assertEqual(compiled.noise_schedule.category_count, 0)
        operations = hooks.lower(circuit, (2,), {}, noise,
                            feedback_plan=hooks.feedback_template(compiled)).operations
        self.assertFalse(any(op.get('record') == record and op['qubits'] == [feedback_wire] for op in operations))
        self.assertTrue(any(op.get('record') == record and op['qubits'] == [selection_wire] for op in operations))
        for values in ({}, {term: 0}, {term: 1}):
            engine = hooks.engine(compiled, values, noise)
            self.assertIs(engine.handle, baseline.handle)
            self.assertEqual(engine.query(engine.pack()).total().real_decimal(), 1)
            for a, b in product((0, 1), repeat=2):
                rows = engine.rows((('m0',), ('m1',), (record,)))
                for selected in (0, 1):
                    mass = engine.query(engine.pack(), event_rows=rows, outcomes=(a, b, selected)).total()
                    self.assertEqual(float(mass.real_decimal()), .25 if selected == (values.get(term, 0) ^ b) else 0)
        self.assertEqual(compiled.compilations, 2)

    def test_no_future_selection_bias_and_replay_with_recovery(self):
        seen = []
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: seen.append(bits) or bits, version='1')
        prefix = ('H 0 1\nM(1/4) 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
                  'CH ctrl[0] 1\n')
        suffix = 'M 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[1]\n'
        recovery = parse_circuit('H 1\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCH ctrl[0] 0',
                                 controllers=registry)
        task = {'outputs': [1], 'sample_records': [0, 1]}
        def draws(selection):
            seen.clear()
            compiled = compile_circuit(parse_circuit(prefix+selection+suffix, controllers=registry),
                task=task, recovery=recovery, ideal_state=parse_circuit(''))
            rng = Random(71)
            result = [compiled.sample(rng) for _ in range(32)]
            return result, seen[:]
        original, observations = draws('')
        selected, actual = draws('POSTSELECT ctrl[0]\n')
        replay, repeated = draws('POSTSELECT ctrl[0]\n')
        self.assertEqual(actual, observations)
        self.assertEqual(actual, repeated)
        self.assertEqual(selected, replay)
        for before, after in zip(original, selected):
            self.assertEqual(before['sampled_records'], after['sampled_records'])
            self.assertEqual(before['exact']['history_mass'], after['exact']['history_mass'])
            self.assertEqual(after['acceptance'], 1-after['sampled_records'][0])
        self.assertEqual({d['acceptance'] for d in selected}, {0, 1})

    def test_selection_distinct_from_each_error_definition(self):
        prefix = ('H 0 1\nM 0 1\nCX rec[-2] 2\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                  'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]')
        # The decoder reads record 0 through a detector and reports the error bit.
        decoded = '\nDETECTOR rec[-2]\nCONTROL[echo] det[-1] -> ctrl[1]\nOBSERVABLE_INCLUDE(0) ctrl[1]'
        result = self.exact(prefix+decoded, {'error_observables': [0]})
        self.assertEqual((result['acceptance'], result['error']), (.75, .25))
        result = self.exact(prefix, {'outputs': [2]}, ideal_state=parse_circuit('I 0'))
        self.assertEqual((result['acceptance'], result['error']), (.75, .25))

    def test_zero_acceptance_and_callback_failure_are_not_filtered(self):
        text = 'X 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]'
        result = run(parse_circuit(text, controllers=self.registry()), shots=5, task={'outputs': [0], 'ideal_state': ZERO_TARGET})
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['completed'], 5)
        self.assertEqual(result['estimates'], {'acceptance': 0., 'error': 0., 'conditional_error': None, 'moments': []})
        registry = self.registry()
        def fail(bits, config):
            raise RuntimeError('selector unavailable')
        registry.register('fail', fail, version='1')
        failed = run(parse_circuit(text+'\nCONTROL[fail] ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]', controllers=registry), shots=2)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['completed'], 0)
        self.assertIn('selector unavailable', str(failed['failures']))
        for value in ((2,), (), (0, 1)):
            bad = ControllerRegistry()
            bad.register('bad', lambda bits, config: value, version='1')
            result = run(parse_circuit('CONTROL[bad] -> ctrl[0]\nPOSTSELECT ctrl[0]', controllers=bad))
            self.assertEqual(result['status'], 'failed')

    def test_importance_weights_rejected_draws_and_aggregate_ratio(self):
        # Noise bits a,b are independent, with P(a)=1/10 and P(b)=1/5.
        # Accept unless a AND b; output is a. Scale two gives Q(a)=1/5,
        # Q(b)=2/5; multiplicities below realize its exact 25-draw mixture.
        circuit = parse_circuit('X_ERROR(1/10) 0\nX_ERROR(1/5) 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
            'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]', controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [0]}, ideal_state=parse_circuit(''),
            sampling={'method': 'importance', 'proposal_scale': '2'})
        sites = [i.op_id for i in circuit.instructions if i.name == 'X_ERROR']
        draws = []
        for bits in product((0, 1), repeat=2):
            weight = proposal_mass = Fraction(1); gates = {}
            for j, (bit, p) in enumerate(zip(bits, (Fraction(1, 10), Fraction(1, 5)))):
                q = 2*p
                weight *= p/q if bit else (1-p)/(1-q)
                proposal_mass *= q if bit else 1-q
                if bit: gates[sites[j]] = (dict(kind='X', qubits=[j]),)
            draw = compiled.sample(Random(1), noise=NoiseDraw(gates, (), (weight.numerator, weight.denominator)))
            self.assertEqual(draw['exact']['noise_likelihood_ratio_hex'], [hex(weight.numerator), hex(weight.denominator)])
            self.assertAlmostEqual(draw['acceptance'], float(weight)*(1-(bits[0] & bits[1])))
            copies = 25*proposal_mass
            self.assertEqual(copies.denominator, 1)
            draws.extend([draw]*copies.numerator)
        estimates = hooks.summary(draws)['estimates']
        self.assertAlmostEqual(estimates['acceptance'], .98)
        self.assertAlmostEqual(estimates['error'], .08)
        self.assertAlmostEqual(estimates['conditional_error'], 4/49)
        self.assertEqual(draws[-1]['acceptance'], 0)
        self.assertEqual(len(draws), 25)
        self.assertEqual(sum(draw['acceptance'] == 0 for draw in draws), 2)

    def test_validation_format_version_and_module_boundaries(self):
        for text in ('POSTSELECT ctrl[0]',
                     'CONTROL[XOR](0) -> ctrl[0]\nPOSTSELECT(0) ctrl[0]',
                     'CONTROL[XOR](0) -> ctrl[0]\nPOSTSELECT(1) ctrl[0]',
                     'CONTROL[XOR](0) -> ctrl[0]\nPOSTSELECT(2) ctrl[0]'):
            with self.assertRaises(CircuitSyntaxError): parse_circuit(text)
        circuit = parse_circuit('CONTROL[XOR](0) -> ctrl[0]\nPOSTSELECT ctrl[0]')
        self.assertEqual(circuit.format_version, 'magna-0.1')
        self.assertEqual(parse_circuit(circuit.to_text()).fingerprint, circuit.fingerprint)
        for kwargs, message in (({'ideal_state': circuit}, 'unitary'),
                                ({'ideal_state': parse_circuit(''), 'recovery': circuit}, 'physical circuit')):
            with self.assertRaisesRegex(ValueError, message): compile_circuit(parse_circuit('I 0'), **kwargs)

    def test_not_on_accept_bit_matches_truth_table_and_preserves_feedback(self):
        prefix = ('H 0 1 2\nT 2\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                  'CONTROL[and] det[-2] det[-1] -> ctrl[0]\n')
        inverse = ('CONTROL[NOT] ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]\n'
                   'CH ctrl[1] 2\nCONTROL[NOT] ctrl[1] -> ctrl[2]\n'
                   'CONTROL[XOR] ctrl[0] ctrl[2] rec[-1] -> ctrl[3]\n'
                   'CONTROL[NOT] ctrl[3] -> ctrl[4]\nPOSTSELECT ctrl[4]')
        reference = 'CH !ctrl[0] 2\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nPOSTSELECT !det[2]\nPOSTSELECT !det[3]'
        task = {'outputs': [2], 'observables': [[1, 0], [0, 1]]}
        self.assertEqual(self.exact(prefix+inverse, task), self.exact(prefix+reference, task))
        self.assertEqual(self.exact(prefix+inverse, task)['acceptance'], .25)
        compiled = compile_circuit(parse_circuit(prefix+inverse, controllers=self.registry()), task=task)
        self.assertEqual(len(compiled._control_metadata.callbacks), 1)
        self.assertEqual(len(hooks.feedback_template(compiled).selections), 1)
        for seed in range(8):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw['decoder_calls'], 1)
        self.assertEqual(compiled.compilations, 1)

    def test_affine_not_selection_and_terminal_feedback(self):
        circuit = parse_circuit('M(1) 0\nCONTROL[NOT] rec[-1] -> ctrl[0]\nPOSTSELECT ctrl[0]')
        compiled = compile_circuit(circuit, ideal_state=parse_circuit('X 0'),
            recovery=parse_circuit('M 0\nCONTROL[NOT] rec[-1] -> ctrl[0]\nCX ctrl[0] 0'))
        self.assertFalse(hooks.feedback_template(compiled).selections)
        draw = compiled.sample(Random(1))
        self.assertEqual((draw['acceptance'], draw['error']), (1, 0))
        self.assertEqual((draw['decoder_calls'], draw['recovery_decoder_calls']), (0, 0))

    def test_not_and_xor_constant_use_zero_projection_template(self):
        prefix = 'M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
        for function in ('CONTROL[NOT]', 'CONTROL[XOR](1)'):
            suffix = f'{function} ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]'
            circuit = parse_circuit(prefix+suffix, controllers=self.registry())
            compiled = compile_circuit(circuit)
            term, = circuit.analysis.external_instructions[0].outputs
            q, _, record, reference = next(iter(hooks.feedback_template(compiled).selections.values()))
            self.assertEqual(reference, 0)
            noise = NoiseDraw({}, ())
            baseline = hooks.base_engine(compiled, {}, noise, stage='protocol', recovery_values={})
            rows = baseline.rows(((record,),))
            self.assertEqual(baseline.query(baseline.pack(), event_rows=rows, outcomes=(0,)).total().real_decimal(), 1)
            for bit in (0, 1):
                engine = hooks.engine(compiled, {term: bit}, noise)
                self.assertIs(engine.handle, baseline.handle)
                self.assertEqual(engine.query(engine.pack(), event_rows=rows, outcomes=(0,)).total().real_decimal(), bit)


if __name__ == '__main__':
    unittest.main()
