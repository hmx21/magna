"""Fixed virtual-control templates versus literal branches and dense matrices."""
from fractions import Fraction
from itertools import product
from random import Random
import unittest

import numpy as np

from magna import ControllerRegistry, compile_circuit, parse_circuit, enumerate_errors
from magna.file_api import NoiseDraw
from test_pauli_feedback_reuse import literal_engine
from tests.support import file_hooks as hooks


class VirtualCliffordFeedbackTests(unittest.TestCase):
    def registry(self):
        registry = ControllerRegistry()
        registry.register('and', lambda bits, config: (bits[0] & bits[1],), version='1')
        registry.register('echo', lambda bits, config: bits, version='1')
        return registry

    def test_two_qubit_clifford_controls_match_dense_states_and_reuse_template(self):
        h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        t = np.diag([1, np.exp(1j*np.pi/4)]); s = t@t
        x = np.array([[0, 1], [1, 0]]); z = np.diag([1, -1])
        paulis = (np.eye(2), x, z, 1j*x@z)
        initial = np.kron(s@t@h@np.array([1, 0]), t@h@np.array([1, 0]))
        cx = np.zeros((4, 4))
        for a in range(4): cx[a ^ ((a & 1) << 1), a] = 1
        for gate, matrix in (('CCZ', np.diag([1, 1, 1, -1])), ('CCX', cx)):
            for mode in ('direct', 'xor', 'linear'):
                text = 'H 0 1 2 3\nT 2 3\nS 3\nM 0 1\n'
                if mode != 'linear':
                    text += 'DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                if mode == 'xor': text += 'CONTROL[XOR] !ctrl[0] rec[-1] -> ctrl[1]\n'
                control = {'direct': 'ctrl[0]', 'xor': 'ctrl[1]', 'linear': '!rec[-1]'}[mode]
                circuit = parse_circuit(text+f'{gate} {control} 2 3', controllers=self.registry())
                compiled = compile_circuit(circuit, task={'outputs': [2, 3]})
                term = circuit.analysis.external_instructions[0].outputs[0] if mode != 'linear' else None
                baseline = hooks.engine(compiled, {}, NoiseDraw({}, ()))
                for b in (0, 1):
                    actual = hooks.engine(compiled, {term: b} if term else {}, NoiseDraw({}, ()))
                    self.assertIs(actual.handle, baseline.handle)
                    for r0, r1 in product((0, 1), repeat=2):
                        active = b if mode == 'direct' else 1 ^ r1 ^ (b if mode == 'xor' else 0)
                        state = (matrix if active else np.eye(4))@initial
                        for p0, p1 in product(range(4), repeat=2):
                            xm = (p0 & 1) | ((p1 & 1) << 1)
                            zm = (p0 >> 1) | ((p1 >> 1) << 1)
                            value = actual.query(actual.pack(), event_rows=actual.rows((('m0',), ('m1',))),
                                outcomes=(r0, r1), x=xm, z=zm).total().real_decimal()
                            expected = np.vdot(state, np.kron(paulis[p1], paulis[p0])@state).real/4
                            self.assertAlmostEqual(float(value), expected, places=12)
                self.assertEqual(compiled.compilations, 1)
                self.assertEqual(bool(hooks.feedback_template(compiled).virtual), mode != 'linear')

    def test_composed_control_simplification_and_versions_determine_route(self):
        circuit = parse_circuit('H 0 1 2\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
            'CONTROL[XOR] !ctrl[0] rec[-1] -> ctrl[1]\n'
            'CONTROL[XOR] ctrl[1] -> ctrl[2]\n'
            'CONTROL[XOR] ctrl[2] ctrl[2] -> ctrl[3]\n'
            'CONTROL[XOR] ctrl[2] ctrl[2] rec[-1] -> ctrl[4]\n'
            'CCZ ctrl[2] 1 2\nCCX ctrl[3] 1 2\nCCZ ctrl[4] 1 2\n'
            'CONTROL[XOR] ctrl[0] -> ctrl[0]\nCH ctrl[0] 2\n'
            'CONTROL[XOR] rec[-1] -> ctrl[0]\nCS ctrl[0] 2', controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [1, 2]})
        # Constant and affine composite functions need no virtual control,
        # even when their expressions originally referenced a nonlinear bit.
        # A fresh affine overwrite is a distinct version and stays linear.
        self.assertEqual(len(hooks.feedback_template(compiled).virtual), 2)
        term, = circuit.analysis.external_instructions[0].outputs
        for bit in (0, 1):
            actual = hooks.engine(compiled, {term: bit}, NoiseDraw({}, ()))
            expected = literal_engine(compiled, {term: bit}, NoiseDraw({}, ()))
            for x, z in product(range(4), repeat=2):
                self.assertEqual(actual.query(actual.pack(), x=x, z=z).total(),
                                 expected.query(expected.pack(), x=x, z=z).total())
        self.assertEqual(compiled.compilations, 1)

    def test_direct_and_xor_cliffords_keep_one_template_and_match_dense_states(self):
        h = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        s = np.diag([1, 1j]); t = np.diag([1, np.exp(1j*np.pi/4)])
        x = np.array([[0, 1], [1, 0]]); z = np.diag([1, -1]); y = 1j*x@z
        for gate, matrix in (('CS', s), ('CS_DAG', s.conj().T), ('CH', h)):
            for mixed in (False, True):
                text = 'H 0 1 2\nT 2\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                text += 'CONTROL[XOR] !ctrl[0] rec[-1] -> ctrl[1]\n' if mixed else ''
                text += f'{gate} ctrl[{int(mixed)}] 2\nT 2'
                circuit = parse_circuit(text, controllers=self.registry())
                compiled = compile_circuit(circuit, task={'outputs': [2]})
                term, = circuit.analysis.external_instructions[0].outputs
                baseline = hooks.engine(compiled, {}, NoiseDraw({}, ()))
                virtual, = hooks.feedback_template(compiled).virtual.values()
                q = virtual[0]
                self.assertNotIn(q, baseline.deferred.ir.outputs)
                self.assertTrue(any(o['kind'] == 'DISCARD' and o['qubits'] == [q]
                                    for o in baseline.deferred.ir.operations))
                for b in (0, 1):
                    engine = hooks.engine(compiled, {term: b}, NoiseDraw({}, ()))
                    self.assertIs(engine.handle, baseline.handle)
                    self.assertEqual(engine.deferred.ir.fingerprint, baseline.deferred.ir.fingerprint)
                    for r0, r1 in product((0, 1), repeat=2):
                        active = b ^ (1 ^ r1 if mixed else 0)
                        state = t @ (matrix if active else np.eye(2)) @ t @ h @ np.array([1, 0])
                        rows = engine.rows((('m0',), ('m1',)))
                        for (xm, zm), observable in (((0, 0), np.eye(2)), ((1, 0), x), ((0, 1), z), ((1, 1), y)):
                            actual = engine.query(engine.pack(), event_rows=rows,
                                outcomes=(r0, r1), x=xm, z=zm).total().real_decimal()
                            expected = float(np.vdot(state, observable @ state).real)/4
                            self.assertAlmostEqual(float(actual), expected, places=12)
                self.assertEqual(compiled.compilations, 1)

    def test_virtual_trace_future_completion_reports_faults_and_reset_match_literal(self):
        circuit = parse_circuit('H 0 1 2\nT 2\nM(1/3) 0 1\n'
            'DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
            'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\nCS ctrl[1] 2\n'
            'Y_ERROR(1/3) 2\nH 2\nM 2\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[2]\n'
            'CH ctrl[2] 2\nR 2\nH 2\nCS_DAG ctrl[0] 2', controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [2]})
        terms = tuple(t for inst in circuit.analysis.external_instructions for t in inst.outputs)
        oid = next(i.op_id for i in circuit.instructions if i.name == 'Y_ERROR')
        for noisy in (False, True):
            noise = NoiseDraw({oid: ({'kind': 'Y', 'qubits': [2]},)} if noisy else {},
                              ('m0', 'm2') if noisy else ())
            for word in product((0, 1), repeat=2):
                values = dict(zip(terms, word))
                actual = hooks.engine(compiled, values, noise)
                expected = literal_engine(compiled, values, noise)
                for outcomes in product((0, 1), repeat=3):
                    names = (('m0',), ('m1',), ('m2',))
                    for x, z in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        self.assertEqual(actual.query(hooks.pack(compiled, actual, noise),
                            event_rows=actual.rows(names), outcomes=outcomes, x=x, z=z).total(),
                            expected.query(hooks.pack(compiled, expected, noise),
                            event_rows=expected.rows(names), outcomes=outcomes, x=x, z=z).total())
            for values, names in (({}, (('m0',), ('m1',))), ({terms[0]: 1}, (('m0',), ('m1',), ('m2',)))):
                actual = hooks.engine(compiled, values, noise); expected = literal_engine(compiled, values, noise)
                for outcomes in product((0, 1), repeat=len(names)):
                    self.assertEqual(actual.query(hooks.pack(compiled, actual, noise),
                        event_rows=actual.rows(names), outcomes=outcomes).total(),
                        expected.query(hooks.pack(compiled, expected, noise),
                        event_rows=expected.rows(names), outcomes=outcomes).total())
        self.assertEqual(compiled.compilations, 1)

    def test_protocol_and_terminal_virtual_wires_do_not_alias_outputs_or_each_other(self):
        registry = self.registry()
        circuit = parse_circuit('H 0 1 2\nT 2\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
                                'CH ctrl[0] 2', controllers=registry)
        # The recovery mentions only wire 0 but has a two-wire output boundary.
        recovery = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
                                 'CS ctrl[0] 0\nCH ctrl[0] 0\nCCZ ctrl[0] 0 1\nCCX ctrl[0] 1 0', controllers=registry)
        compiled = compile_circuit(circuit, task={'outputs': [2, 0], 'sample_records': [0]},
            recovery=recovery, ideal_state=parse_circuit('H 0 1\nT 0'))
        noise = NoiseDraw({}, ())
        a, = circuit.analysis.external_instructions[0].outputs
        b, = recovery.analysis.external_instructions[0].outputs
        for v, w in product((0, 1), repeat=2):
            for stage in ('recovery', 'target'):
                actual = hooks.engine(compiled, {a: v}, noise, stage=stage, recovery_values={b: w})
                expected = literal_engine(compiled, {a: v}, noise, stage=stage, recovery_values={b: w})
                for x, z in ((0, 0), (1, 0), (0, 2), (2, 1)):
                    self.assertEqual(actual.query(actual.pack(), x=x, z=z).total(),
                                     expected.query(expected.pack(), x=x, z=z).total())
        self.assertEqual(compiled.compilations, 2)
        for seed in range(8):
            self.assertEqual(compiled.sample(Random(seed))['recovery_decoder_calls'], 1)
        self.assertEqual(compiled.compilations, 2)

    def test_guarded_pauli_noise_uses_causal_assignments_only(self):
        circuit = parse_circuit('H 0 1\nT 1\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\n'
            'DEPOLARIZE1(1/3) ctrl[1] 1 1\nT 1\nH 1', controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [1], 'observables': [[1, 0], [0, 1], [1, 1]]})
        oid = next(i.op_id for i in circuit.instructions if i.name == 'DEPOLARIZE1')
        candidate = NoiseDraw({oid: ({'kind': 'X', 'qubits': [1]}, {'kind': 'Z', 'qubits': [1]})}, ('m0',))
        # echo(reported) XOR reported is zero on every actual history, including
        # report flips. Supplied multi-action candidates must therefore be inactive.
        for seed in range(8):
            self.assertEqual(compiled.sample(Random(seed), noise=candidate),
                             compiled.sample(Random(seed), noise=NoiseDraw({}, ('m0',))))
        self.assertEqual(compiled.compilations, 1)
        self.assertEqual(compiled._execution.state()['reference_network_preparations'], 2)

    def test_linear_clifford_feedback_stays_coherent_and_enumeration_preserves_mass(self):
        linear = compile_circuit(parse_circuit('H 0 1\nM 0\nCS rec[-1] 1\nCH rec[-1] 1'))
        self.assertFalse(hooks.feedback_template(linear).virtual)
        self.assertFalse(hooks.feedback_template(linear).updates)
        for gate in ('CS', 'CS_DAG', 'CH'):
            circuit = parse_circuit('H 0 1 2\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
                'CONTROL[XOR] ctrl[0] rec[-2] -> ctrl[1]\n'+gate+' ctrl[1] 2', controllers=self.registry())
            result = enumerate_errors(circuit, max_weight=0, task={'outputs': [2]},
                                      ideal_state=parse_circuit('H 0'))
            self.assertEqual(result['status'], 'complete', result)
            self.assertEqual(result['totals']['covered_probability'], 1)
            self.assertEqual(result['totals']['error'], float(Fraction(1, 8)))


if __name__ == '__main__':
    unittest.main()
