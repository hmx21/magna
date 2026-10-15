"""Bounded file-facade semantics against independent gates and exact events."""
from tests.support.readout_target import ZERO_TARGET
from fractions import Fraction
import math
from random import Random
import unittest

import numpy as np

from magna.circuit_file import ControllerRegistry, parse_circuit
from magna.file_api import compile_circuit, run
from tests.support import file_hooks as hooks
from tests.support import native_testing

try:
    import stim
except ImportError:
    stim = None
try:
    from magna import _native
    HAS_NATIVE = hasattr(_native, "make_boundary_trace")
except ImportError:
    HAS_NATIVE = False


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])
H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * math.pi / 4)])


def controlled(matrix):
    """Control on low bit, target on next bit (Stim little-endian order)."""
    out = np.eye(4, dtype=complex)
    out[np.ix_([1, 3], [1, 3])] = matrix
    return out


def moments(vector):
    n = int(math.log2(len(vector)))
    values = []
    masks = []
    for x in range(1 << n):
        for z in range(1 << n):
            if not x and not z:
                continue
            value = sum(np.conj(vector[k ^ x]) * vector[k]
                        * (-1) ** ((k & z).bit_count())
                        * 1j ** ((x & z).bit_count()) for k in range(len(vector)))
            values.append(value.real)
            masks.append([x, z])
    return masks, values


@unittest.skipUnless(HAS_NATIVE, "native boundary extension required")
class FileApiTests(unittest.TestCase):
    def assert_unitary(self, gate, matrix):
        n = int(math.log2(len(matrix)))
        # A fixed prepared product state with both real and imaginary amplitudes.
        circuit = "H 0\n"
        vector = np.array([1, 1], dtype=complex) / math.sqrt(2)
        if n >= 2:
            circuit += "SQRT_X 1\n"
            vector = np.kron(np.array([1, -1j]) / math.sqrt(2), vector)
        if n == 3:
            circuit += "H 2\nT 2\n"
            vector = np.kron(np.array([1, np.exp(1j * math.pi / 4)]) / math.sqrt(2), vector)
        circuit += gate + " " + " ".join(map(str, range(n)))
        masks, expected = moments(matrix @ vector)
        result = run(parse_circuit(circuit), shots=1, task={"observables": masks})
        self.assertEqual(result["status"], "complete", result)
        np.testing.assert_allclose(result["estimates"]["moments"], expected, atol=2e-7)

    @unittest.skipUnless(stim, "Stim required for upstream gate-matrix differential checks")
    def test_untouched_qubit_indices_add_no_wires(self):
        # Indices that no instruction names start in |0> and are traced out.
        # A sparse numbering, also in the recovery, must give the compact result.
        from magna.low_weight import enumerate_errors
        target = parse_circuit("H 0\nT 0\n")
        def exact(protocol, output, recovery):
            result = enumerate_errors(parse_circuit(protocol), max_weight=1, task={"outputs": [output]},
                                      ideal_state=target, recovery=parse_circuit(recovery))
            self.assertEqual(result["status"], "complete")
            return result["totals"]["exact"]
        sparse = exact("H 3\nT 3\nX_ERROR(1/4) 3\nH 5\nM 5\nDETECTOR rec[-1]\nPOSTSELECT det[0]\n", 3,
                       "H 4\nM 4\nCZ rec[-1] 0\n")
        compact = exact("H 0\nT 0\nX_ERROR(1/4) 0\nH 1\nM 1\nDETECTOR rec[-1]\nPOSTSELECT det[0]\n", 0,
                        "H 1\nM 1\nCZ rec[-1] 0\n")
        self.assertEqual(sparse, compact)

    def test_compiled_circuit_rejects_use_from_a_second_thread(self):
        import threading
        entered, release, errors = threading.Event(), threading.Event(), []
        def decoder(bits, config):
            entered.set()
            release.wait(10)
            return (bits[0],)
        registry = ControllerRegistry()
        registry.register("hold", decoder, version="1")
        circuit = parse_circuit("H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[hold] det[-1] -> ctrl[0]\nCX ctrl[0] 1\n",
                                controllers=registry)
        compiled = compile_circuit(circuit, task={"outputs": [1]})
        first = threading.Thread(target=lambda: compiled.sample(Random(1)))
        first.start()
        self.assertTrue(entered.wait(10))
        try:
            compiled.sample(Random(2))
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            release.set()
            first.join(10)
        self.assertEqual(len(errors), 1)
        self.assertIn("in use by another thread", errors[0])
        self.assertIn("exact", compiled.sample(Random(3)))  # usable again once released

    def test_signal_handlers_run_during_one_long_native_batch(self):
        import signal, time
        class Stop(Exception):
            pass
        def handler(*_):
            raise Stop()
        previous = signal.signal(signal.SIGALRM, handler)
        compiled = compile_circuit(parse_circuit("H 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\nPOSTSELECT det[0]\n"))
        started = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.2)
            with self.assertRaises(Stop):
                run(compiled, shots=10**7, progress_every=10**7)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        self.assertLess(time.monotonic() - started, 5)

    def test_supplied_noise_is_validated_against_the_circuit(self):
        from magna.file_api import NoiseDraw
        compiled = compile_circuit(parse_circuit("X_ERROR(1/10) 0\nM 0\nDETECTOR rec[-1]\nPOSTSELECT !det[0]\n"))
        x = lambda qubit: ({"kind": "X", "qubits": [qubit]},)
        self.assertIn("exact", compiled.sample(Random(1), noise=NoiseDraw({"op0": x(0)}, ("m0",))))
        for actions, flips, message in (({"op9": x(0)}, (), "not a noise instruction"),
                                        ({"op1": x(0)}, (), "not a noise instruction"),
                                        ({"op0": x(2**32)}, (), "does not target"),
                                        ({}, ("m1",), "not a measurement record")):
            with self.subTest(actions=actions, flips=flips), self.assertRaisesRegex(ValueError, message):
                compiled.sample(Random(1), noise=NoiseDraw(actions, flips))

    def test_supported_clifford_gates_against_stim(self):
        gates = "I X Y Z H S S_DAG SQRT_X SQRT_X_DAG SQRT_Y SQRT_Y_DAG H_XY H_YZ C_XYZ C_ZYX CX CY CZ SWAP ISWAP ISWAP_DAG"
        for gate in gates.split():
            with self.subTest(gate=gate):
                self.assert_unitary(gate, stim.gate_data(gate).unitary_matrix)

    def test_magic_and_controlled_extensions_against_independent_matrices(self):
        ccx = np.eye(8, dtype=complex)
        ccx[np.ix_([3, 7], [3, 7])] = X
        ccz = np.eye(8, dtype=complex); ccz[7, 7] = -1
        for gate, matrix in (("T", T), ("T_DAG", T.conj().T),
                             ("CS", controlled(S)), ("CS_DAG", controlled(S.conj().T)),
                             ("CH", controlled(H)), ("CCX", ccx), ("CCZ", ccz)):
            with self.subTest(gate=gate):
                self.assert_unitary(gate, matrix)

    def test_nonlinear_mixed_gates_use_actual_decoder_and_global_conditional_moments(self):
        calls = []
        registry = ControllerRegistry()
        def decoder(bits, config):
            calls.append(bits[0])
            return (bits[0],) * 3
        registry.register("choose", decoder, version="1")
        circuit = parse_circuit("""H 0
H 1
T 1
M 0
DETECTOR rec[-1]
CONTROL[choose] det[-1] -> ctrl[0] ctrl[1] ctrl[2]
CS ctrl[0] 1
CH ctrl[1] 1
CY ctrl[2] 1
""", controllers=registry)
        masks, base = moments(T @ H @ np.array([1, 0]))
        _, changed = moments(Y @ H @ S @ T @ H @ np.array([1, 0]))
        compiled = compile_circuit(circuit, task={"outputs": [1], "observables": masks})
        self.assertEqual(calls, [])
        self.assertEqual(compiled.inspect()["route"], "nonlinear")
        for seed in range(8):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw["decoder_calls"], 1)
            self.assertEqual(draw["sampled_parities"], 1)
            self.assertAlmostEqual(draw["acceptance"], 1)
            np.testing.assert_allclose(draw["moments"], changed if calls[-1] else base, atol=1e-12)
        self.assertEqual(set(calls), {0, 1})

    def test_report_flip_changes_feedback_and_task_but_not_true_projection(self):
        circuit = parse_circuit("M(1) 0\nCX rec[-1] 1\nDETECTOR rec[-1]\nPOSTSELECT !det[0]")
        result = run(circuit, task={"observables": [[0, 1], [0, 2]]})
        self.assertEqual(result["estimates"]["acceptance"], 1)
        self.assertEqual(result["estimates"]["moments"], [1, -1])

    def test_pauli_products_resets_and_measure_reset_reports(self):
        one = "\nDETECTOR rec[-1]\nPOSTSELECT det[0]"
        examples = (
            "H 0 1\nMXX 0 1" + one,
            "RY 0 1\nMYY 0 1" + one,
            "H 0\nCX 0 1\nMZZ 0 1" + one,
            "H 0\nCX 0 1\nMPP X0*X1 Z0*Z1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nPOSTSELECT det[0]\nPOSTSELECT det[1]",
            "X 0\nR 0\nM 0" + one,
            "RY 0\nMRY(1) 0\nMY 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nPOSTSELECT !det[0]\nPOSTSELECT det[1]",
            "RX 0\nMRX(1) 0\nMX 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nPOSTSELECT !det[0]\nPOSTSELECT det[1]",
        )
        for text in examples:
            with self.subTest(circuit=text):
                result = run(parse_circuit(text))
                self.assertEqual(result["status"], "complete", result)
                self.assertEqual(result["estimates"]["acceptance"], 1)

    def test_only_required_xor_information_is_sampled_and_postselection_is_preserved(self):
        registry = ControllerRegistry()
        seen = []
        def echo(bits, config):
            seen.append(bits[0]); return bits
        registry.register("echo", echo, version="1")
        circuit = parse_circuit("""H 0 1 2
M 0 1 2
CONTROL[XOR] rec[-3] rec[-2] rec[-1] -> ctrl[0]
CONTROL[echo] ctrl[0] -> ctrl[1]
CX ctrl[1] 3
DETECTOR rec[-3]
POSTSELECT det[0]
""", controllers=registry)
        self.assertEqual(len(circuit.analysis.required_parities), 1)
        compiled = compile_circuit(circuit, task={'outputs': [3], 'ideal_state': ZERO_TARGET})
        for seed in range(4):
            draw = compiled.sample(Random(seed))
            self.assertEqual(draw["sampled_parities"], 1)
            self.assertEqual(draw["acceptance"], .5)
            self.assertEqual(draw["error"], .5 * seen[-1])

    def test_tiny_exact_masses_keep_scientific_values_and_conditional_ratio(self):
        
        from magna import _native
        a = ((9,0,0,0),-1160); e = ((3,0,0,0),-1160); mass=((1,0,0,0),0)
        sci=native_testing.load().file_scalar_ratios(((a,mass),(e,mass)))
        exact={'history_mass':{'coefficients':[1,0,0,0],'power':0},
               'acceptance':{'coefficients':[9,0,0,0],'power':-1160},
               'error':{'coefficients':[3,0,0,0],'power':-1160},'moments':[]}
        result=hooks.summary([{'scientific':sci,'exact':exact}]*2)
        self.assertIsNone(result['estimates']['acceptance'])
        self.assertIsNone(result['estimates']['error'])
        self.assertAlmostEqual(result['estimates']['conditional_error'],1/3)
        self.assertEqual(result['estimates_scientific']['acceptance'],{'mantissa':0.5625,'exponent2':-1156})
        self.assertEqual(result['standard_errors']['acceptance'],0)
        self.assertIn('estimates.acceptance',result['numerics']['underflow_fields'])

    def test_wide_parity_has_balanced_lowering_and_owned_task(self):
        from magna.file_api import NoiseDraw
        text = 'M 0\n'*1200 + 'CONTROL[XOR] ' + ' '.join(f'rec[-{j+1}]' for j in range(1200)) + ' -> ctrl[0]\nCX ctrl[0] 1\n'
        circuit = parse_circuit(text)
        ir = hooks.lower(circuit, (1,), {}, NoiseDraw({}, ()))
        self.assertEqual(len(ir.controls), 1)
        self.assertEqual(len(next(iter(ir.controls.values())).parents), 1200)
        task = {'observables': [[0, 1]]}
        from magna import compile_circuit
        compiled = compile_circuit(parse_circuit('M 0\nDETECTOR rec[-1]\nPOSTSELECT det[0]\n'), task=task)
        task['observables'][0].append(999)
        self.assertEqual(compiled.task['raw']['observables'], [[0, 1]])

    def test_conditional_noise_is_one_physical_channel(self):
        for bit in (0, 1):
            for guarded in (False, True):
                text = f"CONTROL[XOR]({bit}) -> ctrl[0]\nCX ctrl[0] 0\n"
                text += "X_ERROR(1) " + ("ctrl[0] " if guarded else "") + "0\nM 0"
                circuit = parse_circuit(text)
                result = run(circuit, task={'outputs': [0], 'ideal_state': ZERO_TARGET})
                self.assertEqual(result["estimates"]["error"], bit ^ (bit if guarded else 1))
                self.assertEqual(sum(i.name == "X_ERROR" for i in circuit.instructions), 1)

    def test_correlated_alternatives_are_exclusive_and_nonuniform_paulis_have_identity(self):
        # Forced first hit excludes the later alternative even at conditional p=1.
        result = run(parse_circuit("CORRELATED_ERROR(1) X0\nELSE_CORRELATED_ERROR(1) X0\nM 0"),
                     task={'outputs': [0], 'ideal_state': ZERO_TARGET})
        self.assertEqual(result["estimates"]["error"], 1)
        result = run(parse_circuit("CORRELATED_ERROR(0) X0\nELSE_CORRELATED_ERROR(1) X0\nM 0"),
                     task={'outputs': [0], 'ideal_state': ZERO_TARGET})
        self.assertEqual(result["estimates"]["error"], 1)
        circuit = parse_circuit("PAULI_CHANNEL_1(0.25,0.5,0) 0")
        self.assertEqual(circuit.instructions[0].args, (Fraction(1, 4), Fraction(1, 2), Fraction(0)))
        support = set()
        for seed in range(32):
            drawn = hooks.draw_noise(circuit, Random(seed)).gates.get(circuit.instructions[0].op_id, ())
            support.add(tuple(g["kind"] for g in drawn))
        self.assertEqual(support, {(), ("X",), ("Y",)})

    def test_joint_error_ratio_zero_acceptance_and_failed_budgets(self):
        circuit = parse_circuit("H 0 1\nM 0 1")
        selected = parse_circuit("H 0 1\nM 0 1\nDETECTOR rec[-2] rec[-1]\nPOSTSELECT det[0]")
        result = run(selected, shots=3, task={'outputs': [0, 1], 'ideal_state': ZERO_TARGET})
        self.assertEqual(result["estimates"]["acceptance"], .5)
        self.assertEqual(result["estimates"]["error"], .25)
        self.assertEqual(result["estimates"]["conditional_error"], .5)
        self.assertEqual(result["standard_errors"]["conditional_error"], 0)
        empty = run(parse_circuit("X 0\nM 0\nDETECTOR rec[-1]\nPOSTSELECT det[0]"), shots=2,
                    task={'outputs': [0], 'ideal_state': ZERO_TARGET})
        self.assertEqual(empty["estimates"]["acceptance"], 0)
        self.assertIsNone(empty["estimates"]["conditional_error"])
        failed = run(circuit, shots=3, options={"max_queries": 2})
        self.assertEqual((failed["status"], failed["completed"], failed["failed"], failed["not_attempted"]),
                         ("failed", 2, 1, 0))
        self.assertIsNone(failed["estimates"])
        self.assertEqual(failed["partial_statistics"]["estimates"]["acceptance"], 1)
        self.assertEqual(failed["failures"][0]["type"], "ResourceLimit")
        self.assertIn("(max_queries = 2)", failed["failures"][0]["message"])

    def test_exact_target_and_multibit_readout_decoder(self):
        target = run(parse_circuit("H 0\nT 0"), ideal_state=parse_circuit("H 0\nT 0"))
        self.assertEqual(target["status"], "complete", target)
        self.assertEqual(target["estimates"]["error"], 0)
        orthogonal = run(parse_circuit("H 0\nT 0\nZ 0"), ideal_state=parse_circuit("H 0\nT 0"))
        self.assertEqual(orthogonal["estimates"]["error"], 1)
        registry = ControllerRegistry()
        registry.register("readout", lambda bits, config: (0, bits[0]), version="1")
        circuit = parse_circuit("X 0\nM 0\nDETECTOR rec[-1]\n"
                                "CONTROL[readout] det[-1] -> ctrl[0] ctrl[1]\n"
                                "OBSERVABLE_INCLUDE(0) ctrl[1]", controllers=registry)
        result = run(circuit, task={"error_observables": [0]})
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["route"], "nonlinear")
        self.assertEqual(result["estimates"]["error"], 1)

    def test_removed_effect_and_decoder_output_shape_fail_at_compile(self):
        with self.assertRaisesRegex(ValueError, "unknown task fields.*error_effect"):
            compile_circuit(parse_circuit("H 0"), task={
                "error_effect": {"constant": ["1/3", "1/4"]}})
        registry = ControllerRegistry()
        registry.register("readout", lambda bits, config: (0,), version="1")
        circuit = parse_circuit("M 0", controllers=registry)
        with self.assertRaisesRegex(ValueError, "task field 'decoded_error' is no longer supported"):
            compile_circuit(circuit, task={"decoded_error": {"controller": "readout",
                "inputs": [{"records": [0]}], "num_outputs": 1, "output": 1}})
        # The error observable may read only a decoder output that the circuit writes.
        with self.assertRaisesRegex(ValueError, "uninitialized control register ctrl\\[1\\]"):
            parse_circuit("M 0\nDETECTOR rec[-1]\nCONTROL[readout] det[-1] -> ctrl[0]\n"
                          "OBSERVABLE_INCLUDE(0) ctrl[1]", controllers=registry)


if __name__ == "__main__":
    unittest.main()
