"""Detector syntax lowers to the existing causal parity/query contracts."""
import json
import unittest

from magna.circuit_file import CircuitSyntaxError, ControllerRegistry, parse_circuit
from magna.file_api import compile_circuit, run


class DetectorFeedbackTests(unittest.TestCase):
    def registry(self, callback):
        registry = ControllerRegistry()
        registry.register("decode", callback, version="test-1")
        return registry

    def test_order_duplicates_inversion_reports_and_prior_state(self):
        calls = []
        def decode(bits, config):
            calls.append(bits)
            return (bits[0] ^ bits[-1],)
        circuit = parse_circuit("""M(1) 0
DETECTOR rec[-1]
M 1
DETECTOR rec[-1]
CONTROL[XOR](0) -> ctrl[0]
DETECTOR rec[-1]
CONTROL[decode] det[0] det[1] det[0] !det[1] det[-1] ctrl[0] -> ctrl[0]
DETECTOR rec[-1]
CONTROL[decode] det[-4] det[-3] det[0] !det[1] det[-1] ctrl[0] -> ctrl[0]
CX ctrl[0] 2
""", controllers=self.registry(decode))
        result = run(circuit, shots=3, seed=11, task={"outputs": [2], "observables": [[0, 1]]})
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(calls, [(1, 0, 1, 1, 0, 0), (1, 0, 1, 1, 0, 1)] * 3)
        self.assertEqual(result["estimates"]["moments"], [1.0])
        self.assertEqual(circuit.to_dict()["format"], "magna-0.1")
        self.assertEqual(parse_circuit("M 0\nDETECTOR rec[-1]").to_dict()["format"], "magna-0.1")
        self.assertEqual(parse_circuit(circuit.to_text(), controllers=circuit.controllers).fingerprint,
                         circuit.fingerprint)
        json.dumps(circuit.to_dict())

    def test_separate_selection_rows_and_combination(self):
        prefix = ("X 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
                  "CONTROL[NOT] det[0] -> ctrl[0]\nCONTROL[NOT] det[1] -> ctrl[1]\n")
        for selection, task, mass in (
            ("POSTSELECT det[0] det[1]", {}, 0),
            ("POSTSELECT ctrl[0] ctrl[1]", {}, 1),
            ("POSTSELECT ctrl[0] ctrl[0]\nPOSTSELECT !det[0]", {}, 1),
            ("POSTSELECT ctrl[0]\nPOSTSELECT det[0]", {}, 0),
            ("POSTSELECT ctrl[0]\nPOSTSELECT det[0]", {}, 0),
            ("POSTSELECT ctrl[0]\nDETECTOR rec[-2] rec[-1]\nPOSTSELECT !det[2]", {}, 0),
        ):
            with self.subTest(selection=selection, task=task):
                result = run(parse_circuit(prefix+selection), shots=1, task=task)
                self.assertEqual(result["status"], "complete", result)
                self.assertEqual(result["estimates"]["acceptance"], mass)

    def test_repeat_numbering_and_parity_cancellation(self):
        circuit = parse_circuit("""M 0
DETECTOR rec[-1] rec[-1]
REPEAT 3 {
 M 0
 DETECTOR rec[-1] rec[-2]
 CONTROL[XOR](1) det[-1] det[0] -> ctrl[0]
}
POSTSELECT det[-1] det[0]
""")
        controls = [i for i in circuit.instructions if i.name == "CONTROL"]
        self.assertEqual([i.targets[0].index for i in controls], [1, 2, 3])
        self.assertEqual([sorted(t.index for t in circuit.analysis.expression(i.targets[0]).terms)
                          for i in controls], [[0, 1], [1, 2], [2, 3]])
        self.assertFalse(circuit.analysis.detectors[0].terms)

    def test_selection_and_target_do_not_bias_earlier_decoder_inputs(self):
        histories = []
        def decode(bits, config):
            histories.append(bits)
            return bits
        text = "H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[decode] det[0] -> ctrl[0]\nCX ctrl[0] 1\n"
        registry = self.registry(decode)
        task = {"outputs": [1]}
        a = run(parse_circuit(text, controllers=registry), shots=32, seed=9, task=task, ideal_state=parse_circuit("I 0"))
        original = histories[:]; histories.clear()
        b = run(parse_circuit(text+"POSTSELECT det[0]", controllers=registry), shots=32, seed=9, task=task, ideal_state=parse_circuit("I 0"))
        self.assertEqual(a["status"], "complete", a)
        self.assertEqual(b["status"], "complete", b)
        self.assertEqual(histories, original)
        self.assertEqual(set(original), {(0,), (1,)})
        self.assertEqual(b["estimates"]["error"], 0)

    def test_wide_history_and_affine_route(self):
        width = 1025
        calls = []
        text = "M 0\n" + "DETECTOR rec[-1]\n"*width
        text += "CONTROL[decode] " + " ".join(f"det[{i}]" for i in range(width)) + " -> ctrl[0]\nCX ctrl[0] 1"
        circuit = parse_circuit(text, controllers=self.registry(lambda bits, config: calls.append(bits) or (0,)))
        self.assertEqual(len(circuit.analysis.required_parities), 1)
        result = run(circuit, shots=1)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(calls, [(0,)*width])
        affine = parse_circuit("H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[XOR] det[0] -> ctrl[0]\nCX ctrl[0] 0")
        self.assertEqual(affine.analysis.route, "linear")
        self.assertEqual(run(affine, shots=1, task={"observables": [[0, 1]]})["estimates"]["moments"], [1.0])

    def test_future_and_ambiguous_uses_rejected(self):
        for text in (
            "CONTROL[XOR] det[0] -> ctrl[0]",
            "POSTSELECT det[-1]",
            "M 0\nDETECTOR rec[-1]\nCONTROL[XOR] det[1] -> ctrl[0]",
            "M 0\nDETECTOR rec[-1]\nPOSTSELECT(2) det[0]",
            "M 0\nDETECTOR rec[-1]\nPOSTSELECT(0) det[0]",
            "M 0\nDETECTOR rec[-1]\nPOSTSELECT(1) det[0]",
            "M 0\nPOSTSELECT rec[-1]",
            "M 0\nDETECTOR rec[-1]\nCX det[0] 0",
            "POSTSELECT",
        ):
            with self.subTest(text=text), self.assertRaises(CircuitSyntaxError):
                parse_circuit(text)
        selected = parse_circuit("DETECTOR\nPOSTSELECT det[0]")
        with self.assertRaisesRegex(ValueError, "unitary preparation"):
            compile_circuit(parse_circuit("I 0"), ideal_state=selected)
        with self.assertRaisesRegex(ValueError, "physical circuit"):
            compile_circuit(parse_circuit("I 0"), ideal_state=parse_circuit(""), recovery=selected)


if __name__ == "__main__":
    unittest.main()
