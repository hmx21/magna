"""Check literal resource faults against the RM15 acceptance and output target."""
import unittest
from pathlib import Path

from magna import parse_circuit, run
from .generate import generate


class DistillationFileTest(unittest.TestCase):
    def evaluate(self, faults):
        circuit, task = generate("0")
        for q in faults:
            circuit = circuit.replace(f"Z_ERROR(0) {16 + q}\n", f"Z_ERROR(1) {16 + q}\n")
        task["ideal_state"] = str(Path(__file__).with_name("ideal.magna").resolve())
        result = run(parse_circuit(circuit), task=task, shots=1, seed=7)
        self.assertEqual(result["status"], "complete", result["failures"])
        self.assertEqual(result["route"], "linear")
        return result["estimates"]

    def test_noiseless_output_target(self):
        value = self.evaluate(())
        self.assertEqual(value["acceptance"], 1)
        self.assertEqual(value["conditional_error"], 0)

    def test_every_single_resource_z_is_rejected(self):
        for q in range(15):
            with self.subTest(resource=q):
                value = self.evaluate((q,))
                self.assertEqual(value["acceptance"], 0)
                self.assertIsNone(value["conditional_error"])

    def test_undetected_logical_resource_fault(self):
        # These three distinct column syndromes sum to zero and have odd logical parity.
        value = self.evaluate((0, 1, 11))
        self.assertEqual(value["acceptance"], 1)
        self.assertEqual(value["conditional_error"], 1)


if __name__ == "__main__":
    unittest.main()
