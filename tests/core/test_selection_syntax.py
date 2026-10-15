"""Acceptance and classical error declared in the circuit: POSTSELECT and OBSERVABLE_INCLUDE."""
import math
import unittest

from magna import ControllerRegistry, enumerate_errors, parse_circuit


def majority():
    registry = ControllerRegistry()
    registry.register('majority', lambda bits, config: (int(sum(bits) >= 2),), version='1')
    return registry


class SelectionSyntaxTests(unittest.TestCase):
    def totals(self, text, task=None, controllers=None):
        result = enumerate_errors(parse_circuit(text, controllers=controllers), max_weight=4, task=task)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['totals']['covered_probability'], 1)
        return result['totals']

    def test_complemented_detector_requires_a_one(self):
        base = 'H 0\nT 0\nH 0\nX_ERROR(1/10) 0\nM 0\nDETECTOR rec[-1]\n'
        zero = self.totals(base+'POSTSELECT det[0]')['acceptance']
        one = self.totals(base+'POSTSELECT !det[0]')['acceptance']
        # P(m = 1) = (2 - sqrt 2)/4 * 9/10 + (2 + sqrt 2)/4 * 1/10.
        self.assertTrue(math.isclose(one, .5-.2*math.sqrt(2)))
        self.assertTrue(math.isclose(zero+one, 1))
        self.assertEqual(parse_circuit(base+'POSTSELECT !det[0]').format_version, 'magna-0.1')

    def test_complemented_register_matches_explicit_not(self):
        circuit = ('H 0\nM(1/10) 0\nM(1/10) 0\nM(1/10) 0\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                   'CONTROL[majority] det[0] det[1] det[2] -> ctrl[0]\n')
        explicit = self.totals(circuit+'CONTROL[NOT] ctrl[0] -> ctrl[1]\nPOSTSELECT ctrl[1]', controllers=majority())
        direct = self.totals(circuit+'POSTSELECT !ctrl[0]', controllers=majority())
        self.assertEqual(direct['exact']['acceptance'], explicit['exact']['acceptance'])
        self.assertEqual(direct['acceptance'], .5)

    def test_decoder_output_in_observable_defines_the_error(self):
        circuit = ('X_ERROR(1/5) 0\nM(1/10) 0\nM(1/10) 0\nM(1/10) 0\n'
                   'DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                   'CONTROL[majority] det[0] det[1] det[2] -> ctrl[0]\nOBSERVABLE_INCLUDE(0) ctrl[0]')
        totals = self.totals(circuit, task={'error_observables': [0]}, controllers=majority())
        # A wrong majority needs two or three of the three report flips.
        wrong = 3*.1**2*.9+.1**3
        self.assertTrue(math.isclose(totals['error'], .2*(1-wrong)+.8*wrong))
        self.assertEqual(totals['acceptance'], 1)
        self.assertEqual(parse_circuit(circuit, controllers=majority()).format_version, 'magna-0.1')

    def test_observable_reads_the_register_value_at_its_line(self):
        circuit = 'CONTROL[XOR](1) -> ctrl[0]\nOBSERVABLE_INCLUDE(0) ctrl[0]\nCONTROL[XOR](0) -> ctrl[0]'
        self.assertEqual(self.totals(circuit, task={'error_observables': [0]})['error'], 1)
        record_and_register = 'X 0\nM 0\nCONTROL[XOR] rec[-1] -> ctrl[0]\nOBSERVABLE_INCLUDE(0) rec[-1] ctrl[0]'
        self.assertEqual(self.totals(record_and_register, task={'error_observables': [0]})['error'], 0)

    def test_removed_task_fields_explain_the_circuit_spelling(self):
        from magna import compile_circuit
        circuit = parse_circuit('M 0\nDETECTOR rec[-1]')
        for field, value, hint in (('detectors', {'0': 0}, 'POSTSELECT det'),
                                   ('postselection', [[0]], 'after a DETECTOR for each parity'),
                                   ('outcomes', [0], 'POSTSELECT det'),
                                   ('error_checks', [[0]], 'OBSERVABLE_INCLUDE'),
                                   ('decoded_error', {'controller': 'x', 'inputs': []}, 'OBSERVABLE_INCLUDE')):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"task field '{field}' is no longer supported; .*{hint}"):
                compile_circuit(circuit, task={field: value})

    def test_external_decoders_read_detectors_not_records(self):
        with self.assertRaisesRegex(Exception, r'external CONTROL inputs must be det\[\.\.\.\] or ctrl\[\.\.\.\]'):
            parse_circuit('M 0\nCONTROL[majority] rec[-1] -> ctrl[0]', controllers=majority())
        parse_circuit('M 0\nCONTROL[XOR] rec[-1] -> ctrl[0]\nCX rec[-1] 0')

    def test_invalid_targets_name_the_allowed_forms(self):
        for text, message in (('M 0\nPOSTSELECT rec[-1]', 'POSTSELECT targets must be det'),
                              ('M 0\nDETECTOR rec[-1]\nPOSTSELECT(1) det[0]', 'write !det'),
                              ('CONTROL[XOR] -> ctrl[0]\nOBSERVABLE_INCLUDE(0) !ctrl[0]', 'OBSERVABLE_INCLUDE targets'),
                              ('CONTROL[XOR] -> ctrl[0]\nDETECTOR ctrl[0]', 'DETECTOR targets must be measurement records')):
            with self.subTest(text=text), self.assertRaisesRegex(Exception, message):
                parse_circuit(text)


if __name__ == '__main__':
    unittest.main()
