"""Efficiency diagnostics checked against independent finite weighted sums."""
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from io import StringIO
import json
import math
from pathlib import Path
import tempfile
import unittest

from magna import _native, compare_samplers
from magna.circuit_file import parse_circuit
from magna.file_api import main, run, compile_circuit
from tests.support import native_testing


def scientific(x):
    return math.frexp(float(x))


class HistoryEfficiencyTests(unittest.TestCase):
    def test_weighted_residuals_and_noise_correction(self):
        # Includes a zero draw and unequal acceptance, history and noise weights.
        rows = [(Fraction(a, 5), Fraction(e, 7), Fraction(w, 3), Fraction(l, 4))
                for a, e, w, l in [(1, 0, 2, 1), (3, 2, 5, 3), (4, 4, 3, 6), (0, 0, 0, 1)]]
        state = native_testing.load().make_file_efficiency()
        for row in rows:
            native_testing.load().add_file_efficiency(state, *(scientific(x) for x in row))
        f = sum(e for a, e, w, l in rows)/sum(a for a, e, w, l in rows)
        residual = [(e-f*a)**2 for a, e, w, l in rows]
        u = sum(residual)
        b = sum(l*d/w for (a, e, w, l), d in zip(rows, residual) if w)
        result = native_testing.load().file_efficiency_summary(state)
        self.assertEqual(result['status'], 'available')
        self.assertAlmostEqual(result['kappa'], float(u/b))
        self.assertAlmostEqual(result['variance_coefficient'], float(4*u/sum(r[0] for r in rows)**2))
        self.assertAlmostEqual(result['exact_marginal_variance_coefficient'], float(4*b/sum(r[0] for r in rows)**2))
        self.assertEqual(result['zero_weights'], 1)

    def test_extreme_common_scale_and_zero_cases(self):
        for shift in (-10000, 0, 10000):
            state = native_testing.load().make_file_efficiency()
            for error in (0, 1):
                native_testing.load().add_file_efficiency(state, (.5, shift+1),
                    (.5, shift+1) if error else (0., 0), (.5, shift+1), (.5, shift+1))
            summary = native_testing.load().file_efficiency_summary(state)
            self.assertEqual(summary['kappa'], 1)
            self.assertEqual(summary['variance_coefficient'], .25)
        state = native_testing.load().make_file_efficiency()
        for _ in range(2):
            native_testing.load().add_file_efficiency(state, (0., 0), (0., 0), (0., 0), (.5, 1))
        summary = native_testing.load().file_efficiency_summary(state)
        self.assertEqual(summary['status'], 'zero_acceptance')
        self.assertIsNone(summary['kappa'])
        self.assertIsNone(summary['rho'])
        self.assertEqual(summary['ess'], 0)

    def test_default_progress_preserves_draws_and_manual_born(self):
        circuit = parse_circuit('H 0\nT 0\nMX 0')
        kwargs = dict(task={'sample_records': [0]}, ideal_state=parse_circuit('H 0'), shots=21, seed=42)
        baseline = run(circuit, **kwargs)
        stream = StringIO()
        with redirect_stderr(stream):
            progress = run(circuit, progress_every=8, **kwargs)
        self.assertEqual(baseline['status'], 'complete', baseline)
        self.assertEqual(baseline['history_sampler'], 'affine_support')
        self.assertIsNone(baseline['sampling']['exact_marginal_records'])
        self.assertIsNone(compile_circuit(circuit).inspect()['sampling']['exact_marginal_records'])
        self.assertEqual(baseline['estimates'], progress['estimates'])
        self.assertEqual(baseline['sampled_records'], progress['sampled_records'])
        self.assertEqual(baseline['efficiency'], progress['efficiency'])
        self.assertEqual(stream.getvalue().count('kappa='), 3)
        for n in (8, 16, 21):
            self.assertIn(f'[{n}/21]', stream.getvalue())
        self.assertIn('--history-sampler exact_marginal', stream.getvalue())
        born = run(circuit, options={'history_sampler': 'exact_marginal'}, **kwargs)
        self.assertEqual(born['history_sampler'], 'exact_marginal')
        self.assertEqual(born['sampling']['exact_marginal_records'], 'true conditional distribution')
        inspected = compile_circuit(circuit, options={'history_sampler': 'exact_marginal'}).inspect()
        self.assertEqual(inspected['sampling']['exact_marginal_records'], 'true conditional distribution')
        self.assertEqual(born['efficiency']['kappa'], 1)

    def test_pilot_reports_gamma_without_modifying_options(self):
        options = {'history_sampler': 'exact_marginal'}
        result = compare_samplers(parse_circuit('X_ERROR(1/2) 0'), ideal_state=parse_circuit('I 0'),
                                  pilot_shots=32, options=options, seed=13)
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['kappa'], 1)
        self.assertAlmostEqual(result['gamma'], result['affine_support_seconds_per_draw']/result['exact_marginal_seconds_per_draw'])
        self.assertFalse(result['automatic_switching'])
        self.assertEqual(options, {'history_sampler': 'exact_marginal'})
        for mode, pilot in result['pilots'].items():
            self.assertEqual(pilot['completed'], 32)
            self.assertEqual(pilot['history_sampler'], mode)
            self.assertGreater(pilot['timings']['preparation_seconds'], 0)
        inconclusive = compare_samplers(parse_circuit('I 0'), ideal_state=parse_circuit('I 0'), pilot_shots=2)
        self.assertEqual(inconclusive['status'], 'inconclusive')
        self.assertIsNone(inconclusive['recommended_sampler'])
        self.assertIsNone(inconclusive['gamma'])

    def test_cli_json_and_progress_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            circuit = Path(tmp)/'physical.magna';circuit.write_text('X_ERROR(1/2) 0\n')
            target = Path(tmp)/'target.magna';target.write_text('I 0\n')
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = main(['run', str(circuit), '--ideal-state', str(target), '--shots', '8', '--progress-every', '4', '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.getvalue())['history_sampler'], 'affine_support')
            self.assertEqual(err.getvalue().count('kappa='), 2)
            out = StringIO()
            with redirect_stdout(out):
                code = main(['compare-samplers', str(circuit), '--ideal-state', str(target), '--pilot-shots', '16', '--json'])
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(out.getvalue())['automatic_switching'])

    def test_invalid_counts_and_native_decision(self):
        circuit = parse_circuit('I 0')
        for retired in ('capped', 'capped_affine', 'born'):
            with self.assertRaisesRegex(ValueError, 'affine_support or exact_marginal'):
                compile_circuit(circuit, options={'history_sampler': retired})
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): run(circuit, progress_every=count)
        for count in (None, 0, 1, True):
            with self.assertRaises(ValueError): compare_samplers(circuit, pilot_shots=count)
        self.assertEqual(_native.decide_file_sampler(4., 2., 4., 10)['recommended_sampler'], 'exact_marginal')
        self.assertEqual(_native.decide_file_sampler(1., 2., 4., 10)['recommended_sampler'], 'affine_support')

    def test_incomplete_pilot_and_unavailable_metrics(self):
        circuit = parse_circuit('H 0\nT 0\nMX 0')
        result = compare_samplers(circuit, task={'sample_records': [0]}, ideal_state=parse_circuit('H 0'),
                                  pilot_shots=4, options={'max_queries': 1})
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['gamma'])
        self.assertIsNone(result['recommended_sampler'])
        no_metric = run(parse_circuit('I 0'), shots=2)
        self.assertEqual(no_metric['efficiency']['status'], 'no_error_metric')
        self.assertIsNone(no_metric['efficiency']['kappa'])
        extreme = native_testing.load().make_file_efficiency()
        for _ in range(2):
            native_testing.load().add_file_efficiency(extreme, (.5, 1), (.5, -100000), (.5, 1), (.5, 1))
        self.assertEqual(native_testing.load().file_efficiency_summary(extreme)['status'], 'diagnostic_range_exceeded')
