"""Native aggregation must reproduce sequential public draws, including failures."""
from tests.support.readout_target import ZERO_TARGET
from random import Random
import unittest

from magna import _native, compile_circuit
from magna.circuit_file import parse_circuit
from magna.file_api import run
from tests.support import file_hooks as hooks
from tests.support import native_testing


class NativeBatchTests(unittest.TestCase):
    def test_aggregate_matches_single_draws(self):
        circuit = parse_circuit('H 0\nT 0\nZ_ERROR(1/10) 0\nMX 0')
        for method in ('affine_support', 'exact_marginal'):
            for sampling in (None, {'method': 'importance', 'proposal_scale': '2'}):
                kwargs = dict(task={'sample_records': [0]}, ideal_state=parse_circuit('H 0'),
                              options={'history_sampler': method}, sampling=sampling)
                compiled = compile_circuit(circuit, **kwargs)
                rng = Random(391); draws = [compiled.sample(rng) for _ in range(79)]
                expected = hooks.summary(draws); efficiency = native_testing.load().make_file_efficiency()
                for d in draws:
                    native_testing.load().add_file_efficiency(efficiency, d['scientific'][0], d['scientific'][1],
                                                d['weight_scientific'], d['noise_weight_scientific'])
                actual = run(circuit, shots=79, seed=391, **kwargs)
                for key, value in expected.items():
                    self.assertEqual(actual[key], value, (method, sampling, key))
                self.assertEqual(actual['weights'], hooks.weight_summary(draws, compiled.noise_schedule, method))
                self.assertEqual(actual['efficiency'], native_testing.load().file_efficiency_summary(efficiency))
                self.assertEqual(actual['sampled_records'], [d['sampled_records'] for d in draws])
                self.assertEqual(actual['resources']['queries'], compiled.queries)

    def test_failure_retains_only_completed_prefix(self):
        circuit = parse_circuit('X_ERROR(1/3) 0\nM 0')
        kwargs = dict(task={'outputs': [0], 'ideal_state': ZERO_TARGET}, options={'max_queries': 11})
        compiled = compile_circuit(circuit, **kwargs)
        rng = Random(99); draws = []
        for _ in range(20):
            try:
                draws.append(compiled.sample(rng))
            except RuntimeError:
                break
        actual = run(circuit, shots=20, seed=99, **kwargs)
        self.assertEqual(actual['status'], 'failed')
        self.assertEqual(actual['completed'], len(draws))
        self.assertEqual(actual['failures'][0]['shot'], len(draws))
        self.assertEqual(actual['partial_statistics']['estimates'], hooks.summary(draws)['estimates'])
        self.assertEqual(actual['resources']['queries'], compiled.queries)
