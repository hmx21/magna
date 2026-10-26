"""Exact GPU probability queries, including genuinely non-Clifford values."""
import unittest

from tests.support.cuda import require_cuda
from pathlib import Path
from random import Random

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna import compile_circuit, parse_circuit
from magna.cuda_experiment import CudaExperiment
from magna.cuda_queries import GpuQueries
from magna.scalar import ExactScalar

ROOT = Path(__file__).resolve().parents[2]


class ExactGpuQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def check(self, experiment, frames, original=None):
        query = GpuQueries(experiment)
        cpu = query.cpu_evaluate(frames)
        gpu = query.evaluate_host(frames)
        np.testing.assert_array_equal(gpu.coefficients, cpu.coefficients)
        np.testing.assert_array_equal(gpu.powers, cpu.powers)
        if original is None:
            original = experiment.evaluate_frames(frames)
        norm = ExactScalar(*experiment.metadata['target_normalization'])
        for actual, row in zip(gpu.tuples(), original):
            acceptance = ExactScalar(*actual[0])
            self.assertEqual(acceptance, ExactScalar(*row[1]))
            if len(actual) == 2:
                error = acceptance * norm - ExactScalar(*actual[1])
                self.assertEqual(error, ExactScalar(*row[2]))
        return query, gpu

    @staticmethod
    def draws(experiment, count=257):
        rng = Random(931)
        batch = experiment.draw([rng.getrandbits(64) for _ in range(count)])
        return batch, experiment.cpu_frames(batch)

    def test_non_clifford_histogram_is_exact_for_irrational_probabilities(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit(
            'H 0\nT 0\nZ_ERROR(1/3) 0\nH 0\nM 0\nDETECTOR rec[-1]\nPOSTSELECT det[0]')))
        frames = np.stack([experiment.base_frame, experiment.base_frame ^ experiment.columns[0]])
        query, result = self.check(experiment, frames)
        self.assertEqual(query.metadata[0]['mode'], 3)
        self.assertGreater(query.metadata[0]['phase_rank'], 0)
        self.assertEqual(set(row[0] for row in result.tuples()),
                         {((2, 1, 0, -1), -2), ((2, -1, 0, 1), -2)})

    def test_true_t_target_keeps_half_infidelity(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
                                                   ideal_state=parse_circuit('H 0\nT 0')))
        frames = np.vstack([experiment.base_frame, experiment.columns ^ experiment.base_frame])
        _, result = self.check(experiment, frames)
        errors = []
        for row in result.tuples():
            error = ExactScalar(*row[0]) - ExactScalar(*row[1])
            errors.append(float(error.real_decimal()))
        self.assertEqual(sorted(errors), [0, 0.5, 0.5, 1])

    def test_recovery_and_nonunit_target_normalization(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
            recovery=parse_circuit('X 0'), ideal_state={'circuit': parse_circuit('H 0 1\nT 0'),
                                                     'outputs': [0], 'postselection': {1: 0}}))
        self.assertEqual(ExactScalar(*experiment.metadata['target_normalization']), ExactScalar((1, 0, 0, 0), -1))
        frames = np.vstack([experiment.base_frame, experiment.columns ^ experiment.base_frame])
        self.check(experiment, frames)

    def test_inverted_affine_feedback_and_report_noise(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit(
            'H 0\nM(1/3) 0\nCX !rec[-1] 1\nH 1\nT 1\nZ_ERROR(1/4) 1'),
            task={'outputs': [1]}, ideal_state=parse_circuit('H 0\nT 0')))
        batch, frames = self.draws(experiment)
        self.check(experiment, frames, experiment.evaluate_noise(batch))

    def test_public_rp2_cultivation_noise_and_arbitrary_full_frames(self):
        folder = ROOT / 'examples/rp2-d5'
        experiment = CudaExperiment(compile_circuit(folder / 'physical.magna', task=folder / 'task.json'))
        batch, frames = self.draws(experiment, 513)
        query, _ = self.check(experiment, frames, experiment.evaluate_noise(batch))
        self.assertTrue(all(p['mode'] == 1 for p in query.metadata))
        self.assertTrue(all(p['n'] > 32 for p in query.metadata))
        self.assertGreater(experiment.width, 2)
        # Test every coordinate basis, not only typical low-noise sampled frames.
        basis = np.repeat(experiment.base_frame[None, :], 2 * experiment.metadata['frame_bits'], axis=0).copy()
        for axis in range(2):
            for bit in range(experiment.metadata['frame_bits']):
                basis[axis * experiment.metadata['frame_bits'] + bit, axis * experiment.words + bit // 64] ^= np.uint64(1) << np.uint64(bit % 64)
        self.check(experiment, basis)

    def test_n64_dynamic_and_large_bh_static_geometry(self):
        for name in ('bh10-T', 'bh100-T'):
            folder = ROOT / 'examples' / name
            experiment = CudaExperiment(compile_circuit(folder / 'physical.magna', task=folder / 'task.json',
                                                       options={'max_quotient_rank': 12, 'max_terms': 4096}))
            frames = np.vstack([experiment.base_frame, experiment.columns ^ experiment.base_frame])
            query, _ = self.check(experiment, frames)
            if name == 'bh10-T':
                self.assertEqual(query.metadata[1]['n'], 64)
                self.assertEqual(query.metadata[1]['mode'], 1)
            else:
                self.assertTrue(all(p['mode'] == 2 for p in query.metadata))
                raw = query.metadata[0]
                guard = np.frombuffer(raw['guards'], '<u4')[0]
                offsets = np.frombuffer(raw['feature_offsets'], '<u4')
                masks = np.frombuffer(raw['feature_masks'], '<u8')
                words = np.frombuffer(raw['feature_words'], '<u4')
                index = int(offsets[guard])
                mask = int(masks[index])
                bad = experiment.base_frame[None, :].copy()
                bad[0, words[index]] ^= np.uint64(mask & -mask)
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    query.cpu_evaluate(bad)
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    query.evaluate_host(bad)
                unchecked = query.evaluate(self.cp.asarray(bad))
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    unchecked.values()

    def test_empty_batch_padding_and_layout(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0'), ideal_state=parse_circuit('H 0\nT 0')))
        query = GpuQueries(experiment)
        empty = np.empty((0, experiment.width), dtype=np.uint64)
        self.assertEqual(query.evaluate_host(empty).coefficients.shape, (0, 2, 4))
        invalid = experiment.base_frame[None, :].copy()
        bit = experiment.metadata['frame_bits']
        invalid[0, bit // 64] ^= np.uint64(1) << np.uint64(bit % 64)
        with self.assertRaises(ValueError):
            query.evaluate_host(invalid)
        with self.assertRaises(ValueError):
            query.cpu_evaluate(invalid)
        with self.assertRaises(TypeError):
            query.evaluate(self.cp.zeros((1, experiment.width), dtype=self.cp.float64))

    def test_reconfigured_plan_is_rejected_before_gpu_launch(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0'), ideal_state=parse_circuit('H 0\nT 0')))
        query = GpuQueries(experiment)
        frames = self.cp.asarray(experiment.base_frame[None, :])
        options = dict(experiment.compiled.options, max_quotient_rank=0)
        experiment.compiled._execution.configure(options)
        with self.assertRaisesRegex(ValueError, 'reconfigured'):
            query.evaluate(frames)
        with self.assertRaisesRegex(ValueError, 'reconfigured'):
            query.cpu_evaluate(experiment.base_frame[None, :])

    def test_zero_dimensional_empty_and_unmeasured_queries(self):
        for circuit in ('', 'R 0'):
            with self.subTest(circuit=circuit):
                experiment = CudaExperiment(compile_circuit(parse_circuit(circuit)))
                frames = np.repeat(experiment.base_frame[None, :], 3, axis=0)
                query, result = self.check(experiment, frames)
                self.assertEqual(query.metadata[0]['n'], 0)
                self.assertEqual(result.tuples(), ((((1, 0, 0, 0), 0),),) * 3)

    def test_zero_reference_acceptance_can_be_activated_by_noise(self):
        experiment = CudaExperiment(compile_circuit(parse_circuit(
            'X_ERROR(1/2) 0\nM 0\nDETECTOR rec[-1]\nPOSTSELECT !det[0]')))
        self.assertEqual(experiment.actions, 1)
        frames = np.stack([experiment.base_frame, experiment.base_frame ^ experiment.columns[0]])
        _, result = self.check(experiment, frames)
        self.assertEqual(result.tuples(),
                         ((((0, 0, 0, 0), 0),), (((1, 0, 0, 0), 0),)))


if __name__ == '__main__':
    unittest.main()
