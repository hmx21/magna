"""Partial CUDA transport agrees with original exact non-Clifford execution."""
import math
from random import Random
import unittest

from tests.support.cuda import require_cuda

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy is required for the CUDA experiment')

from magna import _native, compile_circuit, parse_circuit
from magna.cuda_experiment import CudaExperiment, CudaFrameMapper, NoiseBatch, validate_csr
from magna.scalar import ExactScalar


def real(raw):
    return float(ExactScalar(*raw).real_decimal())


def explicit(plan, rows):
    offsets = [0]
    indices = []
    for row in rows:
        indices.extend(row)
        offsets.append(len(indices))
    return NoiseBatch(plan, np.array(offsets, dtype=np.uint64), np.array(indices, dtype=np.uint32))


class CsrValidationTests(unittest.TestCase):
    def test_empty_rows_and_repeated_actions_are_valid_for_the_xor_kernel(self):
        offsets, indices = validate_csr([0, 0, 2, 2], [3, 3], 4)
        self.assertEqual(offsets.dtype, np.dtype('uint64'))
        self.assertEqual(indices.dtype, np.dtype('uint32'))
        empty_offsets, empty_indices = validate_csr([0], [], 0)
        self.assertEqual((empty_offsets.size, empty_indices.size), (1, 0))

    def test_rejects_malformed_and_out_of_range_csr_before_launch(self):
        cases = [([], [], 1), ([1], [], 1), ([0, 2, 1], [0], 1),
                 ([0, 2], [0], 1), ([0, 1], [1], 1), ([0, 1], [-1], 1),
                 ([0, -1], [], 1), ([[0]], [], 1), ([0, 1], [[0]], 1),
                 ([0, 1], [2**32], 2**33)]
        for offsets, indices, actions in cases:
            with self.subTest(offsets=offsets, indices=indices):
                with self.assertRaises(ValueError):
                    validate_csr(offsets, indices, actions)
        for offsets, indices in [([0.0], []), ([0, 1], [True]), ([0, 1], [0.5])]:
            with self.assertRaises(TypeError):
                validate_csr(offsets, indices, 1)


class NativeCudaBridgeTests(unittest.TestCase):
    @staticmethod
    def magic_plan():
        return CudaExperiment(compile_circuit(
            parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
            ideal_state=parse_circuit('H 0\nT 0')))

    def test_exhaustive_paulis_preserve_true_t_target_infidelity(self):
        plan = self.magic_plan()
        self.assertEqual(plan.actions, 3)
        batch = explicit(plan, [[], [0], [1], [2]])
        original = plan.evaluate_noise(batch)
        frames = plan.cpu_frames(batch)
        self.assertEqual(plan.evaluate_frames(frames), original)
        self.assertEqual(sorted(real(row[2]) for row in original), [0.0, 0.5, 0.5, 1.0])
        self.assertEqual([real(row[1]) for row in original], [1.0] * 4)

    def test_irrational_acceptance_and_measurement_report_flips(self):
        plan = CudaExperiment(compile_circuit(parse_circuit(
            'H 0\nT 0\nZ_ERROR(1/3) 0\nH 0\nM(1/4) 0\n'
            'DETECTOR rec[-1]\nPOSTSELECT det[0]')))
        batch = explicit(plan, [[], [0], [1], [0, 1]])
        original = plan.evaluate_noise(batch)
        self.assertEqual(plan.evaluate_frames(plan.cpu_frames(batch)), original)
        expected = [0.5 - math.sqrt(2) / 4] * 2 + [0.5 + math.sqrt(2) / 4] * 2
        np.testing.assert_allclose(sorted(real(row[1]) for row in original), expected, atol=1e-15)

    def test_inverted_affine_feedback_preserves_output_infidelity(self):
        plan = CudaExperiment(compile_circuit(parse_circuit(
            'M 0\nCX !rec[-1] 1\nH 1\nT 1\nZ_ERROR(1/3) 1'),
            task={'outputs': [1]}, ideal_state=parse_circuit('H 0\nT 0')))
        batch = explicit(plan, [[], [0]])
        original = plan.evaluate_noise(batch)
        self.assertEqual(plan.evaluate_frames(plan.cpu_frames(batch)), original)
        self.assertEqual([real(row[2]) for row in original], [1.0, 0.0])

    def test_native_draws_partition_reproducibly_and_match_public_sampler(self):
        plan = self.magic_plan()
        rng = Random(981)
        seeds = [rng.getrandbits(64) for _ in range(79)]
        whole = plan.draw(seeds)
        left, right = plan.draw(seeds[:23]), plan.draw(seeds[23:])
        np.testing.assert_array_equal(whole.indices, np.concatenate([left.indices, right.indices]))
        np.testing.assert_array_equal(whole.offsets,
                                      np.concatenate([left.offsets, right.offsets[1:] + left.offsets[-1]]))
        self.assertEqual(plan.evaluate_frames(plan.cpu_frames(whole)), plan.evaluate_noise(whole))
        # With no history sampling, public sample consumes exactly one 64-bit
        # seed from this RNG per draw.  Check the actual public API, too.
        rng = Random(981)
        public = [plan.compiled.sample(rng)['exact'] for _ in seeds]
        exact = plan.evaluate_noise(whole)
        for row, sample in zip(exact, public):
            for value, name in zip(row, ('history_mass', 'acceptance', 'error')):
                self.assertEqual(value, (tuple(sample[name]['coefficients']), sample[name]['power']))

    def test_zero_shots_and_ownership(self):
        plan = self.magic_plan()
        batch = plan.draw([])
        frames = plan.cpu_frames(batch)
        self.assertEqual(frames.shape, (0, plan.width))
        self.assertEqual(plan.evaluate_frames(frames), ())
        self.assertEqual(plan.evaluate_noise(batch), ())
        other = self.magic_plan()
        with self.assertRaisesRegex(ValueError, 'different experiment'):
            other.cpu_frames(batch)
        with self.assertRaises(ValueError):
            plan.evaluate_frames(np.zeros((1, plan.width + 1), dtype=np.uint64))
        for seeds in ([-1], [True], [0.1]):
            with self.assertRaises((TypeError, ValueError)):
                plan.draw(seeds)

    def test_experiment_rejects_multiple_outcomes_at_one_site(self):
        plan = self.magic_plan()
        for rows in ([[0, 1]], [[0, 0]], [[1, 0]]):
            batch = explicit(plan, rows)
            for method in (plan.cpu_frames, plan.gpu_frames, plan.evaluate_noise):
                with self.assertRaisesRegex(ValueError, 'one outcome per site'):
                    method(batch)
        # Row boundaries, including empty shots, do not constrain ordering.
        batch = explicit(plan, [[2], [], [0], [], [1]])
        self.assertEqual(plan.evaluate_frames(plan.cpu_frames(batch)), plan.evaluate_noise(batch))

    def test_recovery_and_postselected_target_normalization(self):
        plan = CudaExperiment(compile_circuit(parse_circuit('H 0\nT 0\nDEPOLARIZE1(3/4) 0'),
            recovery=parse_circuit('X 0'),
            ideal_state={'circuit': parse_circuit('H 0 1\nT 0'),
                         'outputs': [0], 'postselection': {1: 0}}))
        normalization = real(plan.metadata['target_normalization'])
        self.assertEqual(normalization, 0.5)
        batch = explicit(plan, [[], [0], [1], [2]])
        original = plan.evaluate_noise(batch)
        self.assertEqual(plan.evaluate_frames(plan.cpu_frames(batch)), original)
        self.assertEqual(sorted(real(row[2]) / normalization for row in original), [0.0, 0.5, 0.5, 1.0])

    def test_padding_and_stale_native_plan_are_rejected(self):
        plan = self.magic_plan()
        batch = explicit(plan, [[]])
        frames = plan.cpu_frames(batch).copy()
        bit = int(plan.metadata['frame_bits'])
        self.assertNotEqual(bit % 64, 0)
        frames[0, bit // 64] |= np.uint64(1) << np.uint64(bit % 64)
        with self.assertRaises(ValueError):
            plan.evaluate_frames(frames)
        options = dict(plan.compiled.options, max_quotient_rank=0)
        plan.compiled._execution.configure(options)
        with self.assertRaisesRegex(ValueError, 'reconfigured'):
            plan.cpu_frames(batch)

    def test_option_dictionary_edits_invalidate_the_fixed_experiment(self):
        plan = self.magic_plan()
        batch = explicit(plan, [[]])
        frames = plan.cpu_frames(batch)
        plan.compiled.options['max_quotient_rank'] = 0
        operations = [lambda: plan.validate(), lambda: plan.draw([]),
                      lambda: plan.cpu_frames(batch), lambda: plan.gpu_frames(batch),
                      lambda: plan.evaluate_frames(frames), lambda: plan.evaluate_noise(batch)]
        for operation in operations:
            with self.assertRaisesRegex(ValueError, 'options changed'):
                operation()
        with self.assertRaisesRegex(ValueError, 'options changed'):
            CudaExperiment(plan.compiled)

    def test_rejects_history_conditional_and_importance_paths(self):
        cases = [
            (parse_circuit('H 0\nM 0'), {'task': {'sample_records': [0]}}),
            (parse_circuit('M 0\nX_ERROR(1/4) rec[-1] 1'), {}),
            (parse_circuit('X_ERROR(1/4) 0'), {'sampling': {'method': 'importance', 'proposal_scale': '2'}}),
        ]
        for circuit, kwargs in cases:
            with self.subTest(circuit=circuit):
                compiled = compile_circuit(circuit, **kwargs)
                with self.assertRaises(ValueError):
                    CudaExperiment(compiled)

    def test_native_buffer_validation(self):
        plan = self.magic_plan()
        bad = [([0, 2], [0]), ([0, 1], [plan.actions]), ([1], []), ([0, 2, 1], [0])]
        for offsets, indices in bad:
            with self.assertRaises(ValueError):
                _native.cuda_map_frames(plan.handle, np.array(offsets, dtype='<u8').tobytes(),
                                        np.array(indices, dtype='<u4').tobytes())
        with self.assertRaises(ValueError):
            _native.cuda_map_frames(plan.handle, b'x', b'')
        with self.assertRaises(ValueError):
            _native.cuda_evaluate_frames(plan.handle, b'x')


class CudaKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def test_packed_wide_y_frames_affine_offset_tail_and_xor_cancellation(self):
        # Three words per X/Z: includes bit 130, simultaneous X/Z (Y),
        # duplicate actions, empty shots and a nonzero affine offset.
        columns = np.array([[1, 0, 4, 1, 0, 4], [2**63, 3, 0, 0, 2**63, 1],
                            [7, 11, 13, 17, 19, 23]], dtype=np.uint64)
        base = np.array([1, 0, 0, 0, 0, 2], dtype=np.uint64)
        rows = [[], [0], [1], [0, 0], [0, 1, 2], [2, 1, 2]] * 45 + [[2]]
        offsets, indices = [0], []
        expected = []
        for row in rows:
            indices.extend(row)
            offsets.append(len(indices))
            # Independent Python integer XOR oracle, no NumPy reduction.
            result = list(map(int, base))
            for action in row:
                for word, column in enumerate(columns[action]):
                    result[word] ^= int(column)
            expected.append(result)
        mapper = CudaFrameMapper(columns, base)
        np.testing.assert_array_equal(mapper.map(offsets, indices), np.array(expected, dtype=np.uint64))
        device = mapper.prepare_device(offsets, indices)
        np.testing.assert_array_equal(mapper.cp.asnumpy(mapper.launch(device)), expected)
        self.assertEqual(mapper.map([0], []).shape, (0, 6))
        other = CudaFrameMapper(columns, base)
        with self.assertRaisesRegex(ValueError, 'different mapper'):
            other.launch(device)

    def test_gpu_frames_match_original_exact_magic_sampler(self):
        plan = NativeCudaBridgeTests.magic_plan()
        rng = Random(4109)
        batch = plan.draw([rng.getrandbits(64) for _ in range(265)])
        frames = plan.gpu_frames(batch)
        np.testing.assert_array_equal(frames, plan.cpu_frames(batch))
        self.assertEqual(plan.evaluate_frames(frames), plan.evaluate_noise(batch))


if __name__ == '__main__':
    unittest.main()
