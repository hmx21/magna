import cmath
import math
import random
import unittest

from magna.scalar import ExactScalar, ONE
from magna.file_api import ResourceLimit
from magna.normal import NormalForm
from tests.support.global_sampling import exact_bernoulli, real_sign
from tests.support.probability_tensor import (
    affine_probability_sampler, clifford_scalar, compress_probability, trace_normal,
)


class ConstantBits:
    def __init__(self, bit):
        self.bit = bit
        self.calls = 0

    def getrandbits(self, n):
        assert n == 1
        self.calls += 1
        return self.bit


class GlobalSamplingTests(unittest.TestCase):
    @staticmethod
    def dense_normal(normal):
        values = [0j]*(1 << len(normal.boundary))
        if normal.zero:
            return values
        for u in range(1 << normal.n):
            word = sum((((a & u).bit_count() % 2) ^ b) << j for j, (a, b) in enumerate(normal.boundary))
            phase = normal.constant+sum(c*((a & u).bit_count() % 2) for a, c in normal.terms)
            values[word] += 2**(normal.power/2)*cmath.exp(1j*math.pi*phase/4)
        return values

    def test_trace_before_decomposition_dense_with_output_effect(self):
        # Non-injective boundary, internal interference, affine offsets and Y
        # signs; evaluate the definition independently only on this small case.
        normal = NormalForm(4, ((3, 1), (4, 7), (12, 2), (8, 4)), 3, -4,
                            ((1, 1), (6, 0), (8, 1)), 1, ('r0', 'r1'))
        dense = self.dense_normal(normal)
        for keep in ((), (1, 2), (3,)):
            for x, z in ((0, 0), (4, 0), (4, 4), (0, 6)):
                for bit in (0, 1):
                    tensor = trace_normal(normal, keep, event_rows=(1,), outcomes=(bit,), x=x, z=z)
                    actual = self.dense_normal(compress_probability(tensor))
                    expected = [0j]*len(actual)
                    for ket_word, amplitude in enumerate(dense):
                        bra_word = ket_word ^ x
                        if bra_word & 1 != bit:
                            continue
                        record = sum(((row & bra_word).bit_count() % 2) << j for j, row in enumerate(keep))
                        expected[record] += dense[bra_word].conjugate()*amplitude*(1j)**((x & z).bit_count())*(-1)**((z & ket_word).bit_count())
                    for a, e in zip(actual, expected):
                        self.assertAlmostEqual(abs(a-e), 0, places=12)

    def test_many_record_phase_resources_cancel_without_decomposition(self):
        from unittest.mock import patch
        # 128 phase-bearing record coordinates and one requested T output.
        # Mix record labels by a generic invertible CNOT map; no code pattern.
        n = 129
        boundary = [1 << j for j in range(n)]
        rng = random.Random(260916)
        for _ in range(512):
            a, b = rng.sample(range(n-1), 2)
            boundary[b] ^= boundary[a]
        normal = NormalForm(n, tuple((1 << j, 1) for j in range(n)), 0, -n,
                            tuple((a, 0) for a in [boundary[-1]]+boundary[:-1]), 1, tuple(range(n-1)))
        with patch('magna.evaluate.decompose_resource', side_effect=AssertionError('decomposition forbidden')):
            prob = compress_probability(trace_normal(normal, (6,)))
            mix = affine_probability_sampler(prob)
            self.assertEqual(prob.n, 1)
            self.assertEqual(prob.terms, ())
            self.assertEqual(mix.total_weight, ONE)
            self.assertEqual(len(mix.components[0][1].columns), 1)
            out = 1
            for x, z in ((out, 0), (out, out)):
                value = clifford_scalar(compress_probability(trace_normal(normal, x=x, z=z)))
                self.assertEqual(value, ExactScalar((0, 1, 0, -1), -1))
        with self.assertRaises(ResourceLimit):
            trace_normal(normal, max_doubled_path_bits=128)





    def test_exact_comparisons_and_extreme_probabilities(self):
        self.assertEqual(real_sign(ExactScalar((3, -2, 0, 2))), 1)
        self.assertEqual(real_sign(ExactScalar((1, -1, 0, 1))), -1)
        zero_bits = ConstantBits(0)
        self.assertEqual(exact_bernoulli(ExactScalar((1, 0, 0, 0), -2048), ONE, zero_bits), 1)
        self.assertEqual(zero_bits.calls, 2048)
        one_bits = ConstantBits(1)
        self.assertEqual(exact_bernoulli(ONE-ExactScalar((1, 0, 0, 0), -2048), ONE, one_bits), 0)
        self.assertEqual(one_bits.calls, 2048)
        for weight, answer in ((ExactScalar(), 0), (ONE, 1)):
            rng = ConstantBits(0)
            self.assertEqual(exact_bernoulli(weight, ONE, rng), answer)
            self.assertEqual(rng.calls, 0)



if __name__ == '__main__':
    unittest.main()
