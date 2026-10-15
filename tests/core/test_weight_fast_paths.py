"""Weight arithmetic checked against independent Python integers and Fraction."""
from fractions import Fraction
import math
from random import Random
import unittest
from magna import _native
from tests.support import native_testing


def scalar(n, power=0):
    return ((n, 0, 0, 0), power)


class WeightFastPathTests(unittest.TestCase):
    def test_count_weights_match_independent_products_across_group_sizes(self):
        # Multiple factor pairs, shared factors, deterministic endpoints and
        # groups too large for a prepared count table all retain exact payloads.
        for count in (1, 2, 17, 308, 512, 513):
            categories = [((1, 7),)]*count + [((1, 3), (1, 4))]*9
            factors = [((19, 23), (29, 31))]*count + [((19, 23), (37, 41))]*9
            categories += [((0, 1),)]*3 + [((1, 1),)]*3
            factors += [((43, 47), (53, 59))]*6
            plan = native_testing.load().make_file_noise(categories, factors)
            for choices, weight in native_testing.load().draw_file_noise(plan, tuple(range(9))):
                n = d = 1
                for choice, pair in zip(choices, factors):
                    ni, di = pair[choice >= 0];n *= ni;d *= di
                self.assertEqual(weight, (n, d))

    def test_large_rational_rounding_and_extreme_exponents(self):
        rng = Random(5641)
        cases = [(rng.getrandbits(n) or 1, rng.getrandbits(d) or 1)
                 for n in (54, 63, 64, 65, 127, 1024, 5000)
                 for d in (3, 53, 64, 65, 129, 3000) for _ in range(5)]
        # Exactly halfway and one integer unit to each side, both tie parities.
        for bits in (54, 100, 1024):
            for q in (2**52, 2**52+1, 2**53-1):
                cases += [((2*q+1)*2**bits+offset, 2**(bits+54)) for offset in (-1, 0, 1)]
        for n, d in cases:
            exponent = n.bit_length()-d.bit_length()+1
            normalized = Fraction(n, d) * (Fraction(1, 2**exponent) if exponent >= 0 else 2**-exponent)
            mantissa, shift = math.frexp(float(normalized))
            for sign, power in ((1, 0), (-1, -99999), (1, 99999)):
                got, = native_testing.load().file_scalar_ratios(((scalar(sign*n, power), scalar(d)),))
                self.assertEqual(got, (sign*mantissa, exponent+shift+power), (n, d, power))

    def test_grouped_factors_preserve_choices_and_unreduced_weights(self):
        categories = (((1, 17),), ((2, 11), (3, 11)), ((0, 1),), ((1, 1),))*79
        factors = (((17, 19), (29, 31)), ((37, 41), (29, 31)),
                   ((1, 1), (0, 1)), ((5, 7), (0, 1)))*79
        # Also include a unit value encoded as an unreduced rational.
        factors = list(factors);factors[3] = ((5, 7), (3, 3))
        plain = native_testing.load().make_file_noise(categories)
        weighted = native_testing.load().make_file_noise(categories, factors)
        for seed in range(15):
            a, = native_testing.load().draw_file_noise(plain, (seed,))
            b, = native_testing.load().draw_file_noise(weighted, (seed,))
            self.assertEqual(a[0], b[0])
            n = d = 1
            for choice, pair in zip(a[0], factors):
                ni, di = pair[choice >= 0]
                if ni != di:n *= ni;d *= di
            self.assertEqual(b[1], (n, d))
