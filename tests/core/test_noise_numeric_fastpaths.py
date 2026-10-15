"""Distribution oracles and independently rounded exact-value comparisons."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
from random import Random
import math
import shutil
import subprocess
import tempfile
import unittest

from magna import _native
from tests.support import native_testing

ROOT = Path(__file__).resolve().parents[2]


class NoiseNumericFastpathTests(unittest.TestCase):
    def test_rational_ratios_against_python_fraction_rounding(self):
        rng = Random(20260927)
        pairs = []; expected = []
        # Normal hardware division and the large-coefficient fallback.
        for bits in (1, 16, 52, 53, 54, 128, 257):
            for _ in range(50):
                n = rng.randrange(1, 1 << bits) * rng.choice((-1, 1))
                d = rng.randrange(1, 1 << bits)
                a, b = rng.randrange(-500000, 500001), rng.randrange(-500000, 500001)
                pairs.append((((n, 0, 0, 0), a), ((d, 0, 0, 0), b)))
                mantissa, exponent = math.frexp(float(Fraction(n, d)))
                expected.append((mantissa, exponent+a-b))
        self.assertEqual(native_testing.load().file_scalar_ratios(pairs), tuple(expected))

    def test_identical_irrational_coefficients_and_invalid_denominators(self):
        c = (3, -2, 0, 2)  # Positive, with cancellation.
        negative = tuple(-v for v in c)
        pairs = (((c, -1000000), (c, 1000000)), ((negative, 50), (c, -50)))
        self.assertEqual(native_testing.load().file_scalar_ratios(pairs), ((.5, -1999999), (-.5, 101)))
        for denominator in ((negative, 0), ((1, 0, 1, 0), 0)):
            with self.assertRaises((ValueError, RuntimeError)):
                native_testing.load().file_scalar_ratios(((((0, 0, 0, 0), 0), denominator),))

    def test_depolarizing_and_nonuniform_channels_match_declared_law(self):
        groups = (((1, 30),)*15, ((0, 1), (1, 10), (1, 5), (0, 1), (3, 10)),
                  ((1, 1), (0, 1)), ())
        plan = native_testing.load().make_file_noise(groups)
        shots = 32768
        rng = Random(13); seeds = [rng.getrandbits(64) for _ in range(shots)]
        draws = native_testing.load().draw_file_noise(plan, seeds)
        laws = ([.5]+[1/30]*15, [.4, 0, .1, .2, 0, .3])
        for index, law in enumerate(laws):
            counts = Counter(row[0][index]+1 for row in draws)
            for label, p in enumerate(law):
                if not p: self.assertEqual(counts[label], 0)
                else: self.assertLess(abs(counts[label]-shots*p), 6*math.sqrt(shots*p*(1-p))+2)
        joint = sum(row[0][0] >= 0 and row[0][1] >= 0 for row in draws)
        self.assertLess(abs(joint-shots*.3), 6*math.sqrt(shots*.3*.7))
        self.assertTrue(all(row[0][2:]==(0, -1) and row[1]==(1, 1) for row in draws))
        # Batch partitioning and dense/sparse callers must not alter a draw.
        self.assertEqual(draws[:41], native_testing.load().draw_file_noise(plan, seeds[:17])+
                         native_testing.load().draw_file_noise(plan, seeds[17:41]))

    def test_finite_noise_thresholds_and_general_conversion_in_native_module(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None: self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory)/'rational-random'
            subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Wextra',
                '-I'+str(ROOT/'core/magna/native'),
                str(ROOT/'tests/native/rational_random.cpp'), '-o', str(exe)],
                check=True, capture_output=True, text=True, timeout=30)
            subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=10)


if __name__ == '__main__': unittest.main()
