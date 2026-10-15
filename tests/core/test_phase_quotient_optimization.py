"""Exact uncached oracles for sparse native quotient construction and reuse."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import unittest

from tests.support import native_testing


def normalize(terms):
    combined = Counter()
    for a, c in terms:
        combined[a] += c
    return tuple(sorted((a, c % 8) for a, c in combined.items() if a and c % 8))


def phase(terms, x):
    return sum(c*((a & x).bit_count() & 1) for a, c in terms) % 8


class PhaseQuotientOptimizationTests(unittest.TestCase):
    def check_quotient(self, n, terms, *, dense=True):
        result = native_testing.load().phase_polynomial(n, terms)
        if dense:
            axes, core, linear, upper, _ = result
            for x in range(1 << n):
                y = sum(((a & x).bit_count() & 1) << j for j, a in enumerate(axes))
                q = sum(2*linear[j]+4*((upper[j] & x).bit_count() & 1)
                        for j in range(n) if x >> j & 1)
                self.assertEqual(phase(terms, x), (q+phase(core, y)) % 8)
        return result

    def test_complete_small_signatures_include_repeated_indices(self):
        for n in range(4):
            monomials = [sum(1 << j for j in js) for d in (1, 2, 3)
                         for js in combinations(range(n), d)]
            for selection in range(1 << len(monomials)):
                terms = []
                for index, mask in enumerate(monomials):
                    if not selection >> index & 1:
                        continue
                    sub = mask
                    while sub:
                        terms.append((sub, 1 if sub.bit_count() & 1 else -1))
                        sub = (sub-1) & mask
                self.check_quotient(n, terms)

    def test_sign_even_and_duplicate_updates_do_not_contaminate_template(self):
        base = ((1, 1), (3, 3), (5, 5), (7, 7))
        original = self.check_quotient(4, base)
        for _ in range(2):
            for word in range(16):
                changed = tuple((a, -c if word >> j & 1 else c)
                                for j, (a, c) in enumerate(base))
                # The even-only form is absent from the cached odd key.
                changed += ((10, 2*(word % 4)), (3, 5), (3, -5))
                found = self.check_quotient(4, changed)
                self.assertEqual((found[0], found[1], found[4]),
                                 (original[0], original[1], original[4]))
        self.assertEqual(self.check_quotient(4, base), original)
        self.check_quotient(4, ((7, 1), (7, -1), (10, 6)))

    def test_dimension_key_eviction_and_thread_ownership(self):
        terms = ((1, 7), (2, 3), (3, 1))
        expected = self.check_quotient(3, terms)
        # More than eight distinct dimensions evicts the original entry.
        for n in range(4, 15):
            self.check_quotient(n, terms, dense=False)
        self.assertEqual(self.check_quotient(3, terms), expected)
        jobs = [(n, terms) for n in (3, 4, 65, 129)]*3
        with ThreadPoolExecutor(max_workers=3) as pool:
            found = list(pool.map(lambda args: native_testing.load().phase_polynomial(*args), jobs))
        for (n, ts), result in zip(jobs, found):
            self.assertEqual(result, self.check_quotient(n, ts, dense=False))

    def test_sparse_quotient_coordinates_cross_machine_word(self):
        n = 70
        terms = tuple((1 << j, 1+2*(j % 4)) for j in range(n))
        axes, core, linear, upper, lifts = native_testing.load().phase_polynomial(n, terms)
        identity = tuple(1 << j for j in range(n))
        self.assertEqual(axes, identity)
        self.assertEqual(lifts, identity)
        self.assertEqual(core, tuple((a, 1) for a in identity))
        self.assertEqual(linear, tuple(j % 4 for j in range(n)))
        self.assertEqual(upper, (0,)*n)

    def test_invalid_dimension_cannot_use_cached_key(self):
        self.check_quotient(3, ((1, 1),))
        for terms in (((8, 1),), ((1, 1), (8, 2))):
            with self.assertRaises(ValueError):
                native_testing.load().phase_polynomial(3, terms)


if __name__ == '__main__':
    unittest.main()
