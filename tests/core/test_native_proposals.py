"""Independent exact RNG and decimal-context checks for native public statistics."""
from decimal import Decimal, localcontext
from fractions import Fraction
from random import Random
from types import SimpleNamespace
import unittest

from magna.operations import (AffineProposal, IndependentProposal, InstrumentIR, StabilizerKet,
                                        ExactScalar, estimate)
from magna import _native
from tests.support.decimal_estimator import estimate as reference_estimate


class NativeProposalTests(unittest.TestCase):
    def instrument(self):
        return InstrumentIR((0,), (0,), tuple(dict(id=r, kind='MZ', qubits=[0], record=r)
            for r in ('a', 'b', 'c')), StabilizerKet.positive(1))

    def test_exact_independent_and_affine_rng_streams(self):
        ir = self.instrument(); probabilities = {'a': Fraction(2, 7), 'c': Fraction(3, 11)}
        proposal = IndependentProposal(probabilities); native = Random(824); reference = Random(824)
        for _ in range(30):
            truth = {}; q = Fraction(1)
            for name in ir.record_names:
                p = probabilities.get(name, Fraction(1, 2))
                bit = int(reference.randrange(p.denominator) < p.numerator)
                truth[name] = bit; q *= p if bit else 1-p
            draw = proposal.draw(ir, native)
            self.assertEqual((draw.true, draw.proposal_probability), (truth, q))
        self.assertEqual(native.getstate(), reference.getstate())
        proposal = AffineProposal(ir.record_names, (3,), (1,), coverage_certificate='finite test')
        for _ in range(30):
            word = reference.getrandbits(2); point = 1 ^ (3 if word & 1 else 0) ^ (4 if word & 2 else 0)
            draw = proposal.draw(ir, native)
            self.assertEqual(draw.true, {name: point >> j & 1 for j, name in enumerate(ir.record_names)})
            self.assertEqual(draw.proposal_probability, Fraction(1, 4))
        self.assertEqual(native.getstate(), reference.getstate())
        with self.assertRaises(ValueError):
            AffineProposal(ir.record_names, (3,), (2,), coverage_certificate='invalid')

    def test_real_decimal_and_weight_preserve_precision_and_rounding(self):
        rng = Random(432)
        for j in range(35):
            a, b = (rng.getrandbits(j*7) * (-1 if rng.getrandbits(1) else 1) for _ in range(2))
            value = ExactScalar((a, b, 0, -b), rng.randrange(-1000, 1001))
            a, b, _, _ = value.coefficients
            digits = max(80, int(max(abs(x).bit_length() for x in value.coefficients)*.302)+40)
            with localcontext() as ctx:
                ctx.prec = digits
                expected = +(Decimal(a)+Decimal(b)*Decimal(2).sqrt())*(Decimal(2)**value.power)
            self.assertEqual(value.real_decimal(), expected)
            ratio = Fraction(rng.randrange(1, 1000), rng.randrange(1, 1000))
            with localcontext() as ctx:
                ctx.prec = 80
                weighted = expected*Decimal(ratio.numerator)/Decimal(ratio.denominator)
            actual = Decimal(_native.scalar_decimal_value(value._native(), (ratio.numerator, ratio.denominator)))
            self.assertEqual(actual, weighted)

    def test_estimator_matches_independent_decimal_operations(self):
        for weights in (('0',), ('1',), ('0', '1', '2', '3.141592653589793238462643383279'),
                        ('1E-2000', '2E-2000', '3E-2000'), ('1E2000', '2E2000', '3E2000'),
                        ('1', '1.'+'0'*77+'1', '1.'+'0'*77+'2')):
            rows = [SimpleNamespace(compiled=True, weight_decimal=lambda w=w: Decimal(w),
                a=Decimal(j % 3)/2, d=Decimal(j % 2)) for j, w in enumerate(weights)]
            score = lambda row: (row.a, row.d)
            self.assertEqual(estimate(rows, score), reference_estimate(rows, score))


if __name__ == '__main__': unittest.main()
