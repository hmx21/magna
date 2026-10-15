"""Exact scalar comparisons and affine record mixtures used by the reference samplers."""
from dataclasses import dataclass
from magna.scalar import ExactScalar
from tests.support import native_testing


def real_sign(value):
    return native_testing.load().real_sign((value.coefficients, value.power))


def exact_bernoulli(weight, total, rng):
    """Use the caller-owned one-bit stream with exact native comparisons."""
    return native_testing.load().algebra_bernoulli((weight.coefficients, weight.power),
                                                   (total.coefficients, total.power), rng)


@dataclass(frozen=True)
class AffineRecordMixture:
    """Certified nonnegative mixture of uniform affine record distributions."""
    components: tuple  # exact total component weight, affine-support ket
    total_weight: ExactScalar
    width: int
