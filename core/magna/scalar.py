"""Exact Q(omega), omega=exp(i*pi/4), with explicit binary scale."""
from dataclasses import dataclass
from decimal import Decimal
from . import _native


@dataclass(frozen=True)
class ExactScalar:
    coefficients: tuple = (0, 0, 0, 0)
    power: int = 0

    def __post_init__(self):
        c, e = _native.algebra_scalar('normalize', (tuple(self.coefficients), self.power))
        object.__setattr__(self, 'coefficients', c)
        object.__setattr__(self, 'power', e)

    def _native(self):
        return self.coefficients, self.power

    @classmethod
    def _result(cls, value):
        return cls(*value)

    def __bool__(self):
        return any(self.coefficients)

    def __add__(self, other):
        return self._result(_native.algebra_scalar('add', self._native(), other._native()))

    def __neg__(self):
        return self.scaled(-1)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return self._result(_native.algebra_scalar('multiply', self._native(), other._native()))

    def conjugate(self):
        return self._result(_native.algebra_scalar('conjugate', self._native()))

    def scaled(self, n):
        return self._result(_native.algebra_scalar('scale', self._native(), n))

    def real_decimal(self):
        return Decimal(_native.scalar_decimal_value(self._native()))

    def ratio(self, denominator):
        return _native.algebra_scalar('ratio', self._native(), denominator._native())

    def log2(self):
        return _native.algebra_scalar('log2', self._native())

    def complex(self):
        return _native.algebra_scalar('complex', self._native())

    def to_dict(self):
        return dict(coefficients=list(self.coefficients), power=self.power)


ONE = ExactScalar((1, 0, 0, 0))
