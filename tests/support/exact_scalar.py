"""Exact Q(omega), omega=exp(i*pi/4), with explicit binary scale."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from .quadratic import multiply


def rotate(a, exponent):
    for _ in range(exponent % 8):
        a = (-a[3], a[0], a[1], a[2])
    return a


@dataclass(frozen=True)
class ExactScalar:
    coefficients: tuple = (0, 0, 0, 0)
    power: int = 0

    def __post_init__(self):
        c = tuple(self.coefficients)
        if len(c) != 4 or any(type(v) is not int for v in c) or type(self.power) is not int:
            raise ValueError('four integer cyclotomic coefficients and integer exponent required')
        e = self.power
        if not any(c):
            e = 0
        else:
            while all(v % 2 == 0 for v in c):
                c = tuple(v // 2 for v in c); e += 1
        object.__setattr__(self, 'coefficients', c)
        object.__setattr__(self, 'power', e)

    def __bool__(self):
        return any(self.coefficients)

    def __add__(self, other):
        e = min(self.power, other.power)
        return ExactScalar(tuple((a << (self.power-e)) + (b << (other.power-e))
                                 for a, b in zip(self.coefficients, other.coefficients)), e)

    def __neg__(self):
        return self.scaled(-1)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return ExactScalar(multiply(self.coefficients, other.coefficients), self.power+other.power)

    def conjugate(self):
        a, b, c, d = self.coefficients
        return ExactScalar((a, -d, -c, -b), self.power)

    def scaled(self, n):
        return ExactScalar(tuple(n*x for x in self.coefficients), self.power)

    def real_decimal(self):
        a, b, c, d = self.coefficients
        if c or b != -d:
            raise ArithmeticError('expected an exactly real scalar')
        digits = max(80, int(max(abs(x).bit_length() for x in self.coefficients)*.302)+40)
        with localcontext() as ctx:
            ctx.prec = digits
            return +(Decimal(a)+Decimal(b)*Decimal(2).sqrt())*(Decimal(2)**self.power)

    def ratio(self, denominator):
        with localcontext() as ctx:
            ctx.prec = 80
            b = ExactScalar(denominator.coefficients, 0).real_decimal()
            if b <= 0:
                raise ArithmeticError('nonpositive denominator')
            a = ExactScalar(self.coefficients, self.power-denominator.power).real_decimal()
            return float(a/b)

    def log2(self):
        value = ExactScalar(self.coefficients, 0).real_decimal()
        if value < 0:
            raise ArithmeticError('negative probability')
        return float(value.ln()/Decimal(2).ln())+self.power if value else float('-inf')

    def complex(self):
        import cmath
        return sum(c*cmath.exp(1j*j*cmath.pi/4) for j, c in enumerate(self.coefficients))*2.**self.power

    def to_dict(self):
        return dict(coefficients=list(self.coefficients), power=self.power)


ONE = ExactScalar((1, 0, 0, 0))


def scalar(gaussian, phase, sqrt2_power):
    coeff = rotate((gaussian[0], 0, gaussian[1], 0), phase)
    if sqrt2_power & 1:
        coeff = tuple(a-b for a, b in zip(rotate(coeff, 1), rotate(coeff, 3)))
    return ExactScalar(coeff, sqrt2_power//2)
