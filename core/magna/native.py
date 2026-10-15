"""Shared native data conventions and compatibility with the unified kernel."""


def ket_data(k):
    return (k.width, k.base, k.columns, k.linear, k.upper, k.power, k.phase)


def term_data(terms):
    return tuple(((c.coefficients, c.power), ket_data(k)) for c, k in terms)


def term_values(raw):
    from .scalar import ExactScalar
    from .stabilizer import StabilizerKet
    return tuple((ExactScalar(*c), StabilizerKet(*k)) for c, k in raw)
