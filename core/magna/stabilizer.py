"""Affine quadratic stabilizer kets and their exact native operations.

The internal-variable elimination is tested independently against dense sums.
No record tree or physical-time quantum state evolution is used.
"""
from dataclasses import dataclass
from . import _native
from .gf2 import image_basis
from .scalar import ExactScalar
from .native import ket_data


@dataclass(frozen=True)
class StabilizerKet:
    """Amplitude 2**(power/2) omega**phase i**Q(u) at base+B u.

    Columns are independent; they describe support, not an enumerated table.
    This is an unnormalized ket; postselection retains its original scalar.
    """
    width: int
    base: int
    columns: tuple
    linear: tuple
    upper: tuple
    power: int = 0
    phase: int = 0

    def __post_init__(self):
        for key in ('columns', 'linear', 'upper'):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        _native.algebra_ket('validate', ket_data(self))

    @classmethod
    def positive(cls, width, basis=(), base=0):
        basis = tuple(image_basis(basis))
        if base < 0 or base >> width or any(v < 0 or v >> width for v in basis):
            raise ValueError('input support outside register')
        return cls(width, base, basis, (0,)*len(basis), (0,)*len(basis), -len(basis))

    @classmethod
    def contract(cls, width, base, columns, linear, upper, power=0, phase=0):
        """Sum internal paths by quadratic elimination, keeping boundary legs."""
        raw = _native.contract((width, base, tuple(columns), tuple(linear), tuple(upper), power, phase))
        return None if raw is None else cls(*raw)

    def norm(self):
        return ExactScalar(*_native.algebra_ket('norm', ket_data(self)))

    def project(self, rows, outcomes):
        raw = _native.algebra_ket('project', ket_data(self), rows, outcomes)
        return None if raw is None else StabilizerKet(*raw)

    def sample_rows(self, rows, rng):
        return list(_native.algebra_ket('sample', ket_data(self), rows, rng))

    def pauli(self, x=0, z=0):
        """Apply X**x Z**z with the original global Pauli convention."""
        return StabilizerKet(*_native.algebra_ket('pauli', ket_data(self), x, z))
