"""Separated certificate values and native whole-boundary absorption."""
from dataclasses import dataclass
from .normal import NormalForm
from . import _native


@dataclass(frozen=True)
class QuadraticPhaseTensor:
    n: int
    linear: tuple
    upper: tuple
    terms: tuple
    constant: int
    power: int
    boundary: tuple
    output_count: int
    environment: tuple
    zero: bool = False
    certificates: tuple = ()

    def to_normal(self):
        """Materialize the established public certificate format only on demand."""
        terms = _native.phase_value_operation('parities', native_tensor(self))
        return NormalForm(self.n, terms, self.constant, self.power, self.boundary,
                          self.output_count, self.environment, self.zero, self.certificates)



def native_tensor(tensor):
    """Serialize either public certificate representation for native numerics."""
    raw = {key: getattr(tensor, key) for key in
           ('n', 'terms', 'constant', 'power', 'boundary', 'output_count', 'environment', 'zero')}
    raw['environment'] = tuple(map(str, raw['environment']))
    raw['linear'] = getattr(tensor, 'linear', (0,)*tensor.n)
    raw['upper'] = getattr(tensor, 'upper', (0,)*tensor.n)
    return raw


def absorb(tensor):
    from . import _native
    raw = _native.phase_absorb(native_tensor(tensor))
    raw['environment'] = tensor.environment
    raw['certificates'] = tensor.certificates + raw['certificates']
    return QuadraticPhaseTensor(**raw)
