"""Explicit coherent stabilizer decompositions and output insertions."""
from dataclasses import dataclass
from . import _native
from .native import ket_data, term_data, term_values
from .scalar import ExactScalar
from .stabilizer import StabilizerKet


@dataclass(frozen=True)
class StabilizerExpansion:
    terms: tuple  # (ExactScalar coefficient, unnormalized StabilizerKet)

    def norm(self):
        return ExactScalar(*_native.expansion_operation('norm', term_data(self.terms)))

    def pauli_expectation(self, x=0, z=0):
        """Unnormalized expectation of Hermitian i**popcount(x&z) X**x Z**z."""
        return ExactScalar(*_native.expansion_operation('pauli', term_data(self.terms), x, z))

    def transformed_pauli(self, x=0, z=0):
        return StabilizerExpansion(term_values(_native.expansion_operation('transform', term_data(self.terms), x, z)))


def decompose_resource(normal, backend='database', max_quotient_rank=8):
    """Expose the current native decomposition through the existing Python API."""
    if normal.zero:
        return StabilizerExpansion(()), dict(quotient_rank=0, constructive_terms=0, zero=True)
    import json
    import time
    started = time.perf_counter()
    from pathlib import Path
    from . import _native
    from .phase_tensor import native_tensor
    if backend not in ('database', 'fourier'): raise ValueError('backend must be database or fourier')
    database = (json.loads((Path(__file__).parent/'data/phase_decompositions.json').read_text())
                if backend == 'database' else None)
    _, resource, states, _ = _native.phase_decompose(native_tensor(normal),
        dict(max_quotient_rank=max_quotient_rank, max_terms=(1 << max_quotient_rank),
             max_pair_contractions=(1 << (2*max_quotient_rank))), database, False)
    for key in ('axes', 'section_lifts', 'core_phase_mod8', 'clifford_linear_mod4', 'clifford_quadratic_upper'):
        resource[key] = list(resource[key])
    resource['seconds'] = time.perf_counter()-started
    return StabilizerExpansion(tuple((ExactScalar(*c), StabilizerKet(*k)) for c, k in states)), resource


def contract_target(ket, target, output_count):
    """Apply <target| on the first output_count legs, retaining the environment."""
    value = _native.expansion_operation('contract_target', ket_data(ket), ket_data(target), output_count)
    return None if value is None else StabilizerKet(*value)


def target_weight(expansion, target, output_count):
    return ExactScalar(*_native.expansion_operation('target', term_data(expansion.terms), term_data(target.terms), output_count))
