"""Algebraic probability trace BEFORE any stabilizer decomposition.

Input: one fixed-preparation, whole-instrument parity-phase path tensor with
record copies and original environment on its boundary. Output amplitudes are
classical probability contributions, never quantities to square again.
No phase truth table, ket-amplitude enumeration, or rank decomposition is used.
"""
from magna import _native
from magna.normal import absorb_boundaries
from magna.scalar import ExactScalar
from magna.stabilizer import StabilizerKet
from magna.phase_tensor import native_tensor
from tests.support import native_testing
from tests.support.global_sampling import AffineRecordMixture


def trace_normal(normal, keep_rows=(), *, event_rows=(), outcomes=(), x=0, z=0,
                 max_doubled_path_bits=4096):
    """Symbolic tensor from the shared single-sided native trace reduction."""
    from magna.phase_tensor import QuadraticPhaseTensor
    raw = native_tensor(normal)
    result = native_testing.load().trace_probability_tensor(raw, tuple(keep_rows), tuple(event_rows),
        tuple(outcomes), x, z, max_doubled_path_bits)
    return QuadraticPhaseTensor(**result).to_normal()


def compress_probability(tensor):
    """Universal parity/GF(2)/Clifford elimination, no exponential table."""
    return absorb_boundaries(tensor)


def clifford_scalar(tensor):
    """Evaluate an already reduced scalar Clifford tensor; refuse resources.

    Suitable for signed Pauli contributions as well as event probabilities.
    It never invokes a stabilizer decomposition or an amplitude truth table.
    """
    return ExactScalar(*_native.phase_value_operation('scalar', native_tensor(tensor)))


def affine_probability_sampler(tensor):
    """Certify a single uniform affine probability tensor after compression.

    Returns None if non-Clifford resource or nonconstant phase remains. It does
    not attempt a high-rank decomposition to rescue a failed certificate.
    Call only on a probability tensor, not a signed observable contribution.
    """
    value = _native.phase_value_operation('affine', native_tensor(tensor))
    if value is None:
        return None
    total, support = value
    weight = ExactScalar(*total)
    return AffineRecordMixture(((weight, StabilizerKet(*support)),), weight, len(tensor.boundary))
