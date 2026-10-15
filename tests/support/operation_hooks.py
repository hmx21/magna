"""Test access to operation-level internals and to native helpers production no longer calls.

Tests use these to inspect plan and cache counters, to evaluate algebraic reference values
and to check feedback declarations independently. The native entry points live in the
test-only ``_native_testing`` module; none of this is part of the installed package.
"""
from magna import _native
from magna.feedback import ControlRule, FeedbackMismatch
from magna.native import ket_data
from magna.noise import _rational
from magna.scalar import ExactScalar
from tests.support import native_testing


def polynomial_quotient(n, terms):
    return native_testing.load().phase_polynomial(n, tuple(terms))


def restrict(linear, upper, shift, basis):
    q0, ln, bn = native_testing.load().quadratic_restrict(linear, upper, shift, basis)
    return q0, list(ln), list(bn)


def overlap(a, b):
    return ExactScalar(*native_testing.load().overlap(ket_data(a), ket_data(b)))


def solve(rows, rhs, n):
    """Least-first pivots; return particular, kernel and augmented reduced rows."""
    value = native_testing.load().gf2_solve(rows, rhs, n)
    return None if value is None else (value[0], list(value[1]), list(value[2]))


def right_inverse_columns(rows, n):
    return native_testing.load().gf2_right_inverse(rows, n)


def scalar(gaussian, phase, sqrt2_power):
    """(a + b i) omega**phase sqrt(2)**sqrt2_power for gaussian = (a, b)."""
    return ExactScalar(*native_testing.load().gaussian_scalar(tuple(gaussian), (phase, sqrt2_power)))


def weighted_choice(rng, probabilities):
    """Draw exactly from rational probabilities, with residual identity as -1."""
    return native_testing.load().model_weighted_choice(rng, tuple(_rational(p) for p in probabilities))


def boundary_coordinates(records, postselection=(), outcomes=(), feedback=()):
    """Abstract invertible GF(2) record change; no synthesized CNOT circuit.

    First independent postselection rows, then needed feedback information,
    then a deterministic complement to trace. Redundancies retain their RHS
    consistency check. This classification does not inspect record randomness.
    """
    return native_testing.load().native_boundary_coordinates(tuple(records),
        tuple(tuple(row) for row in postselection), tuple(outcomes), tuple(feedback))


def total_plans(trace):
    """Number of scalar-query layouts a NativeBoundaryTrace has cached."""
    return native_testing.load().boundary_service_counts(trace._service)[1]


def record_row(deferred, names):
    """Boundary parity of the named records of a DeferredInstrument."""
    return native_testing.load().indexed_record_row(tuple(name for name, _ in deferred.record_wires),
                                                    deferred.original_output_count, tuple(names))


def is_affine_control(node):
    """Read the declared rule, never probe an opaque callback."""
    return type(node.callback) is ControlRule and native_testing.load().is_affine_rule(node.callback._native_gates())


def feedback_axes(spec):
    """The phase axes of a PhaseFeedback family and the baseline exponent parity of each."""
    return native_testing.load().phase_feedback_metadata((spec.qubits, spec.odd_axes, spec.even_axes))


def resolve_feedback(spec, gates):
    """Return D then X, retaining the exact omega**constant scalar."""
    result = native_testing.load().resolve_phase_feedback((spec.qubits, spec.odd_axes, spec.even_axes), tuple(gates))
    if result[0] is None: raise FeedbackMismatch(result[1])
    return result


def emit(gate, reports):
    """The gate a ControlGate applies for these reported records, or None."""
    result = _native.declared_rule_operation('evaluate', (gate._native_data(),), tuple(reports.items()))
    return result[0] if result else None


def schedule_stats(feedback):
    """Counters of a PreparedFeedback schedule."""
    return native_testing.load().branch_schedule_stats(feedback._handle)


def cache_info(cache):
    return native_testing.load().certificate_cache_info(cache._handle)


def cache_store(cache):
    """Keys of every cached certificate."""
    return tuple(native_testing.load().certificate_cache_keys(cache._handle)[0])


def cache_probabilities(cache):
    """Keys of the cached certificates that store a probability."""
    return tuple(native_testing.load().certificate_cache_keys(cache._handle)[1])


def cache_clear(cache):
    native_testing.load().certificate_cache_clear(cache._handle)
