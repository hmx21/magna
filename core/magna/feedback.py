"""Declared monomial feedback families, checked against each actual callback.

The callback remains authoritative. A declaration only permits reuse of a
Clifford dilation; a mismatch sends that already expanded branch to the literal
compiler. No records, decoder results, or physical noise are synthesized here.
"""
from dataclasses import dataclass
from . import _native


@dataclass(frozen=True)
class ControlExpression:
    """Explicit arithmetic in Z/8Z, with bitwise Boolean operations.

    Records are bits. All intermediates are reduced modulo eight; guards use
    nonzero truth. This makes phase arithmetic finite without enumerating a
    report truth table. ``~x`` is the Boolean not of ``x`` (0 or 1).
    """
    operation: str
    arguments: tuple

    def __post_init__(self):
        object.__setattr__(self, 'arguments', tuple(self.arguments))
        op, args = self.operation, self.arguments
        if op == 'const' and (len(args) != 1 or type(args[0]) is not int):
            raise TypeError('integer control constant required')
        if op == 'record' and (len(args) != 1 or not isinstance(args[0], str)):
            raise TypeError('record name must be a string')
        if op not in ('const', 'record') and any(type(a) is not ControlExpression for a in args):
            raise TypeError('control expression operands must be expressions')
        handle, constant = _native.make_expression(op,
            args if op in ('const', 'record') else tuple(a._native_handle for a in args))
        object.__setattr__(self, '_native_handle', handle)
        if op == 'const': object.__setattr__(self, 'arguments', (constant,))

    @staticmethod
    def _from_data(raw):
        op, args = raw
        return ControlExpression(op, args if op in ('const', 'record') else
            tuple(ControlExpression._from_data(a) for a in args))

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def constant(value): return ControlExpression('const', (value,))
    @staticmethod
    def record(name): return ControlExpression('record', (name,))
    @staticmethod
    def coerce(value):
        return value if type(value) is ControlExpression else ControlExpression.constant(value)

    @property
    def records(self):
        return frozenset(_native.expression_operation('records', self._native_handle))

    def evaluate(self, reports):
        return _native.expression_operation('evaluate', self._native_handle,
            tuple((r, reports[r]) for r in self.records))

    def _binary(self, operation, other):
        return ControlExpression(operation, (self, self.coerce(other)))
    def __xor__(self, other): return self._binary('xor', other)
    def __and__(self, other): return self._binary('and', other)
    def __or__(self, other): return self._binary('or', other)
    def __add__(self, other): return self._binary('add', other)
    def __mul__(self, other): return self._binary('mul', other)
    def __rxor__(self, other): return self.coerce(other).__xor__(self)
    def __rand__(self, other): return self.coerce(other).__and__(self)
    def __ror__(self, other): return self.coerce(other).__or__(self)
    def __radd__(self, other): return self.coerce(other).__add__(self)
    def __rmul__(self, other): return self.coerce(other).__mul__(self)
    def __invert__(self): return ControlExpression('not', (self,))


@dataclass(frozen=True)
class ControlGate:
    """One ordered conditional gate in an authoritative rule."""
    kind: str
    qubits: tuple
    exponent: ControlExpression | int | None = None
    when: ControlExpression | int = 1

    def __post_init__(self):
        object.__setattr__(self, 'qubits', tuple(self.qubits))
        if self.exponent is not None:
            object.__setattr__(self, 'exponent', ControlExpression.coerce(self.exponent))
        object.__setattr__(self, 'when', ControlExpression.coerce(self.when))
        _native.declared_rule_operation('validate', (self._native_data(),))

    def _native_data(self):
        return (self.kind, self.qubits, None if self.exponent is None else self.exponent._native_handle,
                self.when._native_handle)

    @property
    def records(self):
        return frozenset(_native.declared_rule_operation('records', (self._native_data(),)))


@dataclass(frozen=True)
class ControlRule:
    """Explicit, immutable controller; ordinary Python callbacks stay opaque.

    Passing this object as ``ControlNode.callback`` authorizes its serialized
    execution. It is itself the callback, so native execution cannot bypass a
    different user callback or infer semantics from a closure/version string.
    """
    gates: tuple

    def __post_init__(self):
        object.__setattr__(self, 'gates', tuple(self.gates))
        if any(type(g) is not ControlGate for g in self.gates):
            raise TypeError('ControlRule requires ordered ControlGate entries')

    @property
    def records(self):
        return frozenset(_native.declared_rule_operation('records', self._native_gates()))

    def _native_gates(self):
        return tuple(g._native_data() for g in self.gates)

    def __call__(self, reports):
        return _native.declared_rule_operation('evaluate', self._native_gates(),
            tuple(reports.items()))

class FeedbackMismatch(Exception):
    """An actual expansion is outside its optional prepared family."""


@dataclass(frozen=True)
class PhaseFeedback:
    """Paulis and parity phases on a fixed set of live wires.

    Single-wire phase axes are implicit. Extra parity axes may be declared in
    even_axes or odd_axes. The total exponent on each axis must have the stated
    parity after the actual callback has been normalized. Arbitrary classical
    functions of the declared earlier reports are allowed.
    """
    qubits: tuple
    odd_axes: tuple = ()
    even_axes: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'qubits', tuple(self.qubits))
        for field in ('odd_axes', 'even_axes'):
            object.__setattr__(self, field, tuple(tuple(a) for a in getattr(self, field)))
        _native.validate_phase_feedback((self.qubits, self.odd_axes, self.even_axes))
