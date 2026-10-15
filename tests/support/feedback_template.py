"""Test view of the fixed file-control template; all external decisions are Pauli assignments.

Only declared actions and selection boundaries are inspected. Controllers execute on every
physical draw, in causal order, and their return values are never cached here.
Non-Pauli Clifford actions use a fixed controlled gate on a traced virtual wire.
Nonlinear selections use separate virtual wires, projected only in final events.
"""
from tests.support import native_testing


class FeedbackTemplate:
    def __init__(self, circuit, *, minimum_width=0, control_metadata=None):
        from magna import _native
        from magna.circuit_file import _circuit_handle, Target
        self._native_handle, data = native_testing.load().make_feedback_template(_circuit_handle(circuit), minimum_width)
        self.wire_count = data['wire_count']
        self.guards = data['guards']
        self.virtual = data['virtual']
        self.selections = {anchor: (wire, Target(*target), record, bit)
            for anchor, (wire, target, record, bit) in data['selections'].items()}
        self.selection_offsets = data['selection_offsets']
        self.updates = tuple((anchor, wire, pauli, tuple(Target(*t) for t in terms))
            for anchor, wire, pauli, terms in data['updates'])
        self.control_metadata = control_metadata

    def faults(self, values, *, noise=None, prefix='', wires=None):
        from magna import _native
        from magna.circuit_file import _target_data
        return native_testing.load().feedback_template_faults(self._native_handle,
            tuple((_target_data(t), bit) for t, bit in values.items()),
            (), prefix,
            None if wires is None else tuple(wires.items() if isinstance(wires, dict) else enumerate(wires)))


class AssignedBoundaryTrace:
    """Per-query Pauli assignments over one immutable native preparation.

    The view owns no compiler/cache state. Fault transport, ordered phases,
    report flips and quantum evaluation all use the existing native engine.
    """
    def __init__(self, engine, paulis):
        self.engine = engine
        self.paulis = paulis

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def pack(self, faults=(), report_flips=()):
        return self.engine.pack(self.paulis+tuple(faults), report_flips)
