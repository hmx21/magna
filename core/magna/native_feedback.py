"""Language adapters for native causal decisions and literal branch exports."""
from copy import deepcopy
from . import _native
from .batch import CompactAssignment
from .feedback import ControlRule
from .ir import Branch


class BranchDecisions:
    __slots__ = ('owner', 'true', 'measured_true', 'reported', 'decisions',
                 'faults', 'original_faults', '_handle', '_branch', 'true_word', 'reported_word')

    def __init__(self, owner, true, original_faults, handle, data):
        self.owner = owner
        self.true = true
        self.measured_true = data['measured']
        self.reported = data['reported']
        self.decisions = data['decisions']
        self.original_faults = original_faults
        self.faults = tuple(original_faults[j] for j in data['fault_order'])
        self.true_word = data['true_word']
        self.reported_word = data['reported_word']
        self._handle = handle
        self._branch = None

    def materialize(self):
        """Export stored decisions; repeated requests never rerun controls."""
        if self._branch is None:
            prepared = self.owner.prepared
            operations = _native.materialize_collected_branch(self._handle, prepared.ir.operations)
            self._branch = Branch(prepared.fingerprint, prepared.ir.inputs, prepared.ir.outputs,
                prepared.ir.initial, operations, deepcopy(self.true), dict(self.reported),
                deepcopy(self.decisions), self.original_faults)
        return self._branch


class PreparedFeedback:
    """Owned native schedule and its Python callback/data adapters."""
    def __init__(self, prepared):
        self.prepared = prepared
        nodes = prepared.ir.controls
        self._handle = _native.make_branch_schedule(prepared._native_layout,
            tuple((oid, node.callback._native_gates() if type(node.callback) is ControlRule else None)
                  for oid, node in nodes.items()),
            {oid: node.callback for oid, node in nodes.items()})

    def extract(self, assignment):
        return self._collect(assignment)

    def _collect(self, assignment):
        compact = isinstance(assignment, CompactAssignment)
        if compact and assignment.fingerprint != self.prepared.fingerprint:
            raise ValueError('compact assignment belongs to a different IR snapshot')
        true = deepcopy(assignment.true)
        faults = assignment.faults if compact else tuple(deepcopy(assignment.faults))
        handle, data = _native.collect_branch(self._handle, tuple(true.items()), assignment.report_flips,
            tuple((f['after'], f['qubit'], f['pauli']) for f in faults))
        return BranchDecisions(self, true, faults, handle, data)
