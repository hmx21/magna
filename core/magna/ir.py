"""Protocol-independent original instrument and one complete assignment.

Callbacks see only declared earlier *reported* records. Their emitted gates
are instantiated before any quantum contraction. Faults identify original
operation IDs; readout flips change the report, never its quantum projector.
"""
from copy import deepcopy
from dataclasses import dataclass, field, asdict, replace
from fractions import Fraction
from hashlib import sha256
import json
from types import MappingProxyType
from .stabilizer import StabilizerKet
from .feedback import PhaseFeedback, ControlRule

VERSION = 'unified-instrument-v1'


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ControlNode:
    parents: tuple
    version: str
    callback: object = field(compare=False, repr=False)
    feedback: PhaseFeedback | None = None

    def __post_init__(self):
        if self.feedback is not None and not isinstance(self.feedback, PhaseFeedback):
            raise TypeError('feedback must be a PhaseFeedback declaration')
        if type(self.callback) is ControlRule and not self.callback.records <= set(self.parents):
            raise ValueError('control rule reads undeclared record parents')


@dataclass
class InstrumentIR:
    inputs: tuple
    outputs: tuple
    operations: tuple
    initial: StabilizerKet
    controls: dict = field(default_factory=dict)
    name: str = 'instrument'

    def __post_init__(self):
        self.inputs = tuple(self.inputs); self.outputs = tuple(self.outputs)
        self.operations = tuple(deepcopy(self.operations))
        self.controls = dict(self.controls)
        self._validate()

    @classmethod
    def _from_owned(cls, inputs, outputs, operations, initial, controls=None, name='instrument'):
        """Internal transfer of freshly built, unshared operation dictionaries.

        Public construction still snapshots arbitrary mutable input graphs.
        Callers must own every mutable leaf and must relinquish it here.
        Validation is identical to public construction.
        """
        result = cls.__new__(cls)
        result.inputs = tuple(inputs); result.outputs = tuple(outputs)
        result.operations = tuple(operations); result.initial = initial
        result.controls = dict(controls or {}); result.name = name
        result._validate()
        return result

    def _validate(self):
        from . import _native
        self._native_layout, self._record_names, self._operation_ids = _native.make_instrument_layout(
            self.inputs, self.outputs, (self.initial.width, self.initial.power, len(self.initial.columns)),
            tuple((op['id'], op['kind'], op.get('record')) for op in self.operations),
            tuple((oid, node.parents, node.version) for oid, node in self.controls.items()))

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items()
                if key not in ('_native_layout', '_record_names', '_operation_ids')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._validate()

    @property
    def record_names(self):
        return tuple(o['record'] for o in self.operations if 'record' in o)

    @property
    def fingerprint(self):
        return fingerprint((VERSION, self.inputs, self.outputs, self.operations, asdict(self.initial),
                            {k: ((v.parents, v.version) if v.feedback is None else
                             (v.parents, v.version, asdict(v.feedback))) +
                             ((asdict(v.callback),) if type(v.callback) is ControlRule else ())
                             for k, v in self.controls.items()}))


@dataclass(frozen=True)
class SampleAssignment:
    true: dict
    faults: tuple = ()
    report_flips: tuple = ()
    proposal_probability: Fraction = Fraction(1)
    noise_likelihood_ratio: Fraction = Fraction(1)
    sample_id: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'true', deepcopy(self.true))
        object.__setattr__(self, 'faults', tuple(deepcopy(self.faults)))
        object.__setattr__(self, 'report_flips', tuple(self.report_flips))
        for key in ('proposal_probability', 'noise_likelihood_ratio'):
            v = Fraction(getattr(self, key)); object.__setattr__(self, key, v)
            if v < 0 or (key == 'proposal_probability' and not 0 < v <= 1):
                raise ValueError('invalid proposal or noise likelihood ratio')


@dataclass(frozen=True)
class Branch:
    ir_fingerprint: str
    inputs: tuple
    outputs: tuple
    initial: StabilizerKet
    operations: tuple
    true: dict
    reported: dict
    decisions: tuple
    faults: tuple


class PreparedInstrument:
    """Owned snapshot, with metadata reused only while the mutable IR matches.

    Callback results are never cached here: each controlled branch still runs
    its callbacks, even when their declared versions have not changed.
    """
    def __init__(self, ir):
        owned_controls = {oid: replace(node, parents=tuple(node.parents)) for oid, node in ir.controls.items()}
        owned = InstrumentIR(ir.inputs, ir.outputs, ir.operations, ir.initial, owned_controls, ir.name)
        self._initialize(owned)

    @classmethod
    def _from_owned(cls, ir):
        """Internal metadata for an already owned IR; no extra snapshot."""
        result = cls.__new__(cls)
        result._initialize(ir)
        return result

    def _initialize(self, ir):
        self.ir = ir
        self.control_signature = self.controls(ir)
        self.fingerprint = self.ir.fingerprint
        self._native_layout = self.ir._native_layout
        self.record_names = self.ir._record_names
        self.record_index = MappingProxyType({r: j for j,r in enumerate(self.record_names)})
        self.ids = frozenset(self.ir._operation_ids)
        self.records = frozenset(self.record_names)

    @staticmethod
    def controls(ir):
        return tuple((k, tuple(v.parents), v.version, id(v.callback), v.feedback) for k, v in ir.controls.items())

    def matches(self, ir):
        return (ir.inputs == self.ir.inputs and ir.outputs == self.ir.outputs
                and ir.initial == self.ir.initial and ir.operations == self.ir.operations
                and self.controls(ir) == self.control_signature)



def validate_assignment(ir, assignment, prepared=None):
    from .batch import CompactAssignment
    if isinstance(assignment, CompactAssignment):
        if prepared is None or assignment.fingerprint != prepared.fingerprint:
            raise ValueError('compact assignment belongs to a different IR snapshot')
    from . import _native
    _, _, flips, groups = _native.validate_instrument_assignment(
        prepared._native_layout if prepared is not None else PreparedInstrument(ir)._native_layout,
        tuple(assignment.true.items()), assignment.report_flips,
        tuple((f['after'], f['qubit'], f['pauli']) for f in assignment.faults))
    return set(flips), {oid: [assignment.faults[index] for index in indices] for oid, indices in groups}


def instantiate_branch(ir, assignment, *, prepared=None):
    from .native_feedback import PreparedFeedback
    return PreparedFeedback(prepared if prepared is not None else PreparedInstrument(ir)).extract(assignment).materialize()
