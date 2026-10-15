"""Independent retained causal schedule for native operation sampler validation."""
from collections import OrderedDict
from copy import deepcopy
from types import MappingProxyType
from magna.boundary_sampling import BoundaryTask, BoundaryDraw, NativeBoundaryTrace
from magna.feedback import ControlRule
from magna.ir import InstrumentIR, PreparedInstrument, SampleAssignment, validate_assignment
from magna.scalar import ONE
from tests.support.global_sampling import real_sign
from tests.support.operation_hooks import (boundary_coordinates, feedback_axes, is_affine_control, record_row,
                                           resolve_feedback)


class ReferenceBoundarySampler:
    """Deterministic IR routing to one shared C++ Bell-contraction backend.

    Opaque controllers need a PhaseFeedback unitary-family contract. Their
    actual emitted gates are checked on every reached history. This is not an
    automatic proof that arbitrary user Python code satisfies its declaration.
    """
    def __init__(self, source, task=BoundaryTask(), *, max_plans=4, **limits):
        self.prepared = PreparedInstrument(getattr(source, 'ir', source))
        self.ir = self.prepared.ir; self.task = task; self.limits = limits
        if type(max_plans) is not int or max_plans < 1: raise ValueError('positive plan cache bound required')
        self.max_plans = max_plans; self.plans = OrderedDict()
        self.direct = tuple(oid for oid, node in self.ir.controls.items() if is_affine_control(node))
        self.complex = tuple(op['id'] for op in self.ir.operations
                             if op['kind'] == 'CONTROL' and op['id'] not in self.direct)
        for oid in self.complex:
            node = self.ir.controls[oid]
            if type(node.callback) is not ControlRule and node.feedback is None:
                raise NotImplementedError('complex callback needs a declared unitary PhaseFeedback family: '+oid)
        required = {r for oid in self.complex for r in self.ir.controls[oid].parents}
        self.required = tuple(r for r in self.ir.record_names if r in required)
        self.layout = boundary_coordinates(self.ir.record_names, task.postselection, task.outcomes, self.required)
        # Error events have their own query boundary; their free coordinates
        # are traced for A and constrained only for the accepted-good query.
        for x, z in task.observables:
            if type(x) is not int or type(z) is not int or x < 0 or z < 0 or (x | z) >> len(self.ir.outputs):
                raise ValueError('Pauli outside physical output')

    @property
    def route(self):
        return 'complex-feedback' if self.required else 'direct'

    def _postselection(self, engine, layout):
        if not layout['compatible']:
            # A contradictory zero row is an exact zero-mass boundary.
            return (0,), (1,)
        mapped = tuple(record_row(engine.deferred, tuple(r for j, r in enumerate(self.ir.record_names)
            if mask >> j & 1)) for mask in layout['postselection_rows'])
        return mapped, layout['postselection_values']

    def _resolved_ir(self, resolved):
        ops = []; controls = {}; ids = {o['id'] for o in self.ir.operations}
        def append(oid, gates):
            for j, gate in enumerate(gates):
                key = f'__boundary/{oid}/{j}'
                while key in ids: key = '_'+key
                ids.add(key); ops.append(dict(gate, id=key))
            ops.append(dict(id=oid, kind='I', qubits=()))
        for op in self.ir.operations:
            oid = op['id']
            if op['kind'] != 'CONTROL': ops.append(op)
            elif oid in self.direct:
                ops.append(op); controls[oid] = self.ir.controls[oid]
            elif oid in resolved: append(oid, resolved[oid])
            else:
                spec = self.ir.controls[oid].feedback
                gates = () if spec is None else tuple(dict(kind='PHASE', qubits=axis, exponent=e)
                    for axis, e in zip(*feedback_axes(spec)) if e)
                append(oid, gates)
        return InstrumentIR(self.ir.inputs, self.ir.outputs, tuple(ops), self.ir.initial,
                            controls, self.ir.name+'-Bell-query')

    def _engine(self, resolved):
        ir = self._resolved_ir(resolved); key = ir.fingerprint
        if key not in self.plans:
            self.plans[key] = NativeBoundaryTrace(ir, **self.limits)
            while len(self.plans) > self.max_plans: self.plans.popitem(last=False)
        else: self.plans.move_to_end(key)
        return self.plans[key]

    def sample(self, rng, *, faults=(), report_flips=()):
        """Sample only necessary feedback; sum every other record in queries.

        Before the earliest unresolved complex control, all requested parents
        are earlier records. Replacing later unitary controls by declared
        unitary baselines is a trace-preserving completion of that marginal.
        Future postselection is never included in these sampling queries.
        """
        faults = tuple(faults); report_flips = tuple(report_flips)
        flips, _ = validate_assignment(self.ir, SampleAssignment(
            dict.fromkeys(self.ir.record_names, 0), faults, report_flips), self.prepared)
        reports = {}; resolved = {}; stages = []; mass = ONE
        for oid in self.complex:
            node = self.ir.controls[oid]
            missing = tuple(r for r in node.parents if r not in reports)
            if missing:
                engine = self._engine(resolved); packed = engine.pack(faults, report_flips)
                rows = engine.rows((r,) for r in missing)
                events = engine.rows((r,) for r in reports)
                word, mass, info = engine.sample_rows(packed, rows, rng,
                    event_rows=events, outcomes=tuple(reports.values()))
                reports.update((r, word >> j & 1) for j, r in enumerate(missing))
                stages.append(dict(control=oid, records=missing,
                    completion='whole-instrument unitary future; no postselection', **info))
            gates = deepcopy(tuple(node.callback(MappingProxyType({r: reports[r] for r in node.parents}))))
            if any('record' in g or g['kind'] == 'CONTROL' for g in gates):
                raise ValueError('control expansion must be measurement-free')
            if node.feedback is not None: resolve_feedback(node.feedback, gates)
            resolved[oid] = gates
        engine = self._engine(resolved); packed = engine.pack(faults, report_flips)
        known_rows = engine.rows((r,) for r in reports); known_bits = tuple(reports.values())
        base = engine.query(packed, event_rows=known_rows, outcomes=known_bits)
        actual_mass = base.total()
        if actual_mass != mass or real_sign(mass) <= 0:
            raise ArithmeticError('completed actual instrument disagrees with sampled history mass')
        selection, rhs = self._postselection(engine, self.layout)
        events = known_rows+selection
        values = known_bits+rhs
        a = engine.query(packed, event_rows=events, outcomes=values)
        acceptance = a.total(); error = None
        resources = [base.info, a.info]
        if real_sign(acceptance) < 0 or real_sign(mass-acceptance) < 0:
            raise ArithmeticError('postselection mass outside history mass')
        moments = []
        for x, z in self.task.observables:
            moment = engine.query(packed, event_rows=events, outcomes=values, x=x, z=z)
            moments.append(moment.total()); resources.append(moment.info)
        return BoundaryDraw(mass, acceptance, error, tuple(moments), reports,
            {r: b ^ int(r in flips) for r, b in reports.items()},
            dict(route=self.route, stages=stages, final_resources=resources,
                 controls_executed=len(resolved), sampled_records=len(reports),
                 unsampled_records=len(self.ir.record_names)-len(reports)))
