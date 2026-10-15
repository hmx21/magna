"""Test access to compiled-task internals: lowered IR, per-draw engines and draw summaries.

These helpers reach into private CompiledCircuit state so that regression tests can
compare the native runtime with independently lowered instruments. They are not part
of the installed package.
"""
from weakref import WeakValueDictionary

from magna import _native
from magna.boundary_sampling import NativeBoundaryTrace
from magna.circuit_file import _circuit_handle, _target_data
from magna.deferred_feedback import DeferredInstrument
from magna.feedback import ControlExpression, ControlGate, ControlRule
from magna.file_api import NoiseDraw, _NoiseSchedule, _format_weight_summary
from magna.ir import InstrumentIR, ControlNode
from magna.native_file_runtime import noise_data
from magna.stabilizer import StabilizerKet

from .feedback_template import AssignedBoundaryTrace, FeedbackTemplate
from tests.support import native_testing


def lower(circuit, outputs, values, noise, *, feedback_plan=None):
    """Lower one circuit file with fixed decisions and noise to an operation-level instrument."""
    ops, rules = native_testing.load().lower_circuit_file(_circuit_handle(circuit), tuple(outputs),
        tuple((_target_data(t), bit) for t, bit in values.items()),
        tuple((oid, tuple((g['kind'], g['qubits'][0]) for g in gates)) for oid, gates in noise.gates.items()),
        None if feedback_plan is None else feedback_plan._native_handle)
    controls = {oid: ControlNode(parents, 'magna-v1',
        ControlRule(tuple(ControlGate(g['kind'], tuple(g['qubits']), g.get('exponent'), ControlExpression._from_data(guard)) for g in gates)))
        for oid, parents, guard, gates in rules}
    return InstrumentIR._from_owned(tuple(range(circuit.num_qubits)), tuple(outputs), ops,
        StabilizerKet.positive(circuit.num_qubits), controls, name='magna-v1')


class SparseNoiseDraw(NoiseDraw):
    """Selected original actions; materialize mutable gate dictionaries only on access."""
    @classmethod
    def _from_grouped(cls, actions, flips, weight):
        result = object.__new__(cls)
        object.__setattr__(result, '_actions', actions)
        object.__setattr__(result, '_materialized', None)
        object.__setattr__(result, 'report_flips', tuple(flips))
        object.__setattr__(result, 'likelihood_ratio', weight)
        return result

    @property
    def gates(self):
        if self._materialized is None:
            object.__setattr__(self, '_materialized',
                {oid: tuple(dict(kind=p, qubits=[q]) for p, q in gates) for oid, gates in self._actions})
        return self._materialized


def draw_many(schedule, rng, count):
    """Original noise draws of a compiled noise schedule, one 64-bit seed from rng per draw."""
    return [SparseNoiseDraw._from_grouped(*row) if schedule.category_count else NoiseDraw({}, ()) for row in
            native_testing.load().draw_file_noise_random(schedule._native_plan, rng, count)]


def draw(schedule, rng):
    return draw_many(schedule, rng, 1)[0]


def weight_factors(schedule):
    """Exact physical/proposal likelihood factors of each noise category."""
    return native_testing.load().file_noise_factors(schedule._native_plan)


def draw_noise(circuit, rng):
    return draw(_NoiseSchedule(circuit), rng)


def attach_target(ideal_state, ir, record_offset):
    """Attach the target bra/ket effect as inverse U and fixed zero events."""
    first = 1+max((*ir.inputs, *ir.outputs, *(q for op in ir.operations for q in op.get('qubits', ()))), default=-1)
    operations, records = native_testing.load().attach_ideal_target(ideal_state._native_handle, ir.outputs,
                                                                     record_offset, first)
    return InstrumentIR(ir.inputs, ir.outputs, ir.operations+operations,
                        ir.initial, ir.controls, ir.name+'-target'), records


def attach_recovery(recovery, ir, values, record_offset, first_ancilla):
    """Append a terminal recovery with fixed decisions to an operation-level instrument."""
    operations, rules = native_testing.load().attach_terminal_recovery(_circuit_handle(recovery.circuit),
        recovery.width, ir.outputs, record_offset, first_ancilla,
        tuple((_target_data(t), bit) for t, bit in values.items()))
    controls = dict(ir.controls)
    controls.update((oid, ControlNode(parents, 'magna-v1',
        ControlRule(tuple(ControlGate(g['kind'], tuple(g['qubits']), g.get('exponent'),
            ControlExpression._from_data(guard)) for g in gates)))) for oid, parents, guard, gates in rules)
    return InstrumentIR(ir.inputs, ir.outputs, ir.operations+operations,
                        ir.initial, controls, ir.name+'-recovery')


def feedback_template(compiled):
    """The protocol feedback template, built on demand as the kernel builds its own."""
    return FeedbackTemplate(compiled.circuit, control_metadata=compiled._control_metadata)


def faults(compiled, noise, values=None, recovery_values=None, stage='original'):
    return native_testing.load().file_execution_faults(compiled._execution.handle, noise_data(noise),
        tuple((_target_data(t), bit) for t, bit in (values or {}).items()),
        tuple((_target_data(t), bit) for t, bit in (recovery_values or {}).items()), stage)


def check(compiled):
    compiled._configure_execution()
    native_testing.load().check_file_execution(compiled._execution.handle)


def file_boundary_view(boundary):
    """Operation-level view of one native file boundary for direct queries."""
    result = NativeBoundaryTrace.__new__(NativeBoundaryTrace)
    result._file_boundary_owner = boundary
    result.handle, metadata = native_testing.load().inspect_file_boundary(boundary)
    outputs, records = metadata['outputs'], metadata['records']
    ir = InstrumentIR._from_owned(metadata['inputs'],
        outputs+tuple(wire for _, wire in records), metadata['operations'],
        StabilizerKet.positive(len(metadata['inputs'])), name='magna-native-deferred')
    result.deferred = DeferredInstrument(ir, records, len(outputs), metadata['report_anchors'])
    result.locations = metadata['locations']
    result.report_anchors = {r: (oid, q) for r, oid, q in metadata['report_anchors']}
    result.queries = 0; result.kernel_seconds = 0.; result.info = {}
    result._initialize_service()
    return result


def base_engine(compiled, values, noise, *, stage, recovery_values):
    compiled._configure_execution()
    views = compiled.__dict__.setdefault('_test_engine_views', WeakValueDictionary())
    try:
        key, boundary = native_testing.load().get_file_execution_boundary(compiled._execution.handle, stage, noise_data(noise))
        engine = views.get(key)
        if engine is None:
            engine = file_boundary_view(boundary)
            views[key] = engine
        return engine
    finally:
        compiled._sync_execution()


def engine(compiled, values, noise, *, stage='protocol', recovery_values=None):
    check(compiled)
    paulis = faults(compiled, noise, values, recovery_values, stage)
    result = base_engine(compiled, values, noise, stage=stage, recovery_values=recovery_values)
    return AssignedBoundaryTrace(result, paulis) if paulis else result


def pack(compiled, engine, noise):
    return engine.pack(faults(compiled, noise), noise.report_flips)


def raw_samples(compiled, rng, count):
    """Native per-draw results from the production per-draw entry point, before formatting."""
    for _ in range(count):
        yield compiled._execution.sample(rng)


def samples(compiled, rng, count):
    """Per-draw results in the schema of CompiledCircuit.sample."""
    compiled._configure_execution()
    try:
        for result in raw_samples(compiled, rng, count):
            compiled._sync_execution()
            yield compiled._history_result(result)
    finally:
        compiled._sync_execution()


def summary(draws):
    return native_testing.load().summarize_scientific_draws(tuple(draw['scientific'] for draw in draws))


def weight_summary(draws, schedule, history_sampler='affine_support'):
    result = native_testing.load().summarize_file_weights(tuple(d['weight_scientific'] for d in draws))
    return _format_weight_summary(result, schedule, history_sampler)
