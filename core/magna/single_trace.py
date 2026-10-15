"""Public full-record data adapter for the native execution owner."""
from dataclasses import replace
from .ir import InstrumentIR, PreparedInstrument, SampleAssignment
from .native_global import GlobalNativePlan


class SingleTracePlan:
    def __init__(self, prepared, options):
        self.prepared = prepared; self.options = options
        self.execution = GlobalNativePlan(prepared, options)
        self.plan = self.execution if self.execution.handle is not None else None
        self.info = self.execution.info

    def compile(self, assignment, simulator):
        return self.execution.compile(assignment, simulator)

    def evaluate(self, assignments):
        return self.execution.evaluate(assignments)

    def compile_literal(self, branch, simulator):
        from . import _native
        operations, true = _native.literal_program_layout(branch.operations)
        ir = InstrumentIR(branch.inputs, branch.outputs, operations, branch.initial)
        plan = GlobalNativePlan(PreparedInstrument(ir), replace(self.options,reuse_structure=False))
        result = plan.compile(SampleAssignment(true), simulator)
        result.branch = branch
        return result


def trace_tensor(tensor, options, *, x=0, z=0):
    """Probability or Pauli moment from an existing exact amplitude certificate."""
    import json
    from pathlib import Path
    from . import _native
    from .scalar import ExactScalar
    raw = {key: getattr(tensor, key) for key in
           ('n', 'terms', 'constant', 'power', 'boundary', 'output_count', 'environment', 'zero')}
    raw['environment'] = tuple(map(str, raw['environment']))
    raw['linear'] = getattr(tensor, 'linear', (0,)*tensor.n)
    raw['upper'] = getattr(tensor, 'upper', (0,)*tensor.n)
    database = json.loads((Path(__file__).parent/'data/phase_decompositions.json').read_text()) if options.backend == 'database' else None
    limits = dict(max_quotient_rank=options.max_quotient_rank, max_terms=options.max_terms,
                  max_pair_contractions=options.max_pair_contractions)
    return ExactScalar(*_native.trace_phase_tensor(raw, limits, database, x, z, options.max_path_bits))
