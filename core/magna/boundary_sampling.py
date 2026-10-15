"""One boundary-first Bell backend for direct and complex-feedback tasks.

Only postselection fixes task boundaries. Unused records are traced regardless
of whether their outcomes are deterministic. Every query compiles the complete
fixed-input instrument; no physical-time quantum state or trajectory is kept.
"""
from dataclasses import dataclass, field

from .deferred_feedback import defer_affine_feedback, _defer_affine_feedback_owned
from .feedback import ControlRule
from .ir import PreparedInstrument
from .native_compiler import serialize_compiler
from .native_records import ProbabilityMarginal
from .scalar import ExactScalar
from .stabilizer import StabilizerKet


@dataclass(frozen=True)
class BoundaryTask:
    """Reported-record parities and output effects, never assumed probabilities.

    ``postselection`` rows equal ``outcomes`` (zero by default). ``observables``
    are Hermitian physical-output Pauli (x,z) masks, in output-wire order.
    """
    postselection: tuple = ()
    outcomes: tuple | None = None
    observables: tuple = ()

    def __post_init__(self):
        for name in ('postselection', 'observables'):
            object.__setattr__(self, name, tuple(tuple(row) for row in getattr(self, name)))
        values = (0,)*len(self.postselection) if self.outcomes is None else tuple(self.outcomes)
        if len(values) != len(self.postselection) or any(type(b) is not int or b not in (0, 1) for b in values):
            raise ValueError('matching binary postselection outcomes required')
        object.__setattr__(self, 'outcomes', values)


class _NativeMarginal(ProbabilityMarginal):
    def __init__(self, raw, width, info):
        self.terms = tuple((ExactScalar(*c), StabilizerKet(*k)) for c, k in raw)
        self.width = width
        self.raw = raw; self.info = info

    def _raw(self):
        return self.raw

    def total(self, rows=(), outcomes=()):
        from . import _native
        return ExactScalar(*_native.sum_boundary_terms(self.raw, tuple(rows), tuple(outcomes)))


class NativeBoundaryTrace:
    """Owned compact native whole-instrument preparation and dynamic Bell queries."""
    def __init__(self, ir, *, max_path_bits=4096, max_doubled_path_bits=8192,
                 max_quotient_rank=8, max_terms=256, max_table_bytes=64*1024*1024):
        self._initialize(ir, defer_affine_feedback, max_path_bits=max_path_bits,
            max_doubled_path_bits=max_doubled_path_bits, max_quotient_rank=max_quotient_rank,
            max_terms=max_terms, max_table_bytes=max_table_bytes)

    @classmethod
    def _from_owned(cls, ir, **options):
        """Consume a freshly lowered private IR; caller relinquishes all leaves."""
        result = cls.__new__(cls)
        result._initialize(ir, _defer_affine_feedback_owned, **options)
        return result

    def _initialize(self, ir, defer, *, max_path_bits=4096, max_doubled_path_bits=8192,
                    max_quotient_rank=8, max_terms=256, max_table_bytes=64*1024*1024):
        self.deferred = defer(ir, expose_report_faults=True)
        # Deferral just created this owned IR; metadata is consumed locally
        # during serialization, and the native plan owns its independent copy.
        prepared = PreparedInstrument._from_owned(self.deferred.ir)
        payload, metadata = serialize_compiler(prepared, max_path_bits)
        self._finish(payload, metadata, max_doubled_path_bits, max_quotient_rank, max_terms, max_table_bytes)

    def _finish(self, payload, metadata, max_doubled_path_bits, max_quotient_rank, max_terms, max_table_bytes):
        from . import _native
        self.locations = metadata['locations']
        self.report_anchors = {r: (oid, q) for r, oid, q in self.deferred.report_anchors}
        import json
        from pathlib import Path
        from hashlib import sha256
        database = (Path(__file__).parent/'data/phase_decompositions.json').read_bytes()
        self.handle, self.info = _native.make_boundary_trace(payload, dict(
            max_quotient_rank=max_quotient_rank, max_terms=max_terms,
            max_table_bytes=max_table_bytes), max_doubled_path_bits, json.loads(database))
        self.info['database_sha256'] = sha256(database).hexdigest()
        self.queries = 0; self.kernel_seconds = 0.
        self._initialize_service()

    def _initialize_service(self):
        from . import _native
        self._service = _native.make_boundary_service(self.handle, tuple(self.locations.items()),
            tuple((name, self.deferred.original_output_count+j)
                  for j, (name, _) in enumerate(self.deferred.record_wires)), self.deferred.report_anchors)

    def pack(self, faults=(), report_flips=()):
        from . import _native
        return _native.boundary_service_pack(self._service,
            tuple((f['after'], f['qubit'], f['pauli']) for f in faults), tuple(report_flips))

    def rows(self, rows):
        from . import _native
        return _native.boundary_service_rows(self._service, tuple(tuple(row) for row in rows))

    def totals(self, packed_rows, queries):
        """Evaluate actual assignments using native bounded scalar-query layouts."""
        from . import _native
        packed_rows = tuple(packed_rows)
        rows = tuple((tuple(events), tuple(bits), x, z) for events, bits, x, z in queries)
        raw, info, seconds = _native.boundary_service_totals(self._service, packed_rows, rows)
        self.queries += len(packed_rows)*len(rows); self.kernel_seconds += seconds
        return [[ExactScalar(*v) for v in row] for row in raw], info, seconds

    def query(self, packed, *, keep_rows=(), event_rows=(), outcomes=(), x=0, z=0, affine_only=False):
        from . import _native
        raw, info = _native.query_boundary_trace(self.handle, packed, tuple(keep_rows),
            tuple(event_rows), tuple(outcomes), x, z, affine_only)
        self.queries += 1; self.kernel_seconds += info['query_seconds']
        return _NativeMarginal(raw, len(keep_rows), info)

    def prepare_queries(self, specifications, *, include_maps=False):
        """Prepare resources for (retained rows, event rows, x, z) families."""
        from . import _native
        return _native.compile_boundary_queries(self.handle, tuple(specifications), include_maps)

    def sample_rows(self, packed, rows, rng, *, event_rows=(), outcomes=()):
        """Generate commuting affine labels first, then remaining exact weights.

        If the completed query lacks common transported axes, or earlier
        conditioning spoils the affine certificate, use the bounded exact
        doubled tensor. No conjectured certificate is silently substituted.
        """
        from . import _native
        word, mass, info, queries, seconds = _native.boundary_service_sample(self._service,
            packed, tuple(rows), tuple(event_rows), tuple(outcomes), rng)
        self.queries += queries; self.kernel_seconds += seconds
        return word, ExactScalar(*mass), info



@dataclass
class BoundaryDraw:
    """Exact joint masses for one Born-drawn complex-feedback history.

    A/mass and E/mass are the per-history statistical contributions. Keep zero
    acceptance draws; no acceptance probability is supplied by the adapter.
    """
    mass: ExactScalar
    acceptance: ExactScalar
    error: ExactScalar | None
    moments: tuple
    reports: dict
    true: dict
    diagnostics: dict = field(default_factory=dict)

    @property
    def acceptance_probability(self):
        return self.acceptance.ratio(self.mass)

    @property
    def error_contribution(self):
        return None if self.error is None else self.error.ratio(self.mass)


class BoundarySampler:
    """Causal operation-interface sampling through the native boundary owner.

    Opaque controllers need a PhaseFeedback family. Their actual returned gates
    are checked on each reached history; callbacks see only declared reports.
    """
    def __init__(self, source, task=BoundaryTask(), *, max_plans=4, **limits):
        from . import _native
        from .native import ket_data
        import json
        from pathlib import Path
        self.prepared = PreparedInstrument(getattr(source, 'ir', source))
        self.ir = self.prepared.ir; self.task = task; self.limits = limits; self.max_plans = max_plans
        database = json.loads((Path(__file__).parent/'data/phase_decompositions.json').read_text())
        self._handle, metadata = _native.make_operation_boundary(self.prepared._native_layout,
            ket_data(self.ir.initial), self.ir.inputs, self.ir.outputs, self.ir.operations,
            tuple((oid, node.callback._native_gates() if type(node.callback) is ControlRule else None,
                None if node.feedback is None else (node.feedback.qubits, node.feedback.odd_axes, node.feedback.even_axes))
                for oid, node in self.ir.controls.items()),
            (task.postselection, task.outcomes, task.observables),
            dict(limits, max_plans=max_plans), database,
            {oid: node.callback for oid, node in self.ir.controls.items()})
        self.direct = metadata['direct']; self.complex = metadata['complex']; self.required = metadata['required']
        self.layout = metadata['layout']

    @property
    def plans(self):
        from . import _native
        return range(_native.operation_boundary_plan_count(self._handle))

    @property
    def route(self):
        return 'complex-feedback' if self.required else 'direct'

    def sample(self, rng, *, faults=(), report_flips=()):
        """Sample required reported parents and evaluate exact terminal masses."""
        from . import _native
        raw = _native.sample_operation_boundary(self._handle,
            tuple((f['after'], f['qubit'], f['pauli']) for f in faults), tuple(report_flips), rng)
        return BoundaryDraw(ExactScalar(*raw['mass']), ExactScalar(*raw['acceptance']),
            None if raw['error'] is None else ExactScalar(*raw['error']),
            tuple(ExactScalar(*v) for v in raw['moments']), raw['reports'], raw['true'], raw['diagnostics'])
