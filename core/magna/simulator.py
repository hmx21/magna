"""Public one-assignment, globally contracted simulator and bounded caches."""
from dataclasses import dataclass, field
from copy import deepcopy
from fractions import Fraction
from decimal import Decimal
import pickle
import time
from .ir import InstrumentIR, SampleAssignment, fingerprint, PreparedInstrument, validate_assignment
from .phase_tensor import QuadraticPhaseTensor
from .evaluate import target_weight
from .scalar import ExactScalar


@dataclass(frozen=True)
class Options:
    backend: str = 'database'
    max_quotient_rank: int = 8
    max_path_bits: int = 2048
    cache_bytes: int = 32*1024*1024
    cache_entries: int = 128
    max_terms: int = 256
    max_pair_contractions: int = 65536
    reuse_structure: bool = True
    max_structure_bytes: int = 32*1024*1024
    native_batch_size: int = 256

    def __post_init__(self):
        if type(self.reuse_structure) is not bool:
            raise ValueError('reuse_structure must be bool')
        if self.backend not in ('database', 'fourier'):
            raise ValueError('unknown backend')
        if type(self.native_batch_size) is not int or not 1 <= self.native_batch_size <= 65536:
            raise ValueError('native_batch_size must be in 1..65536')
        if any(type(v) is not int or v < 0 for v in (self.max_quotient_rank, self.max_path_bits,
                                                    self.cache_bytes, self.cache_entries,
                                                    self.max_terms, self.max_pair_contractions, self.max_structure_bytes)):
            raise ValueError('nonnegative integer resource budgets required')


class ByteLRU:
    """Serialized Python snapshots with native byte limits and eviction policy."""
    def __init__(self, byte_budget, max_entries):
        from . import _native
        self.budget = byte_budget; self.max_entries = max_entries
        self._handle = _native.make_certificate_cache(byte_budget, max_entries)

    def get(self, key):
        from . import _native
        blob = _native.certificate_cache_operation('get', self._handle, key)
        return None if blob is None else pickle.loads(blob)

    def put(self, key, value, cost):
        if not self.budget or not self.max_entries: return
        from . import _native
        probability = getattr(value, 'probability', None)
        extra = len(pickle.dumps(probability, protocol=5)) if probability is not None else 0
        raw = None if probability is None else (probability.coefficients, probability.power)
        _native.certificate_cache_operation('put', self._handle, key,
            pickle.dumps(value, protocol=5), raw, extra, cost)


def _polynomial_certificate(value):
    return value.to_normal() if isinstance(value, QuadraticPhaseTensor) else value


@dataclass
class CompiledSample:
    branch: object
    normal: object
    absorbed: object
    expansion: object
    resource: dict
    probability: ExactScalar
    timings: dict
    cache_key: str
    cache_hit: bool = False
    status: str = 'completed'

    def materialize(self):
        """Export the established polynomial certificates for public compile."""
        start = time.perf_counter()
        changed = False
        for name in ('normal', 'absorbed'):
            value = getattr(self, name)
            if isinstance(value, QuadraticPhaseTensor):
                setattr(self, name, _polynomial_certificate(value)); changed = True
        if changed: self.timings['certificate_seconds'] = time.perf_counter()-start
        return self


@dataclass
class SampleResult:
    assignment: SampleAssignment
    compiled: CompiledSample | None
    status: str
    diagnostics: dict = field(default_factory=dict)
    error: str | None = None
    partial: dict | None = None

    @property
    def log2_weight(self):
        if self.compiled is None: raise RuntimeError('failed sample has no weight')
        return self._weight(True)

    def _weight(self, logarithm):
        from . import _native
        n = self.assignment.noise_likelihood_ratio; q = self.assignment.proposal_probability
        return _native.sample_weight_value(self.compiled.probability._native(),
            (n.numerator,n.denominator), (q.numerator,q.denominator), logarithm)

    def weight_decimal(self):
        if self.compiled is None: raise RuntimeError('failed sample has no weight')
        return Decimal(self._weight(False))



class Simulator:
    """Exact probabilities of complete outcomes of one instrument.

    ir is an InstrumentIR, or an adapter object with an ``ir`` attribute and an
    optional ``diagnose(simulator, compiled)`` used by sample_one.
    """
    def __init__(self, ir, options=None):
        self.options = options or Options(); self.source = ir
        self.ir = ir if isinstance(ir, InstrumentIR) else ir.ir
        self.cache = ByteLRU(self.options.cache_bytes, self.options.cache_entries)
        self._prepared = None
        self.execution_info = dict(engine='native', probability_kernel='single-sided-trace', native_queries=0, batches=0, timings={})

    def _prepare(self):
        """Own one validated snapshot; all numerical compilation is native."""
        from .single_trace import SingleTracePlan
        if (self._prepared is None or not self._prepared.matches(self.ir)
                or self._prepared.preparation_options != self.options):
            prepared = PreparedInstrument(self.ir)
            prepared.preparation_options = self.options
            plan = SingleTracePlan(prepared, self.options)
            prepared.trace_plan = plan
            prepared.native_plan = plan.plan
            prepared.structure_info = dict(plan.info, enabled=plan.plan is not None)
            self._prepared = prepared
        return self._prepared

    def prepare(self):
        """Prepare reusable whole-instrument structure for the sole kernel."""
        start = time.perf_counter(); prepared = self._prepare()
        self.execution_info.update(prepared.trace_plan.info)
        return dict(prepared.structure_info, execution=deepcopy(self.execution_info),
                    total_seconds=time.perf_counter()-start)

    def _assignment_key(self, assignment, prepared):
        validate_assignment(prepared.ir, assignment, prepared)
        return fingerprint(('single-trace-certificate-v1', prepared.fingerprint,
            tuple(assignment.true[r] for r in prepared.record_names), assignment.faults,
            assignment.report_flips, self.options))

    def compile(self, assignment):
        """Export a complete output ket; its probability uses single-sided trace."""
        prepared = self._prepare()
        key = self._assignment_key(assignment, prepared) if not prepared.ir.controls else None
        if key is not None:
            cached = self.cache.get(key)
            if cached is not None:
                cached.cache_hit = True
                return cached
        tick = time.perf_counter()
        result = prepared.trace_plan.compile(assignment, self)
        if key is None:
            key = fingerprint(('single-trace-certificate-v1', prepared.fingerprint,
                               result.branch.operations, self.options))
        result.cache_key = key
        self.cache.put(key, result, time.perf_counter()-tick)
        self.execution_info.update(prepared.trace_plan.info)
        return result

    def _trace_result(self, plan, result):
        values, timings, count = result
        self.execution_info.update(plan.info)
        self.execution_info['native_queries'] += count
        self.execution_info['batches'] += bool(count)
        for key, value in timings.items():
            self.execution_info['timings'][key] = self.execution_info['timings'].get(key, 0.) + value
        return values

    def probability(self, assignment):
        """Exact full-record probability from the single-sided trace kernel."""
        plan = self._prepare().trace_plan
        return self._trace_result(plan, plan.evaluate([assignment]))[0]

    def probabilities(self, assignments):
        """Stream whole-circuit probabilities using one owned IR snapshot."""
        from .batch import CompactNoiseStream
        plan = self._prepare().trace_plan
        for result in plan.execution.batches(assignments, isinstance(assignments, CompactNoiseStream),
                                             self.options.native_batch_size):
            yield from self._trace_result(plan, result)

    def compact_noise_assignments(self, noise, rows, reported_words):
        """Owned compact original faults and report words; no quantum responses."""
        from .batch import NoiseLayout
        prepared = self._prepare()
        sites = tuple(noise.sites)
        if not hasattr(prepared, 'noise_layout') or prepared.noise_layout.sites != sites:
            prepared.noise_layout = NoiseLayout(prepared, sites)
        return prepared.noise_layout.assignments(rows, reported_words)

    def compile_instantiated(self, branch, *, instantiate_seconds=0.):
        """Compile a literal callback expansion with the same native kernel."""
        result = self._prepare().trace_plan.compile_literal(branch, self)
        result.timings['instantiate_seconds'] = instantiate_seconds
        return result

    def evaluate_observable(self, compiled, *, x=0, z=0, target=None):
        if target is not None:
            return target_weight(compiled.expansion, target, len(compiled.branch.outputs))
        if (x | z) >> len(compiled.branch.outputs) or x < 0 or z < 0:
            raise ValueError('observable may act only on declared outputs')
        from .single_trace import trace_tensor
        return trace_tensor(compiled.absorbed, self.options, x=x, z=z)

    def sample_one(self, assignment, *, capture_failures=False):
        try:
            compiled = self.compile(assignment)
            tick = time.perf_counter()
            diagnostics = self.source.diagnose(self, compiled) if hasattr(self.source, 'diagnose') else {}
            compiled.timings['diagnostic_seconds'] = time.perf_counter()-tick
            return SampleResult(assignment, compiled, compiled.status, diagnostics)
        except (RuntimeError, MemoryError, TimeoutError, NotImplementedError, ValueError, ArithmeticError) as exc:
            if not capture_failures: raise
            result = SampleResult(assignment, None, type(exc).__name__, error=str(exc))
            result.partial = getattr(exc, 'partial', None)
            return result


class IndependentProposal:
    """One draw of each real record, with exact nonuniform Bernoulli q."""
    def __init__(self, probabilities=None):
        self.probabilities = {k: Fraction(v) for k, v in (probabilities or {}).items()}
        if any(not 0 < p < 1 for p in self.probabilities.values()):
            raise ValueError('full-support probabilities strictly between zero and one required')

    def draw(self, ir, rng, *, faults=(), report_flips=(), noise_likelihood_ratio=Fraction(1), sample_id=''):
        from . import _native
        true, q = _native.draw_independent_proposal(ir.record_names,
            tuple((name, (p.numerator, p.denominator)) for name, p in self.probabilities.items()), rng)
        return SampleAssignment(true, faults, report_flips, Fraction(*q), noise_likelihood_ratio, sample_id)



class AffineProposal:
    """Uniform real-record proposal on an externally certified affine support.

    The caller must supply the support certificate for this IR/input/noise
    family. This class verifies algebra and pushforward q, not physical coverage.
    It never samples virtual quantum path variables.
    """
    def __init__(self, record_names, rows, rhs, *, coverage_certificate):
        from . import _native
        self.records = tuple(record_names)
        if not coverage_certificate:
            raise ValueError('distinct true records and a physical coverage certificate are required')
        self._handle, self.solution = _native.make_affine_proposal(self.records, tuple(rows), tuple(rhs))
        self.coverage_certificate = deepcopy(coverage_certificate)

    def draw(self, ir, rng, **kwargs):
        from . import _native
        true, q = _native.draw_affine_proposal(self._handle, ir.record_names, rng)
        return SampleAssignment(true, proposal_probability=Fraction(*q), **kwargs)


def estimate(results, score=lambda result: (1, 0)):
    """Weighted acceptance and joint diagnostic means; zero weights count.

    score returns acceptance in [0,1], conditional diagnostic in [0,1].
    Ratios are reported with delta-method standard errors, not as unbiased.
    A failed proposal invalidates the aggregate instead of removing a sample.
    """
    from . import _native
    handle = _native.make_proposal_estimate()
    failures = []
    for result in results:
        if result.compiled is None:
            failures.append(dict(sample_id=result.assignment.sample_id, status=result.status, error=result.error))
            _native.update_proposal_estimate(handle)
        else:
            weight = result.weight_decimal()
            accept, diagnostic = score(result)
            _native.update_proposal_estimate(handle, (str(weight), str(accept), str(diagnostic)))
    count, zeros, nonzero, values = _native.finish_proposal_estimate(handle)
    base = dict(proposals=count, zero_weights=zeros, failures=failures,
                status='incomplete' if failures else 'completed', ratio_unbiased=False)
    if failures: return base
    return dict(base, **{key: None if value is None else float(value) for key, value in values.items()},
                uncertainty_status='empirical standard errors only' if nonzero else
                'no nonzero weights observed; zero empirical variance does not establish zero probability')
