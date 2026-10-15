"""Independent Python history scheduler retained only as a migration oracle."""
from time import monotonic
from magna.scalar import ONE, ExactScalar
from magna.file_api import _ordinary
from tests.support import file_hooks as hooks

from collections.abc import Mapping
from tests.support import native_testing
from tests.support.global_sampling import real_sign

class ControlValues(Mapping):
    def __init__(self, metadata):
        self.metadata = metadata
        self.bits = [None]*len(metadata.index)

    def __getitem__(self, term):
        value = self.bits[self.metadata.index[term]]
        if value is None: raise KeyError(term)
        return value

    def __iter__(self):
        return (term for term, index in self.metadata.index.items() if self.bits[index] is not None)

    def __len__(self):
        return sum(bit is not None for bit in self.bits)

    def parity(self, indices, *, required=False):
        value = 0
        for index in indices:
            bit = self.bits[index]
            if bit is None:
                if required: raise ValueError('unresolved causal controller input')
            else: value ^= bit
        return value


class ReferenceMetadata:
    def __init__(self, source):
        self.index = source.index
        self.callbacks = source.callbacks

    def values(self):
        return ControlValues(self)


class ReferenceHistory:
    def __init__(self, compiled, engine):
        self.compiled = compiled
        self.target_normalization = ExactScalar(*compiled._execution.state(False)['target_normalization'])
        self._control_metadata = ReferenceMetadata(compiled._control_metadata)
        self._recovery_control_metadata = None if compiled._recovery_control_metadata is None else ReferenceMetadata(compiled._recovery_control_metadata)
        self._engine = engine
        self.queries = 0
        self.kernel_seconds = self.decoder_seconds = 0.
        self.last_resources = []

    def __getattr__(self, name):
        return getattr(self.compiled, name)

    def _query(self, engine, packed, **kwargs):
        hooks.check(self.compiled); self.queries += 1
        tick = monotonic()
        marginal = engine.query(packed, **kwargs)
        self.kernel_seconds += monotonic()-tick; self.last_resources.append(marginal.info)
        return marginal

    def sample(self, rng, *, noise=None, _branch_selector=None):
        noise = hooks.draw(self.noise_schedule, rng) if noise is None else noise
        known_rows = []; known_bits = []; known_basis = {}; values = self._control_metadata.values(); decoder_calls = 0
        recovery_values = ({} if self._recovery_control_metadata is None else self._recovery_control_metadata.values())
        recovery_decoder_calls = 0
        self.last_resources = []
        def reduce(mask, bit=0):
            while mask:
                pivot = mask.bit_length()-1
                if pivot not in known_basis: break
                row, value = known_basis[pivot]; mask ^= row; bit ^= value
            return mask, bit
        def sample_inputs(expressions, *, stage='protocol', record_offset=0, masks=False):
            # Canonicalize already known linear information before asking for a
            # native marginal. Repeated decoder calls do not grow duplicate legs.
            requests = []; pending = []; indexes = {}
            for names, offset in expressions:
                mask = names << record_offset if masks else 0
                if not masks:
                    for r in names: mask ^= 1 << (r+record_offset)
                residual, known = reduce(mask)
                if residual:
                    if residual not in indexes:
                        indexes[residual] = len(pending); pending.append(residual)
                    requests.append((indexes[residual], offset ^ known))
                else: requests.append((None, offset ^ known))
            if not pending: return tuple(bit for index, bit in requests)
            rows = tuple(tuple('m'+str(r) for r in range(mask.bit_length()) if mask >> r & 1) for mask in pending)
            engine = self._engine(values, noise, stage=stage, recovery_values=recovery_values)
            packed = hooks.pack(self.compiled, engine, noise)
            marginal = self._query(engine, packed, keep_rows=engine.rows(rows),
                event_rows=engine.rows(known_rows), outcomes=tuple(known_bits))
            from magna import _native
            word = (native_testing.load().sample_boundary_terms(marginal.raw, len(rows), rng.getrandbits(64))
                    if _branch_selector is None else _branch_selector(marginal.raw, len(rows)))
            for j, (mask, row) in enumerate(zip(pending, rows)):
                bit = word >> j & 1; residual, value = reduce(mask, bit)
                if residual:
                    known_basis[residual.bit_length()-1] = (residual, value)
                    known_rows.append(row); known_bits.append(bit)
                elif value: raise ArithmeticError('sampled parities violate an existing linear relation')
            return tuple(bit if index is None else bit ^ (word >> index & 1) for index, bit in requests)
        def invoke(metadata, state, *, stage='protocol', record_offset=0):
            nonlocal decoder_calls, recovery_decoder_calls
            for inst, binding, inputs, outputs in metadata.callbacks:
                expressions = [(mask, constant ^ state.parity(indices, required=True))
                               for mask, constant, indices in inputs]
                bits = sample_inputs(expressions, stage=stage, record_offset=record_offset, masks=True)
                tick = monotonic()
                out = binding.invoke(bits, len(outputs), source=inst.source)
                self.decoder_seconds += monotonic()-tick
                if stage == 'protocol': decoder_calls += 1
                else: recovery_decoder_calls += 1
                for index, bit in zip(outputs, out): state.bits[index] = bit
        invoke(self._control_metadata, values)
        samples = sample_inputs([((r,), 0) for r in self.task['sample_records']]) if self.task['sample_records'] else ()
        # Terminal controllers have their own namespace and channel. They read
        # only terminal reports; target projection and protocol postselection
        # never condition these Born draws. All quantum queries stay global.
        if self.recovery is not None:
            invoke(self._recovery_control_metadata, recovery_values,
                   stage='recovery', record_offset=self.circuit.num_records)
        mass, acceptance, error, moments, target_success = self._evaluate_rows(
            [noise], values, recovery_values, known_rows, tuple(known_bits))[0]
        return self._result(noise, mass, acceptance, error, moments, target_success,
                            samples, decoder_calls, recovery_decoder_calls, len(known_rows), _branch_selector)

    def _scalar_queries(self, engine, known_rows, outcomes):
        known = engine.rows(known_rows)
        events = known+engine.rows(self.task['postselection']); rhs = outcomes+self.task['outcomes']
        queries = [(known, outcomes, 0, 0), (events, rhs, 0, 0)]
        queries.extend((events, rhs, x, z) for x, z in self.task['observables'])
        return tuple(queries), events, rhs

    def _totals(self, engine, packed_rows, queries):
        hooks.check(self.compiled)
        # Preserve checks between individual global queries when the caller
        # requests a time or query budget. The same kernel remains the oracle
        # for unprepared queries and dynamic retained-record marginals.
        if 'max_seconds' in self.options or 'max_queries' in self.options or len(queries) > 4096:
            return [[self._query(engine, packed, event_rows=events, outcomes=bits, x=x, z=z).total()
                     for events, bits, x, z in queries] for packed in packed_rows]
        rows, info, seconds = engine.totals(packed_rows, queries)
        self.queries += len(packed_rows)*len(queries); self.kernel_seconds += seconds
        self.last_resources.extend(info)
        return rows

    def _evaluate_rows(self, noises, values, recovery_values, known_rows, outcomes):
        engine = self._engine(values, noises[0], stage='recovery', recovery_values=recovery_values)
        queries, events, rhs = self._scalar_queries(engine, known_rows, outcomes)
        totals = self._totals(engine, [hooks.pack(self.compiled, engine, noise) for noise in noises], queries)
        target_values = [None]*len(noises)
        if self.ideal_state is not None:
            target_engine = self._engine(values, noises[0], stage='target', recovery_values=recovery_values)
            target_events = target_engine.rows(tuple(known_rows)+self.task['postselection']+
                                               tuple((r,) for r in self.target_records))
            target_values = [row[0] for row in self._totals(target_engine,
                [hooks.pack(self.compiled, target_engine, noise) for noise in noises],
                ((target_events, rhs+(0,)*len(self.target_records), 0, 0),))]
            traced = self.ideal_state.preparation_width-self.ideal_state.width-len(self.ideal_state.postselection)
            target_values = [value.scaled(1 << traced) for value in target_values]
        result = []
        for row, target_success in zip(totals, target_values):
            mass, acceptance = row[:2]; position = 2; error = None
            if real_sign(mass) <= 0: raise ArithmeticError('nonpositive sampled history mass')
            if real_sign(acceptance) < 0 or real_sign(mass-acceptance) < 0: raise ArithmeticError('invalid accepted mass')
            if target_success is not None: error = acceptance*self.target_normalization-target_success
            if error is not None and (real_sign(error) < 0 or real_sign(acceptance*self.target_normalization-error) < 0):
                raise ValueError('error effect mass outside accepted probability')
            result.append((mass, acceptance, error, row[position:], target_success))
        return result

    def _result(self, noise, mass, acceptance, error, moments, target_success,
                samples=(), decoder_calls=0, recovery_decoder_calls=0, sampled_parities=0, _branch_selector=None):
        from magna import _native
        if _branch_selector is not None:
            return tuple(None if v is None else (v.coefficients, v.power)
                         for v in [mass, acceptance, error]+moments)
        exact_values = [acceptance, error] + moments
        denominators = [mass, mass*self.target_normalization]+[mass]*len(moments)
        pairs = tuple(((value.coefficients,value.power),(denominator.coefficients,denominator.power))
                      for value, denominator in zip(exact_values, denominators) if value is not None)
        weighted = native_testing.load().file_scalar_ratios(pairs+(((ONE.coefficients,ONE.power),(ONE.coefficients,ONE.power)),),
                                               noise.likelihood_ratio)
        converted = iter(weighted[:-1])
        scientific = [None if value is None else next(converted) for value in exact_values]
        ordinary = [_ordinary(value) for value in scientific]
        underflow = [j for j,(encoded,value) in enumerate(zip(scientific,ordinary))
                     if encoded is not None and encoded[0] != 0 and value == 0]
        overflow = [j for j,(encoded,value) in enumerate(zip(scientific,ordinary))
                    if encoded is not None and value is None]
        for j in underflow: ordinary[j] = None
        result = dict(acceptance=ordinary[0], error=ordinary[1], moments=ordinary[2:],
                    scientific=scientific, underflow_columns=underflow, overflow_columns=overflow,
                    weight_scientific=weighted[-1], sampled_records=list(samples),
                    decoder_calls=decoder_calls, recovery_decoder_calls=recovery_decoder_calls,
                    sampled_parities=sampled_parities,
                    exact=dict(history_mass=mass.to_dict(), acceptance=acceptance.to_dict(),
                               error=None if error is None else error.to_dict(), moments=[m.to_dict() for m in moments],
                               target_success=None if target_success is None else target_success.to_dict(),
                               noise_likelihood_ratio_hex=[hex(v) for v in noise.likelihood_ratio]))
        if self.target_normalization != ONE:
            result['exact']['target_normalization'] = self.target_normalization.to_dict()
        return result
