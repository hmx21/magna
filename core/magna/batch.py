"""Immutable, snapshot-bound input for probability queries on original faults.

Codes use IXZY two-bit little-endian Pauli coordinates per physical wire;
a single-channel site uses code 1. This packs assignments, never quantum
answers. Records are true outcomes; reported words are XORed with readout
errors before projectors are evaluated. Repeated report errors cancel.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from . import _native


@dataclass(frozen=True, slots=True)
class Fault(Mapping):
    after: str
    qubit: int
    pauli: str

    def __getitem__(self, key):
        if key not in ('after', 'qubit', 'pauli'): raise KeyError(key)
        return getattr(self, key)
    def __iter__(self): return iter(('after', 'qubit', 'pauli'))
    def __len__(self): return 3
    def __deepcopy__(self, memo): return dict(self)


@dataclass(frozen=True, slots=True)
class RecordBits(Mapping):
    names: tuple
    index: object
    word: int

    def __getitem__(self, key): return (self.word >> self.index[key]) & 1
    def __iter__(self): return iter(self.names)
    def __len__(self): return len(self.names)
    def __deepcopy__(self, memo): return dict(self)


@dataclass(frozen=True, slots=True, init=False)
class CompactAssignment:
    fingerprint: str
    true: RecordBits
    faults: tuple
    report_word: int
    proposal_probability: Fraction = Fraction(1)
    noise_likelihood_ratio: Fraction = Fraction(1)
    sample_id: str = ''

    def __init__(self, prepared, true_word, faults=(), report_word=0):
        values = tuple((f['after'], f['qubit'], f['pauli']) for f in faults)
        _native.check_compact_assignment(prepared._native_layout, true_word, report_word, values)
        self._set(prepared, true_word, tuple(Fault(*value) for value in values), report_word)

    def _set(self, prepared, word, faults, report_word):
        object.__setattr__(self, 'fingerprint', prepared.fingerprint)
        object.__setattr__(self, 'true', RecordBits(prepared.record_names, prepared.record_index, word))
        object.__setattr__(self, 'faults', faults)
        object.__setattr__(self, 'report_word', report_word)
        object.__setattr__(self, 'proposal_probability', Fraction(1))
        object.__setattr__(self, 'noise_likelihood_ratio', Fraction(1))
        object.__setattr__(self, 'sample_id', '')

    @property
    def true_word(self): return self.true.word

    @property
    def report_flips(self):
        return _native.record_names_from_word(self.true.names, self.report_word)


class NoiseLayout:
    def __init__(self, prepared, sites):
        self.prepared = prepared
        self.sites = sites
        self._native_handle = _native.make_noise_layout(prepared._native_layout,
            tuple(site._native_handle for site in sites))

    def assignments(self, rows, reported_words):
        words = tuple(reported_words)
        _native.noise_layout_words(self._native_handle, words)
        def iterate():
            for row in rows:
                raw_faults, flips, true_words = _native.decode_noise_layout(self._native_handle, tuple(row), words)
                faults = tuple(Fault(*value) for value in raw_faults)
                for word in true_words:
                    result = object.__new__(CompactAssignment)
                    result._set(self.prepared, word, faults, flips)
                    yield result
        return CompactNoiseStream(iterate())


class CompactNoiseStream:
    """Explicitly permits bounded prefetch of immutable packed assignments.

    Public SampleAssignment iterables retain one-at-a-time consumption. This
    separate compact API permits reading up to native_batch_size original rows
    ahead, with each row copied when it enters the block.
    """
    def __init__(self, iterator): self._iterator = iterator
    def __iter__(self): return self
    def __next__(self): return next(self._iterator)
