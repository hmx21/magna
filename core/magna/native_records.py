"""Adapters for native completed record marginals and classical sampling."""
from . import _native
from .native import term_data, term_values
from .scalar import ExactScalar


class ProbabilityMarginal:
    def __init__(self, terms, width):
        self.terms = tuple(terms)
        self.width = width

    def _raw(self):
        return term_data(self.terms)

    def total(self, rows=(), outcomes=()):
        return ExactScalar(*_native.marginal_operation('total', self._raw(), self.width, rows, outcomes))

    def value(self, word):
        return ExactScalar(*_native.marginal_operation('value', self._raw(), self.width, word))

    def sample(self, rng):
        return _native.marginal_operation('sample', self._raw(), self.width, rng)


class NativeRecordMarginals:
    def __init__(self, rows, *, event_rows=(), outcomes=(), max_plans=12):
        self.rows = tuple(rows)
        self.events = tuple(event_rows)
        self.outcomes = tuple(outcomes)
        self.max_plans = max_plans
        self._handle = _native.record_marginals_operation('make', self.rows, self.events, self.outcomes, max_plans)

    def evaluate(self, expansion, *, x=0, z=0):
        raw = _native.record_marginals_operation('evaluate', self._handle, term_data(expansion.terms), x, z)
        return ProbabilityMarginal(term_values(raw), len(self.rows))
