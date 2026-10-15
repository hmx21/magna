"""Seeded comparison with the former independent operation boundary schedule."""
from random import Random
import unittest
from magna.operations import (InstrumentIR, NativeBoundaryTrace, StabilizerKet, BoundarySampler,
                                        BoundaryTask, ControlNode, PhaseFeedback)
from magna import _native
from tests.support.boundary_sampler import ReferenceBoundarySampler
from tests.support import native_testing


def reference_sample(engine, packed, rows, rng):
    sector, _ = native_testing.load().boundary_commuting_rows(engine.handle, rows)
    events = (); bits = ()
    if sector:
        selected = tuple(__import__('functools').reduce(int.__xor__,
            (row for j, row in enumerate(rows) if mask >> j & 1), 0) for mask in sector)
        marginal = engine.query(packed, keep_rows=selected, affine_only=True)
        if marginal.info['affine']:
            word, _ = marginal.sample(rng)
            events += selected; bits += tuple(word >> j & 1 for j in range(len(selected)))
    marginal = engine.query(packed, keep_rows=rows, event_rows=events, outcomes=bits)
    word, method = marginal.sample(rng)
    return word, marginal.total(tuple(1 << j for j in range(len(rows))),
        tuple(word >> j & 1 for j in range(len(rows)))), method


class NativeBoundaryServiceTests(unittest.TestCase):
    def test_boundary_sum_validates_rows_for_every_term_width(self):
        from magna.native import term_data
        from magna.scalar import ONE, ExactScalar
        terms = term_data(((ONE, StabilizerKet.positive(2)),
                           (ONE, StabilizerKet.positive(1))))
        self.assertEqual(ExactScalar(*_native.sum_boundary_terms(terms, (1, 1), (0, 0))), ONE+ONE)
        self.assertEqual(ExactScalar(*_native.sum_boundary_terms(terms, (1, 1), (0, 1))), ExactScalar())
        # The first term fits this row, but the second does not.
        with self.assertRaisesRegex(ValueError, 'outside tensor'):
            _native.sum_boundary_terms(terms, (2,), (0,))
        for rows, outcomes in (((1,), ()), ((1,), (2,))):
            with self.assertRaises(ValueError):
                _native.sum_boundary_terms(terms, rows, outcomes)

    def test_causal_controller_order_rng_and_terminal_masses(self):
        def fixture(seen):
            def callback(reports):
                seen.append(dict(reports))
                return (dict(kind='S', qubits=[1], exponent=2*reports['a']),)
            return InstrumentIR((0, 1), (1,), (
                dict(id='t', kind='T', qubits=[0]), dict(id='a', kind='MX', qubits=[0], record='a'),
                dict(id='c', kind='CONTROL'), dict(id='b', kind='MX', qubits=[1], record='b'),
                dict(id='drop', kind='DISCARD', qubits=[0])), StabilizerKet.positive(2, (1, 2)),
                {'c': ControlNode(('a',), 'test', callback, PhaseFeedback((1,)))})
        task = BoundaryTask((('b',),), (1,), ((0, 1), (1, 0)))
        seen = []; reference_seen = []
        native = BoundarySampler(fixture(seen), task, max_plans=1)
        reference = ReferenceBoundarySampler(fixture(reference_seen), task, max_plans=1)
        for seed in range(12):
            rng = Random(seed); reference_rng = Random(seed)
            kwargs = dict(faults=(dict(after='c', qubit=1, pauli='Y'),), report_flips=('a',))
            draw = native.sample(rng, **kwargs); expected = reference.sample(reference_rng, **kwargs)
            self.assertEqual((draw.mass, draw.acceptance, draw.error, draw.moments, draw.reports, draw.true),
                (expected.mass, expected.acceptance, expected.error, expected.moments, expected.reports, expected.true))
            self.assertEqual(seen, reference_seen)
            self.assertEqual(rng.getstate(), reference_rng.getstate())
            self.assertLessEqual(len(native.plans), 1)

    def test_completed_marginal_rng_and_mass_match_reference_schedule(self):
        for extra in ((), (dict(id='h', kind='H', qubits=[0]), dict(id='u', kind='T', qubits=[0]))):
            ir = InstrumentIR((0, 1), (), (dict(id='t', kind='PHASE', qubits=[0, 1], exponent=1),
                *extra, dict(id='a', kind='MX_DESTRUCT', qubits=[0], record='a'),
                dict(id='b', kind='MX_DESTRUCT', qubits=[1], record='b')), StabilizerKet.positive(2, (1, 2)))
            engine = NativeBoundaryTrace(ir); rows = engine.rows((('a',), ('b',)))
            for seed in range(8):
                actual_rng = Random(seed); reference_rng = Random(seed)
                packed = engine.pack((dict(after='t', qubit=0, pauli='Y'),), ('a',))
                expected = reference_sample(engine, packed, rows, reference_rng)
                word, mass, info = engine.sample_rows(packed, rows, actual_rng)
                self.assertEqual((word, mass, info['method']), expected)
                self.assertEqual(actual_rng.getstate(), reference_rng.getstate())


if __name__ == '__main__': unittest.main()
