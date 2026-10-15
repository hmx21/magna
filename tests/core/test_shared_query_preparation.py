"""Shared symbolic maps against fresh queries and complete fixed-record sums."""
import itertools
import unittest

from magna.operations import (InstrumentIR, NativeBoundaryTrace, Options, SampleAssignment,
                                        Simulator, StabilizerKet)
from magna.scalar import ExactScalar


class SharedQueryPreparationTests(unittest.TestCase):
    def test_effect_offsets_events_and_faults_match_independent_branch_sums(self):
        ir = InstrumentIR((0, 1, 2), (0, 1), (
            dict(id='t', kind='T', qubits=(0,)),
            dict(id='s', kind='S', qubits=(1,)),
            dict(id='ccz', kind='CCZ', qubits=(0, 1, 2)),
            dict(id='a', kind='MX', qubits=(2,), record='a'),
            dict(id='h', kind='H', qubits=(1,)),
            dict(id='u', kind='T', qubits=(1,)),
            dict(id='b', kind='MZ', qubits=(2,), record='b'),
            dict(id='drop', kind='DISCARD', qubits=(2,))),
            StabilizerKet.positive(3, (1, 2, 4)))
        engine = NativeBoundaryTrace(ir)
        a, b = engine.rows((('a',), ('b',)))
        specifications = [(keep, events, x, z)
            for keep, events in (((), (a,)), ((b,), (a, a)), ((), (b,)))
            for x in range(4) for z in range(4)]
        reports = engine.prepare_queries(specifications, include_maps=True)
        self.assertTrue(any(q['shared_symbolic_maps'] for q in reports))
        reference = Simulator(ir, Options(reuse_structure=False, cache_entries=0))
        for faults, flips in (((), ()), ((dict(after='ccz', qubit=0, pauli='Y'),), ('a',))):
            branches = []
            for bits in itertools.product((0, 1), repeat=2):
                compiled = reference.compile(SampleAssignment(dict(zip(('a', 'b'), bits)), faults, flips))
                reports = dict(zip(('a', 'b'), (bits[0] ^ bool(flips), bits[1])))
                branches.append((reports, compiled))
            packed = engine.pack(faults, flips)
            for keep, events, x, z in specifications:
                for outcomes in itertools.product((0, 1), repeat=len(events)):
                    actual = engine.query(packed, keep_rows=keep, event_rows=events,
                                          outcomes=outcomes, x=x, z=z)
                    for retained in range(1 << len(keep)):
                        expected = ExactScalar()
                        for values, compiled in branches:
                            if any(values['a' if row == a else 'b'] != bit
                                   for row, bit in zip(events, outcomes)):
                                continue
                            if keep and values['b'] != retained:
                                continue
                            expected += reference.evaluate_observable(compiled, x=x, z=z)
                        found = actual.total((1,), (retained,)) if keep else actual.total()
                        self.assertEqual(found, expected, (keep, events, x, z, outcomes, retained))
                    self.assertEqual(actual.info['pairing_seconds'], 0)
        # A new owner has no preparation sharing; both orders retain identical
        # resources and scalar values, including zero Clifford pairings.
        for spec, report in zip(specifications, engine.prepare_queries(specifications)):
            fresh = NativeBoundaryTrace(ir)
            isolated, = fresh.prepare_queries((spec,))
            self.assertFalse(isolated['shared_symbolic_maps'])
            for key in ('reference_phase_coordinates', 'reference_terms', 'reference_zero'):
                self.assertEqual(report[key], isolated[key])
            keep, events, x, z = spec
            self.assertEqual(engine.query(engine.pack(), keep_rows=keep, event_rows=events,
                outcomes=(0,)*len(events), x=x, z=z).total(),
                fresh.query(fresh.pack(), keep_rows=keep, event_rows=events,
                outcomes=(0,)*len(events), x=x, z=z).total())


if __name__ == '__main__':
    unittest.main()
