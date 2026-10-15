"""A common ket phase cancels in traces; paired coefficients do not."""
from dataclasses import replace
import unittest

from magna.operations import InstrumentIR, NativeBoundaryTrace, StabilizerKet
from magna import _native
from magna.scalar import ExactScalar, ONE
from test_native_core import dense
from tests.support import native_testing


class TraceGlobalPhaseTests(unittest.TestCase):
    def test_common_phase_with_faults_events_reset_and_retained_labels(self):
        ops = (
            dict(id='h', kind='H', qubits=(0,)),
            dict(id='t', kind='T', qubits=(0,)),
            dict(id='cx', kind='CX', qubits=(0, 1)),
            dict(id='y', kind='Y', qubits=(1,)),
            dict(id='m', kind='MX', qubits=(1,), record='a'),
            dict(id='reset', kind='RESET_PLUS', qubits=(1,)),
            dict(id='cz', kind='CZ', qubits=(0, 1)))
        initial = StabilizerKet(2, 0, (), (), (), 0, 0)
        reference = None
        for phase in range(8):
            engine = NativeBoundaryTrace(InstrumentIR((0, 1), (0, 1), ops, replace(initial, phase=phase)))
            packed = engine.pack((dict(after='cx', qubit=0, pauli='Y'),), ('a',))
            row, = engine.rows((('a',),))
            results = []
            for x, z in ((0, 0), (1, 0), (0, 1), (1, 1)):
                results.append(engine.query(packed, x=x, z=z).total())
                for outcome in (0, 1):
                    results.append(engine.query(packed, event_rows=(row,), outcomes=(outcome,), x=x, z=z).total())
                marginal = engine.query(packed, keep_rows=(row,), x=x, z=z)
                results.extend(marginal.total((1,), (outcome,)) for outcome in (0, 1))
            if reference is None:
                reference = results
            self.assertEqual(results, reference)
            self.assertEqual(results[0], ONE)

    def test_paired_sign_and_subnormalization_survive(self):
        raw = dict(n=0, linear=(), upper=(), terms=(), constant=0, power=0,
                   boundary=((0, 1),), output_count=1, environment=(), zero=False)
        for phase in range(8):
            negative = _native.trace_phase_tensor(dict(raw, constant=phase), None, None, 0, 1, 64)
            self.assertEqual(ExactScalar(tuple(negative[0]), negative[1]), ExactScalar((-1, 0, 0, 0)))
            paired = native_testing.load().trace_probability_tensor(dict(raw, constant=phase), (), (), (), 0, 1, 128)
            self.assertEqual(paired['constant'], 4)
            scaled = _native.trace_phase_tensor(dict(raw, constant=phase, power=-2), None, None, 0, 0, 64)
            self.assertEqual(ExactScalar(tuple(scaled[0]), scaled[1]), ExactScalar((1, 0, 0, 0), -2))

    def test_amplitude_compiler_preserves_the_requested_phase(self):
        # This interface returns a ket, not its paired trace. Its phase remains
        # observable to callers and the default exact-amplitude path is intact.
        for phase in range(8):
            raw = dict(initial=(1, 0, (), (), (), 0, phase), inputs=(0,), outputs=(0,), records=0,
                       max_path_bits=32, operations=(('Y', (0,), 0, -1, 0, 0, -1, '', ''),))
            found = native_testing.load().debug_prepare_compiler(raw)['skeleton']
            self.assertEqual(dense(found), dense((1, 1, (), (), (), 0, (phase + 2) % 8)))


if __name__ == '__main__':
    unittest.main()
