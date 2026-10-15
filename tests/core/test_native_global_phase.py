"""Finite exact oracles for the native P3 quotient and boundary absorption.

These checks certify the bounded implementations, not completeness of absorption
or a minimum stabilizer rank. All amplitudes include output/environment legs.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import unittest

from magna.operations import (ExactScalar, Simulator, Options, InstrumentIR, StabilizerKet,
                                        SampleAssignment)
from magna.phase_tensor import QuadraticPhaseTensor
from magna.file_api import ResourceLimit
from tests.support import native_testing
from tests.support.operation_hooks import scalar
try:
    from magna import _native
except ImportError:
    _native = None


def dense_tensor(tensor):
    if tensor.zero:
        return {}
    result = {}
    for x in range(1 << tensor.n):
        y = sum((((a & x).bit_count() & 1) ^ b) << j
                for j, (a, b) in enumerate(tensor.boundary))
        q = sum(tensor.linear[j] + 2*(tensor.upper[j] & x).bit_count()
                for j in range(tensor.n) if x >> j & 1)
        phase = tensor.constant + 2*q + sum(c*((a & x).bit_count() & 1)
                                           for a, c in tensor.terms)
        result[y] = result.get(y, ExactScalar()) + scalar((1, 0), phase, tensor.power)
    return {y: v for y, v in result.items() if v}


def dense_states(states):
    result = {}
    for coefficient, ket in states:
        c = ExactScalar(tuple(coefficient[0]), coefficient[1])
        width, base, columns, linear, upper, power, phase = ket
        t = QuadraticPhaseTensor(len(columns), tuple(linear), tuple(upper), (), phase,
            power, tuple((sum(((col >> j) & 1) << k for k, col in enumerate(columns)),
                          (base >> j) & 1) for j in range(width)), width, ())
        for y, v in dense_tensor(t).items():
            result[y] = result.get(y, ExactScalar()) + c*v
    return {y: v for y, v in result.items() if v}


@unittest.skipUnless(_native is not None,
                     'native global compiler extension is not installed')
class NativeGlobalPhaseTests(unittest.TestCase):
    @staticmethod
    def database():
        return json.loads((Path(__file__).resolve().parents[2] /
                           'core/magna/data/phase_decompositions.json').read_text())


    def test_actual_absorption_and_decomposition_against_independent_dense_sums(self):
        rng = random.Random(26091484)
        database = self.database()
        for n in range(8):
            for trial in range(24):
                width = rng.randrange(5)
                terms = tuple(sorted((a, rng.randrange(1, 8)) for a in
                    rng.sample(range(1, 1 << n), min((1 << n)-1, 6))))
                t = QuadraticPhaseTensor(n, tuple(rng.randrange(4) for _ in range(n)),
                    tuple(rng.randrange(1 << n) & ~((1 << (j+1))-1) for j in range(n)),
                    terms, rng.randrange(8), rng.randrange(-8, 4),
                    tuple((rng.randrange(1 << n), rng.randrange(2)) for _ in range(width)),
                    max(width-1, 0), ('reset-environment',) if width else ())
                found = _native.phase_absorb(asdict(t))
                expected = dense_tensor(t)
                self.assertEqual(dense_tensor(QuadraticPhaseTensor(**found)), expected)
                for db in (None, database):
                    result, resource, states, value = _native.phase_decompose(asdict(t), {}, db)
                    self.assertEqual(dense_states(states), expected)
                    expected_norm = sum((v.conjugate()*v for v in expected.values()), ExactScalar())
                    self.assertEqual(ExactScalar(tuple(value[0]), value[1]), expected_norm)

    def test_absorption_precedes_final_resource_budget(self):
        # Summing x2 enforces x0=x1. The two odd phases then become Clifford.
        t = QuadraticPhaseTensor(3, (0, 0, 0), (4, 4, 0), ((1, 1), (2, 1)),
                                 5, -3, ((1, 0), (2, 1)), 1, ('environment',))
        limits = dict(max_quotient_rank=0, max_terms=1, max_pair_contractions=1)
        with self.assertRaises(ResourceLimit) as stop:
            _native.phase_decompose(asdict(t), limits, None, False)
        self.assertEqual((stop.exception.option, stop.exception.limit, stop.exception.needed), ('max_quotient_rank', 0, 2))
        result, resource, states, value = _native.phase_decompose(asdict(t), limits, None, True)
        self.assertEqual(resource['quotient_rank'], 0)
        self.assertEqual(resource['constructive_terms'], 1)
        self.assertEqual(result['n'], 1)
        self.assertEqual(dense_states(states), dense_tensor(t))
        self.assertIn('global_radical_contraction', [c['stage'] for c in result['certificates']])
        with self.assertRaisesRegex(ResourceLimit, 'max_terms = 0'):
            _native.phase_decompose(asdict(t), dict(max_terms=0), None)
        with self.assertRaisesRegex(ResourceLimit, 'max_table_bytes = 0'):
            _native.phase_decompose(asdict(t), dict(max_table_bytes=0), None)

    def test_rank_zero_empty_database_identity_obeys_byte_budget(self):
        database = dict(schema_version=1, entries=[])
        self.assertEqual(native_testing.load().phase_database_validate(database), 0)
        t = QuadraticPhaseTensor(0, (), (), (), 3, 0, (), 0, ())
        result, resource, states, value = _native.phase_decompose(asdict(t), {}, database)
        self.assertEqual(resource['quotient_rank'], 0)
        self.assertEqual(resource['constructive_terms'], 1)
        self.assertEqual(dense_states(states), dense_tensor(t))
        self.assertEqual(ExactScalar(tuple(value[0]), value[1]), ExactScalar((1, 0, 0, 0)))
        # Four bytes hold only the single phase-table value, not its unit term.
        for absorb_first in (False, True):
            with self.subTest(absorb_first=absorb_first):
                with self.assertRaisesRegex(ResourceLimit, 'max_table_bytes = 4'):
                    _native.phase_decompose(asdict(t), dict(max_table_bytes=4), database, absorb_first)

    def test_environment_surrogate_preserves_python_conversion_exception(self):
        t = QuadraticPhaseTensor(0, (), (), (), 0, 0, ((0, 0),), 0, ('\ud800',))
        for entry, args in ((_native.phase_absorb, (asdict(t),)),
                            (_native.phase_decompose, (asdict(t), {}, None))):
            with self.subTest(entry=entry.__name__):
                with self.assertRaises(UnicodeEncodeError):
                    entry(*args)

    def test_environment_unicode_and_embedded_nul_roundtrip(self):
        labels = ('\0', 'before\0after', '雪-é-\0-🙂')
        t = QuadraticPhaseTensor(0, (), (), (), 0, 0,
                                 ((0, 0),)*len(labels), 0, labels)
        absorbed = _native.phase_absorb(asdict(t))
        decomposed, _, _, _ = _native.phase_decompose(asdict(t), {}, None)
        self.assertEqual(absorbed['environment'], labels)
        self.assertEqual(decomposed['environment'], labels)

    def test_database_exact_reconstruction_and_product_identification(self):
        database = self.database()
        self.assertEqual(native_testing.load().phase_database_validate(database), 15)
        broken = deepcopy(database)
        broken['entries'][0]['terms'][0]['coefficient'][0] += 1
        with self.assertRaisesRegex(ValueError, 'reconstruction'):
            native_testing.load().phase_database_validate(broken)
        duplicate = deepcopy(database)
        duplicate['entries'].append(duplicate['entries'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            native_testing.load().phase_database_validate(duplicate)
        for n in range(1, 7):
            t = QuadraticPhaseTensor(n, (0,)*n, (0,)*n,
                tuple((1 << j, 1) for j in range(n)), 3, -n,
                tuple((1 << j, 0) for j in range(n)), n, ())
            result, resource, states, value = _native.phase_decompose(asdict(t), {}, database)
            self.assertTrue(resource['database_hit'])
            self.assertEqual(dense_states(states), dense_tensor(t))

    @unittest.skipUnless(_native is not None and hasattr(_native, 'make_full_record'),
                         'native global integration is not installed')
    def test_original_ir_exceeds_fixed_budget_and_succeeds_after_native_absorption(self):
        # Retained cancelling ancilla phase legs keep this constraint out of the
        # prepared Clifford skeleton. Attaching their actual phases permits it.
        operations = (
            dict(id='t0', kind='T', qubits=[0]),
            dict(id='t1', kind='T', qubits=[1]),
            dict(id='t2', kind='T', qubits=[2]),
            dict(id='tdg2', kind='T', qubits=[2], exponent=-1),
            dict(id='cz0', kind='CZ', qubits=[0, 2]),
            dict(id='cz1', kind='CZ', qubits=[1, 2]),
            dict(id='m', kind='MX_DESTRUCT', qubits=[2], record='m'))
        ir = InstrumentIR((0, 1, 2), (0, 1), operations,
                          StabilizerKet.positive(3, (1, 2, 4)))
        assignments = [SampleAssignment({'m': bit}) for bit in (0, 1)]
        expected = [ExactScalar((1, 0, 0, 0), -1)]*2
        native = Simulator(ir, Options(cache_entries=0, max_quotient_rank=0, max_terms=1, max_pair_contractions=1))
        self.assertEqual(list(native.probabilities(assignments)), expected)
        self.assertEqual(native.execution_info['native_queries'], 2)
        for assignment in assignments:
            compiled = native.compile(assignment)
            self.assertEqual(compiled.probability, expected[0])
            self.assertEqual(compiled.resource['quotient_rank'], 0)
            self.assertEqual(compiled.resource['constructive_terms'], 1)
        # An actual ancilla fault changes the Clifford radical sum and
        # restores rank two. The rank-zero budget must reject that sample.
        expensive = SampleAssignment({'m': 0},
            (dict(after='t2', qubit=2, pauli='X'),))
        self.assertEqual(native.probability(expensive), expected[0])
        with self.assertRaises(ResourceLimit):
            native.compile(expensive)

    def test_malformed_dimensions_and_zero_phase_conventions(self):
        with self.assertRaises(ValueError):
            native_testing.load().phase_polynomial(2, ((4, 1),))
        t = dict(n=1, linear=(0,), upper=(1,), terms=(), constant=0,
                 power=0, boundary=((1, 0),), output_count=1, environment=())
        with self.assertRaisesRegex(ValueError, 'quadratic'):
            _native.phase_absorb(t)


if __name__ == '__main__':
    unittest.main()
