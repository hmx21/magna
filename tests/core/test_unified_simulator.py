"""Independent physical matrices, explicit path sums, and common-core checks.

Exponential enumeration is deliberately confined to this tiny test oracle.
"""
import itertools
import random
import unittest
from fractions import Fraction
import numpy as np
from magna.operations import *
from magna.scalar import ONE
from magna.file_api import ResourceLimit
from tests.support.operation_hooks import cache_info


def ket_vector(ket):
    v = np.zeros(1 << ket.width, complex)
    for u in range(1 << len(ket.columns)):
        x = ket.base; q = 0
        for j, c in enumerate(ket.columns):
            if u >> j & 1:
                x ^= c; q += ket.linear[j]
                q += 2*sum(1 for k in range(j+1, len(ket.columns)) if u >> k & 1 and ket.upper[j] >> k & 1)
        v[x] += 2.**(ket.power/2)*np.exp(1j*np.pi*(ket.phase+2*q)/4)
    return v


def expansion_vector(e, width):
    return sum((c.complex()*ket_vector(t) for c, t in e.terms), np.zeros(1 << width, complex))


def path_vector(m):
    v = np.zeros(1 << len(m.boundary), complex)
    if m.zero: return v
    for u in range(1 << m.n):
        x = sum((((a & u).bit_count() % 2) ^ b) << j for j, (a, b) in enumerate(m.boundary))
        e = m.constant+sum(c*((a & u).bit_count() % 2) for a, c in m.terms)
        v[x] += 2.**(m.power/2)*np.exp(1j*np.pi*e/4)
    return v


def physical_vector(branch):
    """Literal fixed-branch matrices, independently retaining reset environments."""
    live = list(branch.inputs); env = []; v = ket_vector(branch.initial)
    H = np.array([[1, 1], [1, -1]], complex)/np.sqrt(2)
    P = {'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1]), 'I': np.eye(2)}
    def mat(g, qs):
        nonlocal v
        inds = [live.index(q) for q in qs]; out = np.zeros_like(v)
        for x, amplitude in enumerate(v):
            col = sum(((x >> q) & 1) << j for j, q in enumerate(inds))
            base = x & ~sum(1 << q for q in inds)
            for row in range(1 << len(inds)):
                y = base | sum(((row >> j) & 1) << q for j, q in enumerate(inds))
                out[y] += g[row, col]*amplitude
        v = out
    for op in branch.operations:
        k = op['kind']; qs = op.get('qubits', [])
        if k.startswith('RESET') or k == 'DISCARD':
            old = ('env', len(env)); live[live.index(qs[0])] = old; env.append(old)
        if k.startswith('PREP') or k.startswith('RESET'):
            plus = k.endswith('PLUS'); v = np.kron(np.array([1, 1] if plus else [1, 0])/np.sqrt(2 if plus else 1), v)
            live.append(qs[0])
        elif k == 'DISCARD': pass
        elif k == 'H': mat(H, qs)
        elif k in ('T', 'S', 'PHASE'):
            exp = op.get('actual_exponent', op.get('exponent', 2 if k == 'S' else 1))
            mat(np.diag([np.exp(1j*np.pi*exp*(x.bit_count() % 2)/4) for x in range(1 << len(qs))]), qs)
        elif k in ('X', 'Y', 'Z', 'PAULI'): mat(P[op.get('pauli', k)], qs)
        elif k == 'CX':
            m = np.zeros((4, 4));
            for x in range(4): m[x ^ (2 if x & 1 else 0), x] = 1
            mat(m, qs)
        elif k in ('CZ', 'CCZ'):
            m = np.ones(1 << len(qs)); m[-1] = -1; mat(np.diag(m), qs)
        elif k in ('MZ', 'MX', 'MXX', 'MP'):
            bodies = op.get('paulis', ('Z' if k == 'MZ' else 'X')*len(qs))
            obs = np.array([[1]])
            for body in reversed(bodies): obs = np.kron(obs, P[body])
            mat((np.eye(len(obs))+(-1)**(op['true'] ^ op.get('sign', 0))*obs)/2, qs)
        elif k.endswith('DESTRUCT'):
            i = live.index(qs[0]); out = np.zeros(len(v)//2, complex)
            for x, a in enumerate(v):
                b = (x >> i) & 1; y = (x & ((1 << i)-1)) | ((x >> (i+1)) << i)
                if k == 'MX_DESTRUCT': out[y] += a*(-1)**(b*op['true'])/np.sqrt(2)
                elif b == op['true']: out[y] += a
            v = out; live.pop(i)
        elif k != 'I': raise ValueError(k)
    order = list(branch.outputs)+env; result = np.zeros_like(v)
    for x, a in enumerate(v):
        y = sum(((x >> live.index(q)) & 1) << j for j, q in enumerate(order)); result[y] = a
    return result


class UnifiedTests(unittest.TestCase):
    def check_ir(self, ir, assignment):
        sim = Simulator(ir, Options(cache_bytes=0)); c = sim.compile(assignment)
        v = physical_vector(c.branch); width = len(ir.outputs)+len(c.normal.environment)
        np.testing.assert_allclose(path_vector(c.normal), v, atol=2e-12)
        if c.absorbed.zero:
            self.assertLess(np.max(abs(v)), 2e-12)
        else: np.testing.assert_allclose(path_vector(c.absorbed), v, atol=2e-12)
        np.testing.assert_allclose(expansion_vector(c.expansion, width), v, atol=2e-12)
        self.assertAlmostEqual(float(c.probability.real_decimal()), float(np.vdot(v, v).real), places=12)
        self.assertEqual(c.probability, Simulator(ir, Options(backend='fourier')).compile(assignment).probability)
        return sim, c, v

    def test_small_instruments_all_records(self):
        initial = StabilizerKet(2, 1, (1, 2), (1, 3), (2, 0), -2, 3)
        ops = [dict(kind='H', qubits=[9]), dict(kind='T', qubits=[9]), dict(kind='S', qubits=[2], exponent=-2),
               dict(kind='CX', qubits=[9, 2]), dict(kind='PAULI', qubits=[9], pauli='Y'),
               dict(kind='MP', qubits=[9, 2], paulis='YX', record='a'),
               dict(kind='CONTROL', qubits=[]), dict(kind='MZ', qubits=[9], record='b'),
               dict(kind='RESET_PLUS', qubits=[2]), dict(kind='T', qubits=[2], exponent=-1),
               dict(kind='MX_DESTRUCT', qubits=[2], record='c')]
        ops = tuple(dict(o, id=str(i)) for i, o in enumerate(ops))
        ir = InstrumentIR((9, 2), (9,), ops, initial, {'6': ControlNode(('a',), 'test-v1',
                         lambda r: [dict(kind='T', qubits=[9], exponent=1+2*r['a'])])})
        total = 0
        for a, b, c in itertools.product(range(2), repeat=3):
            _, comp, _ = self.check_ir(ir, SampleAssignment(dict(a=a, b=b, c=c), report_flips=('a',)))
            total += float(comp.probability.real_decimal())
        self.assertAlmostEqual(total, 1, places=12)

    def test_random_noncommuting_networks(self):
        rng = random.Random(617104)
        for n in (1, 2, 3):
            for trial in range(20):
                initial = StabilizerKet(n, rng.randrange(1 << n), tuple(1 << j for j in range(n)),
                                        tuple(rng.randrange(4) for _ in range(n)),
                                        tuple(rng.randrange(1 << n) & ~((1 << (j+1))-1) for j in range(n)), -n, rng.randrange(8))
                ops = []; true = {}
                for j in range(8):
                    kind = rng.choice(['H', 'S', 'T', 'MP', 'RESET_ZERO', 'Y']+(['CX', 'CZ'] if n > 1 else []))
                    qs = rng.sample(list(range(n)), 2 if kind in ('CX', 'CZ') else 1)
                    op = dict(id=str(j), kind=kind, qubits=qs)
                    if kind == 'MP':
                        op.update(paulis=rng.choice('XYZ'), record=f'm{j}'); true[f'm{j}'] = rng.randrange(2)
                    if kind in ('S', 'T'): op['exponent'] = rng.choice((-1, 1))*(2 if kind == 'S' else 1)
                    ops.append(op)
                self.check_ir(InstrumentIR(tuple(range(n)), tuple(range(n)), tuple(ops), initial), SampleAssignment(true))

    def test_gaussian_internal_contraction(self):
        rng = random.Random(9001)
        for n in range(7):
            for _ in range(20):
                width = rng.randrange(4); cols = tuple(rng.randrange(1 << width) for _ in range(n))
                linear = tuple(rng.randrange(4) for _ in range(n))
                upper = tuple(rng.randrange(1 << n) & ~((1 << (j+1))-1) for j in range(n))
                base = rng.randrange(1 << width); phase = rng.randrange(8)
                expected = np.zeros(1 << width, complex)
                for x in range(1 << n):
                    y = base; q = 0
                    for j in range(n):
                        if x >> j & 1:
                            y ^= cols[j]; q += linear[j]+2*sum((upper[j] >> k)&(x >> k)&1 for k in range(j+1, n))
                    expected[y] += np.exp(1j*np.pi*(phase+2*q)/4)*2**(-n/2)
                ket = StabilizerKet.contract(width, base, cols, linear, upper, -n, phase)
                got = np.zeros_like(expected) if ket is None else ket_vector(ket)
                np.testing.assert_allclose(got, expected, atol=2e-12)

    def test_fixed_input_absorption_and_noise(self):
        ir = InstrumentIR((31,), (31,), (dict(id='t', kind='T', qubits=[31]),), StabilizerKet.positive(1))
        sim, c, _ = self.check_ir(ir, SampleAssignment({}))
        self.assertEqual(c.resource['quotient_rank'], 0)
        ir = InstrumentIR((0,), (0,), tuple(dict(id=str(j), kind=k, qubits=[0]) for j, k in enumerate(('H', 'S', 'H', 'T'))), StabilizerKet.positive(1))
        self.check_ir(ir, SampleAssignment({}, (dict(after='1', pauli='Y', qubit=0),)))
        plus = InstrumentIR((0, 1, 2), (0, 1, 2), (dict(id='ccz', kind='CCZ', qubits=[0, 1, 2]),), StabilizerKet.positive(3, (1, 2, 4)))
        _, c, _ = self.check_ir(plus, SampleAssignment({}))
        self.assertEqual(c.resource['quotient_rank'], 3); self.assertEqual(len(c.expansion.terms), 2)

    def test_output_environment_observables(self):
        ops = (dict(id='h', kind='H', qubits=[0]), dict(id='cx', kind='CX', qubits=[0, 1]), dict(id='r', kind='RESET_ZERO', qubits=[1]))
        ir = InstrumentIR((0, 1), (0, 1), ops, StabilizerKet.positive(2))
        sim, c, v = self.check_ir(ir, SampleAssignment({}))
        self.assertEqual(sim.evaluate_observable(c, z=1), ExactScalar())
        target = StabilizerExpansion(((ONE, StabilizerKet.positive(2)),))
        self.assertEqual(sim.evaluate_observable(c, target=target), ExactScalar((1, 0, 0, 0), -1))
        with self.assertRaises(ValueError): sim.evaluate_observable(c, x=4)

    def test_zero_limits_cache_and_proposal_statistics(self):
        ir = InstrumentIR((0,), (0,), (dict(id='m', kind='MZ', qubits=[0], record='m'),), StabilizerKet.positive(1))
        sim = Simulator(ir, Options(cache_entries=1))
        a = SampleAssignment({'m': 0}, proposal_probability=Fraction(1, 3))
        c = sim.compile(a); c.resource['quotient_rank'] = 999
        self.assertEqual(sim.compile(a).resource['quotient_rank'], 0)
        self.assertTrue(sim.compile(SampleAssignment({'m': 0}, proposal_probability=Fraction(1, 4))).cache_hit)
        zero = sim.sample_one(SampleAssignment({'m': 1}, proposal_probability=Fraction(2, 3)))
        self.assertEqual(zero.status, 'zero'); self.assertEqual(zero.weight_decimal(), 0)
        self.assertGreater(cache_info(sim.cache)['evictions'], 0)
        r = estimate([sim.sample_one(a), zero, zero])
        self.assertEqual(r['acceptance'], 1); self.assertEqual(r['proposals'], 3); self.assertEqual(r['zero_weights'], 2)
        bad = sim.sample_one(SampleAssignment({}), capture_failures=True)
        self.assertEqual(estimate([bad, zero])['status'], 'incomplete')
        ir2 = InstrumentIR((0,), (0,), (dict(id='t', kind='T', qubits=[0]),), StabilizerKet.positive(1, (1,)))
        with self.assertRaises(ResourceLimit): Simulator(ir2, Options(max_quotient_rank=0)).compile(SampleAssignment({}))
        with self.assertRaises(ValueError): IndependentProposal({'m': 0})
        with self.assertRaises(ValueError): sim.compile(SampleAssignment({'m': 2}))

    def test_causal_callback_and_ir_rejection(self):
        with self.assertRaises(ValueError):
            InstrumentIR((0,), (0,), (dict(id='c', kind='CONTROL'),), StabilizerKet.positive(1),
                         {'c': ControlNode(('future',), 'v1', lambda r: ())})
        ir = InstrumentIR((0,), (0,), (dict(id='m', kind='MZ', qubits=[0], record='a'), dict(id='c', kind='CONTROL')),
                          StabilizerKet.positive(1), {'c': ControlNode(('a',), 'v1', lambda r: r['hidden'])})
        with self.assertRaises(KeyError): Simulator(ir).compile(SampleAssignment({'a': 0}))

    def test_initial_path_budget_and_invalid_gate_lifetimes(self):
        initial = StabilizerKet.positive(2, (1, 2))
        ir = InstrumentIR((0, 1), (0, 1), (), initial)
        failed = Simulator(ir, Options(max_path_bits=1)).sample_one(SampleAssignment({}), capture_failures=True)
        self.assertEqual(failed.status, 'ResourceLimit')
        self.assertIsNone(failed.compiled)
        for op in (dict(id='cx', kind='CX', qubits=[0, 1, 2]),
                   dict(id='reset', kind='RESET_ZERO', qubits=[9])):
            invalid = InstrumentIR((0, 1, 2), (0, 1, 2), (op,), StabilizerKet.positive(3))
            with self.assertRaises(ValueError): Simulator(invalid).compile(SampleAssignment({}))


if __name__ == '__main__': unittest.main()
