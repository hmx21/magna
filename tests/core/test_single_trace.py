"""Exact single-side trace checks, including finite legacy-domain coverage."""
import itertools
import random
import unittest

from magna.operations import (InstrumentIR, NativeBoundaryTrace, Options, SampleAssignment,
                                        Simulator, StabilizerKet)
from magna.scalar import ExactScalar, ONE
from magna.file_api import ResourceLimit
from test_boundary_sampling import HAS_NATIVE




@unittest.skipUnless(HAS_NATIVE, 'native extension required')
class SingleTraceTests(unittest.TestCase):
    def test_retained_labels_preserve_inherited_rank_squared_bound(self):
        # CCZ|+++> has a certified two-term decomposition. Keep ALL three
        # measurement labels: the whole distribution must fit four paired
        # terms. The generic direct phase representation can require a larger
        # Fourier expansion; its size is not the rank of the record distribution.
        ops=(dict(id='ccz',kind='CCZ',qubits=(0,1,2)),)+tuple(
            dict(id=f'm{j}',kind='MX_DESTRUCT',qubits=(j,),record=f'r{j}') for j in range(3))
        ir=InstrumentIR((0,1,2),(),ops,StabilizerKet.positive(3,(1,2,4)))
        engine=NativeBoundaryTrace(ir,max_quotient_rank=6,max_terms=4)
        keep=engine.rows((r,) for r in ir.record_names)
        for flips in ((),('r1',)):
            q=engine.query(engine.pack(report_flips=flips),keep_rows=keep)
            self.assertEqual(q.total(),ONE)
            self.assertEqual(q.info['resource_expansion'],'inherited-product')
            self.assertEqual(q.info['inherited_side_entries'],2)
            self.assertLessEqual(q.info['constructive_terms'],4)
            for word in range(8):
                actual=word^(2 if flips else 0)
                self.assertEqual(q.total((1,2,4),tuple(word>>j&1 for j in range(3))),
                    ExactScalar((9 if actual==0 else 1,0,0,0),-4))

    def test_entangled_trace_and_postselection_are_distinct(self):
        ir = InstrumentIR((0,1),(0,),(
            dict(id='t',kind='T',qubits=(0,)),dict(id='cx',kind='CX',qubits=(0,1)),
            dict(id='m',kind='MX_DESTRUCT',qubits=(1,),record='a')),
            StabilizerKet.positive(2,(1,)))
        engine = NativeBoundaryTrace(ir)
        packed = engine.pack(); row, = engine.rows((('a',),))
        self.assertEqual(engine.query(packed).total(),ONE)
        for x,z in ((1,0),(1,1),(0,1)):
            self.assertEqual(engine.query(packed,x=x,z=z).total(),ExactScalar())
            a = engine.query(packed,event_rows=(row,),outcomes=(0,),x=x,z=z)
            b = engine.query(packed,event_rows=(row,),outcomes=(1,),x=x,z=z)
            self.assertEqual(a.total()+b.total(),ExactScalar())
            if x: self.assertNotEqual(a.total(),ExactScalar())

    def test_random_instruments_exact_marginals_effects_and_basis_changes(self):
        rng = random.Random(26092381)
        for trial in range(12):
            ops = []
            for j in range(12):
                kind = rng.choice(('H','S','T','CX','CZ','CCZ','PHASE'))
                arity = 3 if kind == 'CCZ' else 2 if kind in ('CX','CZ','PHASE') else 1
                op = dict(id=f'g{j}',kind=kind,qubits=tuple(rng.sample(range(3),arity)))
                if kind in ('T','S','PHASE'): op['exponent'] = rng.choice((-1,1,3,5))*(2 if kind=='S' else 1)
                ops.append(op)
                if j == 4: ops.append(dict(id='m0',kind='MP',qubits=(0,1),paulis='YX',sign=1,record='a'))
                if j == 7: ops.append(dict(id='reset',kind='RESET_PLUS',qubits=(1,)))
            ops.append(dict(id='m1',kind='MX_DESTRUCT',qubits=(2,),record='b'))
            initial = StabilizerKet(3,5,(3,6),(1,3),(2,0),-2,7)
            equivalent = StabilizerKet(3,5,(6,3),(3,1),(2,0),-2,7)
            ir = InstrumentIR((0,1,2),(1,0),tuple(ops),initial)
            faults = (dict(after='g3',qubit=ops[3]['qubits'][0],pauli=('X','Y','Z')[trial%3]),)
            flips = ('a',) if trial%2 else ('a','b','a')
            sim = Simulator(ir,Options(reuse_structure=False,cache_entries=0,max_quotient_rank=10,max_terms=1024))
            table = []
            for a,b in itertools.product(range(2),repeat=2):
                c = sim.compile(SampleAssignment(dict(a=a,b=b),faults,flips))
                vals = [c.probability]+[sim.evaluate_observable(c,x=x,z=z) for x,z in ((1,0),(1,1),(0,1))]
                table.append(((a^(flips.count('a')%2),b^(flips.count('b')%2)),vals))
            self.assertEqual(sum((v[0] for _,v in table),ExactScalar()),ONE)
            for initial in (initial,equivalent):
                engine = NativeBoundaryTrace(InstrumentIR(ir.inputs,ir.outputs,ir.operations,initial),max_quotient_rank=10,max_terms=1024)
                packed=engine.pack(faults,flips);ra,rb=engine.rows((('a',),('b',)))
                for moment,(x,z) in enumerate(((0,0),(1,0),(1,1),(0,1))):
                    # An invertible record change: retain a, condition a XOR b.
                    for parity in (0,1):
                        kw=dict(keep_rows=(ra,),event_rows=(ra^rb,),outcomes=(parity,),x=x,z=z)
                        result=engine.query(packed,**kw)
                        for bit in (0,1):
                            expected=sum((vals[moment] for (a,b),vals in table if a==bit and a^b==parity),ExactScalar())
                            self.assertEqual(result.total((1,),(bit,)),expected)
                        self.assertFalse(result.info['fallback_used'])
                        self.assertEqual(result.info['reduction'],'single')

    def test_support_cache_offsets_redundancies_and_budget_failures(self):
        ir=InstrumentIR((0,),(),(dict(id='t',kind='T',qubits=(0,)),
            dict(id='m',kind='MX_DESTRUCT',qubits=(0,),record='a')),StabilizerKet.positive(1,(1,)))
        engine=NativeBoundaryTrace(ir,max_quotient_rank=2,max_terms=4)
        row,=engine.rows((('a',),));results=[]
        for bit in (0,1,0):
            q=engine.query(engine.pack(),event_rows=(row,row),outcomes=(bit,bit));results.append(q)
        self.assertEqual(results[0].total()+results[1].total(),ONE)
        self.assertEqual(results[-1].info['support_factorizations'],0)
        self.assertEqual(results[-1].info['pairing_factorizations'],0)
        bad=engine.query(engine.pack(),event_rows=(row,row),outcomes=(0,1))
        self.assertEqual(bad.total(),ExactScalar())
        with self.assertRaisesRegex(ValueError,'binary'): engine.query(engine.pack(),event_rows=(row,),outcomes=(2,))
        with self.assertRaisesRegex(ValueError,'commute'): engine.query(engine.pack(),event_rows=(row,),outcomes=(0,),x=row)
        with self.assertRaises(ResourceLimit):
            NativeBoundaryTrace(ir,max_quotient_rank=0).query(engine.pack(),event_rows=(row,),outcomes=(0,))
        # r single-side terms permit r^2 pairs. Coordinate dimension may double;
        # comparing equal numerical rank caps would compare different budgets.
        old=Simulator(ir,Options(cache_entries=0,max_quotient_rank=1,max_terms=2,max_pair_contractions=4))
        self.assertEqual(old.probability(SampleAssignment({'a':0})),results[0].total())
        # Only the reduced pairing consumes the pairing-path budget.
        with self.assertRaisesRegex(ResourceLimit,'max_doubled_path_bits = 0'):
            NativeBoundaryTrace(ir,max_doubled_path_bits=0).query(engine.pack())


if __name__ == '__main__': unittest.main()
