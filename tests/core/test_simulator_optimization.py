"""Independent exact contraction oracle and prepared-cache invalidation tests."""
import random
import unittest
from fractions import Fraction
from magna.operations import (InstrumentIR, StabilizerKet, Simulator, SampleAssignment, ControlNode,
                                        Options)
from magna.operations import NoiseModel, NoiseSite
from magna.gf2 import image_basis
from magna.scalar import ExactScalar
from magna.file_api import ResourceLimit
from tests.support.operation_hooks import (cache_clear, cache_info, cache_probabilities, cache_store, overlap,
                                           restrict, right_inverse_columns, scalar, solve)


def quadratic(x, linear, upper):
    return (sum(l for j,l in enumerate(linear) if x>>j&1)
            +2*sum(1 for j in range(len(linear)) for k in range(j+1,len(linear))
                   if x>>j&1 and x>>k&1 and upper[j]>>k&1)) % 4


def tensor(base, columns, linear, upper, power, phase):
    out={}
    for x in range(1<<len(columns)):
        y=base
        for j,c in enumerate(columns):
            if x>>j&1:y^=c
        out[y]=out.get(y,ExactScalar())+scalar((1,0),phase+2*quadratic(x,linear,upper),power)
    return {k:v for k,v in out.items() if v}


class OptimizationTests(unittest.TestCase):
    def test_unbiased_noise_preserves_exact_rng_stream(self):
        noise=NoiseModel((NoiseSite('a',(0,), (('X',),('Y',),('Z',)),Fraction(1,3)),
                          NoiseSite('b',(),(('REPORT','r'),),Fraction(2,5))))
        left=random.Random(123);right=random.Random(123)
        for _ in range(100):
            faults=[];flips=[]
            for site in noise.sites:
                if right.randrange(site.probability.denominator)<site.probability.numerator:
                    channel=site.channels[right.randrange(len(site.channels))]
                    if channel[0]=='REPORT':flips.append(channel[1])
                    else:faults.extend(dict(after=site.after,qubit=q,pauli=p) for q,p in zip(site.qubits,channel) if p!='I')
            self.assertEqual(noise.draw(left),(tuple(faults),tuple(flips),Fraction(1)))
            self.assertEqual(left.getstate(),right.getstate())
        left=random.Random(12);right=random.Random(12)
        for _ in range(20):
            self.assertEqual(noise.draw(left,physical_rate=.125),noise.draw(right,physical_rate=.125,proposal_scale=1))
        with self.assertRaises(ValueError):noise.draw(left,physical_rate=2)

    def test_common_support_overlap_including_affine_shift_and_phase(self):
        rng=random.Random(617109)
        for n in range(7):
            width=n+1
            for _ in range(20):
                columns=tuple(image_basis([rng.randrange(1<<width) for _ in range(n)]))
                k=len(columns);kets=[]
                for side in range(2):
                    kets.append(StabilizerKet(width,rng.randrange(1<<width),columns,
                        tuple(rng.randrange(4) for _ in range(k)),
                        tuple(rng.randrange(1<<k)&~((1<<(j+1))-1) for j in range(k)),
                        rng.randrange(-10,1),rng.randrange(8)))
                a,b=kets
                va=tensor(a.base,a.columns,a.linear,a.upper,a.power,a.phase)
                vb=tensor(b.base,b.columns,b.linear,b.upper,b.power,b.phase)
                expected=sum((v.conjugate()*vb.get(y,ExactScalar()) for y,v in va.items()),ExactScalar())
                self.assertEqual(overlap(a,b),expected)

    def test_exact_sparse_contraction_with_dependent_boundaries(self):
        rng=random.Random(617106)
        for n in range(10):
            for trial in range(24):
                width=rng.randrange(5);base=rng.randrange(1<<width)
                columns=tuple(rng.randrange(1<<width) for _ in range(n))
                linear=tuple(rng.randrange(4) for _ in range(n))
                upper=tuple(sum(1<<k for k in range(j+1,n) if rng.random() < (.2 if trial%2 else .8)) for j in range(n))
                power=rng.randrange(-12,3);phase=rng.randrange(8)
                expected=tensor(base,columns,linear,upper,power,phase)
                ket=StabilizerKet.contract(width,base,columns,linear,upper,power,phase)
                found={} if ket is None else tensor(ket.base,ket.columns,ket.linear,ket.upper,ket.power,ket.phase)
                self.assertEqual(found,expected,(n,trial))

    def test_packed_pullback_against_direct_quadratic_values(self):
        rng=random.Random(617107)
        for n in range(9):
            linear=tuple(rng.randrange(4) for _ in range(n))
            upper=tuple(rng.randrange(1<<n)&~((1<<(j+1))-1) for j in range(n))
            for _ in range(12):
                basis=tuple(rng.randrange(1<<n) for _ in range(5));shift=rng.randrange(1<<n)
                q0,ln,bn=restrict(linear,upper,shift,basis)
                for u in range(32):
                    x=shift
                    for j,b in enumerate(basis):
                        if u>>j&1:x^=b
                    self.assertEqual((q0+quadratic(u,ln,bn))%4,quadratic(x,linear,upper))

    def test_batched_inverse_same_free_coordinate_convention(self):
        rng=random.Random(617108)
        for n in range(13):
            for _ in range(12):
                rows=tuple(image_basis([rng.randrange(1<<n) for _ in range(rng.randrange(n+1))]))
                columns=right_inverse_columns(rows,n)
                for j,c in enumerate(columns):
                    rhs=[int(i==j) for i in range(len(rows))]
                    self.assertEqual(c,solve(rows,rhs,n)[0])
        with self.assertRaises(ValueError):right_inverse_columns((1,1),1)

    def test_prepared_ir_nested_mutation_and_returned_snapshot(self):
        ir=InstrumentIR((0,),(0,),(
            dict(id='h',kind='H',qubits=[0],metadata={'marks':[1]}),
            dict(id='m',kind='MZ',qubits=[0],record='m')),StabilizerKet.positive(1))
        sim=Simulator(ir);a=SampleAssignment({'m':0})
        original=sim.compile(a);self.assertEqual(sim.probability(a),ExactScalar((1,0,0,0),-1))
        original.branch.operations[0]['qubits'][0]=999
        original.branch.operations[0]['metadata']['marks'][0]=999
        hit=sim.compile(a);self.assertTrue(hit.cache_hit)
        self.assertEqual(hit.branch.operations[0]['qubits'],[0])
        self.assertEqual(hit.branch.operations[0]['metadata']['marks'],[1])
        ir.operations[0]['kind']='I'
        self.assertEqual(sim.probability(a),ExactScalar((1,0,0,0)))
        ir.operations[0]['metadata']['marks'][0]=2
        self.assertFalse(sim.compile(a).cache_hit)
        self.assertEqual(sim.compile(a).branch.operations[0]['metadata']['marks'],[2])
        ir.initial=StabilizerKet.positive(1,base=1)
        self.assertEqual(sim.probability(a),ExactScalar())

    def test_compact_key_noise_reports_weights_and_limits(self):
        ir=InstrumentIR((0,),(0,),(
            dict(id='i',kind='I',qubits=[0]),dict(id='m',kind='MZ',qubits=[0],record='m')),StabilizerKet.positive(1))
        sim=Simulator(ir,Options(cache_entries=2));a=SampleAssignment({'m':0})
        one=sim.probability(a);a.true['m']=1
        self.assertEqual(sim.probability(a),ExactScalar())
        a=SampleAssignment({'m':1},(dict(after='i',qubit=0,pauli='X'),),('m',),Fraction(1,3))
        self.assertEqual(sim.probability(a),one)
        r=sim.sample_one(a);self.assertEqual(r.compiled.branch.reported,{'m':0});self.assertEqual(r.weight_decimal(),3)
        a.faults[0]['pauli']='Z';self.assertEqual(sim.probability(a),ExactScalar())
        with self.assertRaises(ValueError):sim.probability(SampleAssignment({'m':2}))
        self.assertEqual(set(cache_store(sim.cache)),set(cache_probabilities(sim.cache)))
        self.assertLessEqual(cache_info(sim.cache)['serialized_bytes'],sim.cache.budget)
        cache_clear(sim.cache);self.assertFalse(cache_probabilities(sim.cache))
        magic=InstrumentIR((0,),(),(dict(id='t',kind='T',qubits=[0]),
            dict(id='m',kind='MX_DESTRUCT',qubits=[0],record='m')),StabilizerKet.positive(1,(1,)))
        sim=Simulator(magic);a=SampleAssignment({'m':0});sim.probability(a)
        sim.options=Options(max_quotient_rank=0)
        with self.assertRaises(ResourceLimit):sim.probability(a)

    def test_control_callbacks_always_execute_and_new_gate_table_is_used(self):
        emitted={'kind':'I'};calls=[]
        def callback(records):
            calls.append(dict(records));return [dict(kind=emitted['kind'],qubits=[0])]
        ir=InstrumentIR((0,),(0,),(
            dict(id='c',kind='CONTROL'),dict(id='m',kind='MZ',qubits=[0],record='m')),
            StabilizerKet.positive(1),{'c':ControlNode((),'unchanged-version',callback)})
        sim=Simulator(ir);a=SampleAssignment({'m':0})
        self.assertTrue(sim.probability(a));self.assertTrue(sim.probability(a))
        emitted['kind']='X';self.assertFalse(sim.probability(a));self.assertEqual(len(calls),3)
        ir.controls['c']=ControlNode((),'unchanged-version',lambda r:[dict(kind='I',qubits=[0])])
        self.assertTrue(sim.probability(a))

    def test_probability_batch_snapshot_and_zero_cache_budget(self):
        ir=InstrumentIR((0,),(0,),(dict(id='m',kind='MZ',qubits=[0],record='m'),),StabilizerKet.positive(1))
        sim=Simulator(ir);a=SampleAssignment({'m':0})
        def assignments():
            yield a
            ir.initial=StabilizerKet.positive(1,base=1)
            yield a
        self.assertEqual(list(sim.probabilities(assignments())),[ExactScalar((1,0,0,0))]*2)
        self.assertEqual(sim.probability(a),ExactScalar())
        sim=Simulator(ir,Options(cache_bytes=0))
        self.assertEqual(list(sim.probabilities([a,a])),[ExactScalar()]*2)
        self.assertFalse(cache_store(sim.cache));self.assertFalse(cache_probabilities(sim.cache))

    def test_mutable_control_parent_list_updates_ir_certificate(self):
        parents=['a']
        ir=InstrumentIR((0,),(0,),(
            dict(id='a',kind='MZ',qubits=[0],record='a'),dict(id='b',kind='MZ',qubits=[0],record='b'),
            dict(id='c',kind='CONTROL'),dict(id='m',kind='MZ',qubits=[0],record='m')),
            StabilizerKet.positive(1),{'c':ControlNode(parents,'v1',lambda r:[dict(kind='X' if 'b' in r else 'I',qubits=[0])])})
        sim=Simulator(ir);a=SampleAssignment({'a':0,'b':0,'m':0});c=sim.compile(a)
        self.assertTrue(c.probability);parents.append('b');d=sim.compile(a)
        self.assertFalse(d.probability);self.assertNotEqual(c.branch.ir_fingerprint,d.branch.ir_fingerprint)
        self.assertEqual(d.branch.ir_fingerprint,ir.fingerprint)


if __name__=='__main__':unittest.main()
