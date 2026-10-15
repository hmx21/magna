"""Boundary-first results against independent literal complete-branch sums."""
from dataclasses import replace
import itertools
import random
import unittest

from magna.operations import (InstrumentIR, ControlNode, ControlRule, ControlGate,
                                        ControlExpression as E, PhaseFeedback, SampleAssignment, Simulator,
                                        Options, StabilizerKet)
from magna.scalar import ExactScalar, ONE
from magna.file_api import ResourceLimit
from tests.support import native_testing

try:
    from magna import _native
    HAS_NATIVE = hasattr(_native, 'make_boundary_trace')
except ImportError:
    HAS_NATIVE = False


@unittest.skipUnless(HAS_NATIVE, 'native boundary trace extension required')
class BoundarySamplingTests(unittest.TestCase):
    def test_virtual_record_coordinates_and_redundant_postselection(self):
        from tests.support.operation_hooks import boundary_coordinates
        layout = boundary_coordinates(('a','b','c','unused'),
            (('a','b','c'),('b','c','a')), (0,0), ())
        self.assertEqual((layout['postselection_rank'],layout['feedback_rank'],layout['trace_rank']), (1,0,3))
        self.assertTrue(layout['compatible'])
        images = {tuple((row & word).bit_count()%2 for row in layout['transform']) for word in range(16)}
        self.assertEqual(len(images),16)
        bad = boundary_coordinates(('a','b'), (('a','b'),('b','a')), (0,1))
        self.assertFalse(bad['compatible'])

    def exhaustive(self, ir, faults, flips):
        sim = Simulator(ir, Options(reuse_structure=False, cache_entries=0, max_quotient_rank=8))
        table = []
        for word in range(1 << len(ir.record_names)):
            true = {r: word >> j & 1 for j,r in enumerate(ir.record_names)}
            compiled = sim.compile(SampleAssignment(true, faults, flips))
            reports = {r: b ^ (flips.count(r)%2) for r,b in true.items()}
            moments = tuple(sim.evaluate_observable(compiled, x=x, z=z) for x,z in ((1,0),(1,1),(0,1)))
            table.append((reports,compiled.probability,moments))
        self.assertEqual(sum((p for _,p,_ in table),ExactScalar()), ONE)
        return table

    def assert_draw(self, draw, table, task):
        mass=ExactScalar(); a=ExactScalar(); e=None; moments=[ExactScalar()]*3
        for reports,p,values in table:
            if any(reports[r]!=b for r,b in draw.reports.items()): continue
            mass+=p
            accepted=all(sum(reports[r] for r in row)%2==bit for row,bit in zip(task.postselection,task.outcomes))
            if not accepted: continue
            a+=p
            moments=[a+b for a,b in zip(moments,values)]
        self.assertEqual((draw.mass,draw.acceptance,draw.error,draw.moments),(mass,a,e,tuple(moments)))

    def test_all_affine_cliffords_keep_exact_acceptance_and_effects(self):
        from magna.operations import BoundaryTask, BoundarySampler
        task=BoundaryTask((('a','b'),), observables=((1,0),(1,1),(0,1)))
        for kind,qs in (('X',(1,)),('Y',(1,)),('S',(1,)),('H',(1,)),('CX',(1,2)),('CZ',(1,2)),('PHASE',(1,2))):
            ops=(dict(id='h',kind='H',qubits=(0,)),dict(id='t',kind='T',qubits=(1,)),
                 dict(id='m',kind='MZ',qubits=(0,),record='a'),dict(id='c',kind='CONTROL',qubits=()),
                 dict(id='n',kind='MX',qubits=(2,),record='b'),dict(id='r',kind='RESET_ZERO',qubits=(0,)))
            gate=ControlGate(kind,qs,exponent=2 if kind=='PHASE' else None,when=~E.record('a'))
            ir=InstrumentIR((0,1,2),(1,2,0),ops,StabilizerKet.positive(3,(2,4)),
                            {'c':ControlNode(('a',),'test',ControlRule((gate,)))})
            sampler=BoundarySampler(ir,task)
            self.assertEqual(sampler.route,'direct')
            for flips in ((),('a',),('a','a','b')):
                faults=(dict(after='c',qubit=1,pauli='Y'),)
                draw=sampler.sample(random.Random(19),faults=faults,report_flips=flips)
                self.assertFalse(draw.reports)
                self.assert_draw(draw,self.exhaustive(ir,faults,flips),task)

    def test_complex_feedback_global_queries_with_future_postselection(self):
        from magna.operations import BoundaryTask, BoundarySampler
        ops=(dict(id='t',kind='T',qubits=(0,)),dict(id='a',kind='MX',qubits=(0,),record='a'),
             dict(id='b',kind='MZ',qubits=(1,),record='b'),dict(id='c1',kind='CONTROL',qubits=()),
             dict(id='s',kind='MX',qubits=(2,),record='s'),dict(id='c2',kind='CONTROL',qubits=()),
             dict(id='j',kind='MX_DESTRUCT',qubits=(1,),record='junk'))
        def control(reports):
            return (dict(kind='S',qubits=(2,),exponent=2*(reports['a'] & reports['b'])),)
        def correction(reports):
            return (dict(kind='X' if reports['s'] ^ reports['a'] else 'I',qubits=(0,)),)
        ir=InstrumentIR((0,1,2),(0,2),ops,StabilizerKet.positive(3,(1,2,4)),{
            'c1':ControlNode(('a','b'),'nonlinear',control,PhaseFeedback((2,))),
            'c2':ControlNode(('a','s'),'output',correction,PhaseFeedback((0,)))})
        task=BoundaryTask((('s','junk'),), observables=((1,0),(1,1),(0,1)))
        sampler=BoundarySampler(ir,task,max_quotient_rank=6)
        self.assertEqual(sampler.route,'complex-feedback')
        self.assertEqual(sampler.required,('a','b','s'))
        for faults,flips in (((),()),((dict(after='c1',qubit=2,pauli='Y'),),('a','s'))):
            table=self.exhaustive(ir,faults,flips)
            for seed in range(12):
                draw=sampler.sample(random.Random(seed),faults=faults,report_flips=flips)
                self.assertNotIn('junk',draw.reports)
                self.assert_draw(draw,table,task)
                self.assertEqual(draw.diagnostics['controls_executed'],2)

    def test_affine_phase_arithmetic_is_direct_and_odd_control_is_complex(self):
        from magna.operations import BoundaryTask, BoundarySampler
        ops=(dict(id='a',kind='MZ',qubits=(0,),record='a'),dict(id='c',kind='CONTROL',qubits=()),
             dict(id='b',kind='MX',qubits=(1,),record='b'))
        task=BoundaryTask((('b',),),observables=((1,0),(1,1),(0,1)))
        for coefficient,route in ((2,'direct'),(4,'direct'),(1,'complex-feedback')):
            rule=ControlRule((ControlGate('PHASE',(1,),exponent=2+coefficient*E.record('a')),))
            ir=InstrumentIR((0,1),(1,0),ops,StabilizerKet.positive(2,(1,2)),{'c':ControlNode(('a',),'arithmetic',rule)})
            sampler=BoundarySampler(ir,task);self.assertEqual(sampler.route,route)
            table=self.exhaustive(ir,(),('a',))
            for seed in range(3): self.assert_draw(sampler.sample(random.Random(seed),report_flips=('a',)),table,task)

    def test_unused_t_subcircuits_trace_before_resource_budget(self):
        from magna.operations import BoundaryTask, BoundarySampler
        n=20; ops=[]
        for j in range(n):
            ops.extend((dict(id=f't{j}',kind='T',qubits=(j,)),
                        dict(id=f'h{j}',kind='H',qubits=(j,)),
                        dict(id=f'm{j}',kind='MZ_DESTRUCT',qubits=(j,),record=f'm{j}')))
        # Include an unused deterministic measurement: it has the same trace
        # classification as the unused Z readouts of the T/H subcircuits.
        ops.append(dict(id='fixed',kind='MZ_DESTRUCT',qubits=(n,),record='fixed'))
        ir=InstrumentIR(tuple(range(n+1)),(),tuple(ops),StabilizerKet.positive(n+1,tuple(1<<j for j in range(n))))
        sampler=BoundarySampler(ir,BoundaryTask(),max_quotient_rank=0,max_terms=1)
        draw=sampler.sample(random.Random(2),faults=(dict(after='t3',qubit=3,pauli='Y'),),report_flips=('m4','fixed'))
        self.assertEqual(draw.acceptance,ONE)
        self.assertEqual(sampler.layout['trace_rank'],n+1)
        self.assertTrue(all(r['quotient_rank']==0 and r['constructive_terms']==1 for r in draw.diagnostics['final_resources']))

    def test_scalar_residual_and_zero_postselection(self):
        from magna.operations import BoundaryTask, BoundarySampler
        ir=InstrumentIR((0,),(),(dict(id='t',kind='T',qubits=(0,)),
            dict(id='m',kind='MX_DESTRUCT',qubits=(0,),record='a')),StabilizerKet.positive(1,(1,)))
        sampler=BoundarySampler(ir,BoundaryTask((('a',),)),max_terms=2,max_quotient_rank=2)
        draw=sampler.sample(random.Random(1))
        expected=ExactScalar((1,0,0,0),-1)+ExactScalar((0,1,0,-1),-2)
        self.assertEqual(draw.acceptance,expected)
        bad=BoundarySampler(ir,BoundaryTask((('a',),('a',)),(0,1)),max_terms=2)
        self.assertEqual(bad.sample(random.Random(1)).acceptance,ExactScalar())
        with self.assertRaises(ResourceLimit):
            BoundarySampler(ir,BoundaryTask((('a',),)),max_quotient_rank=0).sample(random.Random(1))

    def test_commuting_kernel_includes_joint_parities(self):
        from magna.operations import NativeBoundaryTrace
        ir=InstrumentIR((0,1),(),(dict(id='t',kind='PHASE',qubits=(0,1),exponent=1),
            dict(id='a',kind='MX_DESTRUCT',qubits=(0,),record='a'),
            dict(id='b',kind='MX_DESTRUCT',qubits=(1,),record='b')),
            StabilizerKet.positive(2,(1,2)))
        engine=NativeBoundaryTrace(ir)
        rows=engine.rows((('a',),('b',)))
        sector,common=native_testing.load().boundary_commuting_rows(engine.handle,rows)
        self.assertTrue(common);self.assertEqual(sector,(3,))
        for seed in range(4):
            word,mass,info=engine.sample_rows(engine.pack(),rows,random.Random(seed))
            self.assertIn(word,(0,3));self.assertEqual(info['sampled_affine_parities'],1)
            literal=Simulator(ir).probability(SampleAssignment(dict(a=word&1,b=word>>1)))
            self.assertEqual(mass,literal)

    def test_noncommuting_reference_uses_exact_global_queries(self):
        from magna.operations import BoundaryTask, BoundarySampler
        ops=(dict(id='t1',kind='T',qubits=(0,)),dict(id='h',kind='H',qubits=(0,)),
             dict(id='t2',kind='T',qubits=(0,)),dict(id='a',kind='MX',qubits=(0,),record='a'),
             dict(id='c',kind='CONTROL',qubits=()))
        ir=InstrumentIR((0,1),(1,0),ops,StabilizerKet.positive(2,(1,2)),{
            'c':ControlNode(('a',),'opaque-linear',lambda r:(dict(kind='Z' if r['a'] else 'I',qubits=(1,)),),PhaseFeedback((1,)))})
        task=BoundaryTask(observables=((1,0),(1,1),(0,1)))
        sampler=BoundarySampler(ir,task);table=self.exhaustive(ir,(),())
        for seed in range(5):
            draw=sampler.sample(random.Random(seed));self.assert_draw(draw,table,task)
            self.assertFalse(draw.diagnostics['stages'][0]['common_axis_certificate'])

    def test_undeclared_callback_and_invalid_family_fail(self):
        from magna.operations import BoundarySampler
        ir=InstrumentIR((0,),(0,),(dict(id='c',kind='CONTROL',qubits=()),),StabilizerKet.positive(1),
                        {'c':ControlNode((),'opaque',lambda _: (dict(kind='H',qubits=(0,)),))})
        with self.assertRaisesRegex(NotImplementedError,'declared unitary'): BoundarySampler(ir)
        declared=replace(ir,controls={'c':replace(ir.controls['c'],feedback=PhaseFeedback((0,)))})
        from magna.feedback import FeedbackMismatch
        with self.assertRaises(FeedbackMismatch): BoundarySampler(declared).sample(random.Random(1))


if __name__=='__main__': unittest.main()
