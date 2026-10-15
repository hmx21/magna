"""Independent literal branch sum versus shared feedback dilation."""
import itertools
import unittest
from magna.operations import Simulator, Options
from magna.deferred_feedback import defer_affine_feedback
from magna.feedback import ControlRule, ControlGate, ControlExpression as E
from magna.ir import InstrumentIR, ControlNode, SampleAssignment
from magna.stabilizer import StabilizerKet
from tests.support.probability_tensor import trace_normal, compress_probability, clifford_scalar
from tests.support.operation_hooks import emit, record_row
from magna.scalar import ExactScalar
from magna.file_api import ResourceLimit
from magna.boundary_sampling import NativeBoundaryTrace


class DeferredFeedbackTests(unittest.TestCase):
    def test_signed_pauli_products_and_report_fault_anchors(self):
        ir=InstrumentIR((0,1),(0,1),(
            dict(id='h',kind='H',qubits=[0]),
            dict(id='m',kind='MP',qubits=[0,1],paulis='YZ',sign=1,record='a'),
            dict(id='c',kind='CONTROL',qubits=[]),
            dict(id='n',kind='MP',qubits=[0,1],paulis='XX',record='b')),
            StabilizerKet.positive(2),{'c':ControlNode(('a',),'test',
                ControlRule((ControlGate('Y',(1,),when=E.record('a')),)))})
        sim=Simulator(ir,Options(reuse_structure=False,cache_entries=0))
        trace=NativeBoundaryTrace(ir)
        rows=trace.rows((('a',),('b',)))
        for flips in ((),('a',),('b',),('a','b')):
            faults=(dict(after='m',qubit=0,pauli='Z'),)
            packed=trace.pack(faults,flips)
            for av,bv in itertools.product((0,1),repeat=2):
                true=dict(a=av^('a' in flips),b=bv^('b' in flips))
                compiled=sim.compile(SampleAssignment(true,faults,flips))
                actual=tuple(trace.query(packed,event_rows=rows,outcomes=(av,bv),x=x,z=z).total()
                             for x,z in ((0,0),(1,1),(2,0)))
                expected=tuple(sim.evaluate_observable(compiled,x=x,z=z) for x,z in ((0,0),(1,1),(2,0)))
                self.assertEqual(actual,expected)

    def test_literal_joint_events_and_output_observables(self):
        self.check_joint_events(native=False)

    def test_native_literal_joint_events_and_output_observables(self):
        self.check_joint_events(native=True)

    def check_joint_events(self, native):
        # Both destructive and nondestructive measurements; repeated record
        # use, complemented XOR, parity S and ordered Y/X feedback. No code.
        ops = (
            dict(id='h',kind='H',qubits=[0]), dict(id='t',kind='T',qubits=[1]),
            dict(id='m0',kind='MZ',qubits=[0],record='a'),
            dict(id='h2',kind='H',qubits=[0]),
            dict(id='m1',kind='MX_DESTRUCT',qubits=[0],record='b'),
            dict(id='c',kind='CONTROL',qubits=[]),
        )
        a,b=E.record('a'),E.record('b')
        rule=ControlRule((ControlGate('Y',(1,),when=a),
                          ControlGate('PHASE',(1,2),exponent=6,when=~(a^b)),
                          ControlGate('X',(2,),when=b),
                          ControlGate('S',(2,),when=1)))
        ir=InstrumentIR((0,1,2),(1,2),ops,StabilizerKet.positive(3,(2,4)),
                        {'c':ControlNode(('a','b'),'test',rule)})
        sim=Simulator(ir,Options(reuse_structure=False,cache_entries=0))
        trace=NativeBoundaryTrace(ir) if native else None
        for flips in ((),('a',),('a','a','b')):
            d=defer_affine_feedback(ir,report_flips=flips)
            sk=Simulator(d.ir,Options(max_path_bits=256))
            for faults in ((),(dict(after='h',qubit=0,pauli='Y'),),
                           (dict(after='m0',qubit=0,pauli='X'),),
                           (dict(after='c',qubit=1,pauli='Z'),)):
                n=sk.compile(SampleAssignment({},faults)).normal
                for event in (None,0,1):
                    rows=() if event is None else (record_row(d,('a','b')),)
                    outcomes=() if event is None else (event,)
                    queries=((0,0),(1,0),(1,1),(3,2))
                    if native:
                        # Production events read the reported records; each flip of a or b
                        # complements the reported parity of the true event.
                        packed=trace.pack(faults,flips)
                        native_rows=() if event is None else trace.rows((('a','b'),))
                        reported=() if event is None else ((event+flips.count('a')+flips.count('b'))%2,)
                        native_values=[trace.query(packed,event_rows=native_rows,outcomes=reported,x=x,z=z).total()
                                       for x,z in queries]
                    for qi,(x,z) in enumerate(queries):
                        expected=ExactScalar()
                        for av,bv in itertools.product((0,1),repeat=2):
                            if event is not None and av^bv!=event: continue
                            c=sim.compile(SampleAssignment(dict(a=av,b=bv),faults,flips))
                            expected+=sim.evaluate_observable(c,x=x,z=z)
                        actual=clifford_scalar(compress_probability(trace_normal(n,event_rows=rows,outcomes=outcomes,x=x,z=z)))
                        self.assertEqual(actual,expected)
                        if native: self.assertEqual(native_values[qi],expected)

    def test_explicit_unsupported_controls(self):
        ops=(dict(id='m',kind='MZ',qubits=[0],record='a'),dict(id='c',kind='CONTROL',qubits=[]))
        for cb in (lambda r: [],ControlRule((ControlGate('T',(0,),when=E.record('a')),)),
                   ControlRule((ControlGate('Z',(0,),when=E.record('a')&E.record('a')),))):
            ir=InstrumentIR((0,),(0,),ops,StabilizerKet.positive(1),{'c':ControlNode(('a',),'test',cb)})
            with self.assertRaises(NotImplementedError): defer_affine_feedback(ir)

    def test_native_wide_trace_and_explicit_resource_failure(self):
        n=129
        ir=InstrumentIR(tuple(range(n)),tuple(range(n)),
             tuple(dict(id=f't{j}',kind='T',qubits=[j]) for j in range(n)),
             StabilizerKet.positive(n,tuple(1<<j for j in range(n))))
        trace=NativeBoundaryTrace(ir)
        packed=trace.pack()
        values=tuple(trace.query(packed,event_rows=(3,),outcomes=(0,),x=x,z=z).total()
                     for x,z in ((0,0),(1<<128,0),(1<<128,1<<128)))
        half=ExactScalar((1,0,0,0),-1);isqrt=ExactScalar((0,1,0,-1),-1)
        self.assertEqual(values,(half,half*isqrt,half*isqrt))
        self.assertEqual(half*values[0]-half*isqrt*values[1]-half*isqrt*values[2],ExactScalar())
        # No implicit exponential table or rank decomposition rescues failure.
        small=InstrumentIR((0,),(0,),tuple(dict(id=str(j),kind=k,qubits=[0])
                           for j,k in enumerate('HTH')),StabilizerKet.positive(1))
        unsupported=NativeBoundaryTrace(small)
        self.assertEqual(unsupported.query(unsupported.pack(),event_rows=(1,),outcomes=(0,)).total(),
                         ExactScalar((2,1,0,-1),-2))
        with self.assertRaisesRegex(ValueError,'commute'):
            trace.query(packed,event_rows=(3,),outcomes=(0,),x=1)
        with self.assertRaises(ResourceLimit):
            NativeBoundaryTrace(ir,max_doubled_path_bits=2).query(packed,event_rows=(3,),outcomes=(0,))


class DeclaredExpressionNativeTests(unittest.TestCase):
    def test_value_conversion_and_inactive_exponent(self):
        from copy import deepcopy
        from magna.operations import ControlExpression as E, ControlGate, ControlRule
        a,b = E.record('a'),E.record('b')
        expression = 2*(a ^ ~b)+E.constant(2**200+3)
        self.assertEqual(expression,deepcopy(expression))
        for x in (0,1):
            for y in (0,1):
                self.assertEqual(expression.evaluate({'a':x,'b':y}), (2*(x ^ (1-y))+3)%8)
        gate = ControlGate('PHASE',(0,),b,when=a)
        self.assertIsNone(emit(gate,{'a':False}))
        self.assertEqual(ControlRule((gate,))({'a':True,'b':1}),[{'kind':'PHASE','qubits':[0],'exponent':1}])
        with self.assertRaises(KeyError): emit(gate,{'a':1})
