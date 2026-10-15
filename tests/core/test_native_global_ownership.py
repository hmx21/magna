"""Input ownership across GIL release and independent native workspaces."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import patch
import unittest
from magna.operations import (Simulator, Options, InstrumentIR, StabilizerKet, SampleAssignment,
                                        ControlNode, PhaseFeedback, ControlExpression, ControlGate,
                                        ControlRule)
from tests.support.operation_hooks import schedule_stats
try:
    from magna import _native
except ImportError:
    _native=None


@unittest.skipUnless(_native is not None and hasattr(_native,'make_full_record'),'native global extension required')
class NativeGlobalOwnershipTests(unittest.TestCase):
    def instrument(self, mismatch=False):
        bit=ControlExpression.record('m')
        return InstrumentIR((0,),(0,),(
            dict(id='m',kind='MZ',qubits=[0],record='m'),dict(id='c',kind='CONTROL'),
            dict(id='n',kind='MX',qubits=[0],record='n')),
            StabilizerKet.positive(1,(1,)),{'c':ControlNode(('m',),'owned-rule-v1',
                ControlRule((ControlGate('T',(0,),exponent=1+bit if mismatch else 1+6*bit),)),
                PhaseFeedback((0,),odd_axes=((0,),)))})

    def test_compile_counts_preparation_fallback_once(self):
        ir=InstrumentIR((0,),(0,),(),StabilizerKet.positive(1,(1,)))
        assignment=SampleAssignment({})
        expected=Simulator(ir,Options(reuse_structure=False,cache_entries=0)).compile(assignment)
        sim=Simulator(ir,Options(cache_entries=0,max_structure_bytes=0))
        found=sim.compile(assignment)
        self.assertEqual(found.probability,expected.probability)
        self.assertEqual(found.branch,expected.branch)
        self.assertEqual(found.normal.certificates[0]['stage'],'native_global_preparation')
        self.assertTrue(found.resource.get('native_global_compile',False))
        self.assertIsNone(sim._prepared.native_plan)
        self.assertEqual(sim.execution_info['engine'],'native')
        self.assertEqual(sim.execution_info['literal_compilations'],1)

    def test_native_fallback_retains_values_packed_before_kernel(self):
        ir=self.instrument(True);a=SampleAssignment({'m':1,'n':0})
        expected=Simulator(ir,Options(reuse_structure=False)).probability(a)
        sim=Simulator(ir,Options(cache_entries=0));sim.prepare()
        original=_native.evaluate_full_records
        def changed(*args):
            result=original(*args)
            a.true['m']=0;a.true['n']=1
            return result
        with patch.object(_native,'evaluate_full_records',side_effect=changed):
            self.assertEqual(sim.probability(a),expected)
        self.assertEqual(sim.execution_info['literal_compilations'],1)
        self.assertEqual(schedule_stats(sim._prepared.native_plan.feedback)['rule_evaluations'],0)

    def test_certificate_retains_input_before_native_return(self):
        for controlled in (False,True):
            ir=self.instrument()
            if not controlled:
                ir.controls={};ir.operations=tuple(o for o in ir.operations if o['kind']!='CONTROL')
            a=SampleAssignment({'m':1,'n':0},(dict(after='m',qubit=0,pauli='Y'),))
            expected=deepcopy(Simulator(ir,Options(reuse_structure=False)).compile(a))
            sim=Simulator(ir,Options(cache_entries=0));sim.prepare()
            original=_native.evaluate_full_records
            def changed(*args):
                result=original(*args)
                a.true['m']=0;a.faults[0]['pauli']='X'
                return result
            with patch.object(_native,'evaluate_full_records',side_effect=changed):
                found=sim.compile(a)
            self.assertEqual(found.branch,expected.branch)
            self.assertEqual(found.probability,expected.probability)

    def test_same_immutable_plan_has_independent_concurrent_workspaces(self):
        ir=self.instrument();sim=Simulator(ir,Options(cache_entries=0));sim.prepare();plan=sim._prepared.native_plan
        rows=[SampleAssignment({'m':i&1,'n':i>>1},
            (dict(after='c',qubit=0,pauli=p),)) for i in range(4) for p in 'IXYZ']
        expected=[plan.evaluate([row])[0] for row in rows]
        with ThreadPoolExecutor(max_workers=4) as pool:
            found=list(pool.map(lambda row:plan.evaluate([row])[0], rows*8))
        self.assertEqual(found,expected*8)



if __name__=='__main__':unittest.main()
