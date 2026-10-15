"""Ownership, record semantics and redundant-work regression for batch inputs."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
from unittest.mock import patch
import unittest
from magna.operations import (Simulator, Options, InstrumentIR, StabilizerKet, SampleAssignment,
                                        NoiseModel, NoiseSite, ControlNode)
from magna.batch import CompactAssignment
from magna.ir import instantiate_branch


def instrument():
    return InstrumentIR((0,), (0,), (dict(id='h',kind='H',qubits=[0]),
        dict(id='t',kind='T',qubits=[0]),dict(id='m',kind='MZ',qubits=[0],record='m')),
        StabilizerKet.positive(1))


class BatchTests(unittest.TestCase):
    def test_no_probability_fingerprint_without_cache(self):
        for options in (Options(cache_entries=0), Options(cache_bytes=0)):
            sim=Simulator(instrument(),options);sim.prepare()
            with patch.object(sim,'_assignment_key',side_effect=AssertionError('unused key')):
                self.assertEqual(len(list(sim.probabilities([SampleAssignment({'m':0})]*3))),3)
                with self.assertRaises(ValueError): sim.probability(SampleAssignment({'m':2}))
            self.assertTrue(sim.compile(SampleAssignment({'m':0})).cache_key)
        sim=Simulator(instrument());sim.prepare()
        with patch.object(sim,'_assignment_key',wraps=sim._assignment_key) as key:
            sim.probability(SampleAssignment({'m':0}));self.assertEqual(key.call_count,0)

    def test_immutable_group_with_report_errors_and_mutation(self):
        ir=instrument();sim=Simulator(ir)
        noise=NoiseModel((NoiseSite('t',(0,),(('X',),('Y',),('Z',)),Fraction(1,3)),
                          NoiseSite('m',(),(('REPORT','m'),),Fraction(1,5))))
        row=[(0,3),(1,1)]
        stream=sim.compact_noise_assignments(noise,[row],(0,1))
        a=next(stream);row.clear();b=next(stream)
        self.assertIs(a.faults,b.faults)
        self.assertEqual((a.true_word,b.true_word),(1,0))
        with self.assertRaises(FrozenInstanceError):a.faults[0].pauli='Z'
        with self.assertRaises(TypeError):a.true.index['m']=9
        ref=[SampleAssignment({'m':bit},(dict(after='t',qubit=0,pauli='Y'),),('m',)) for bit in (1,0)]
        self.assertEqual(list(sim.probabilities([a,b])),list(sim.probabilities(ref)))
        # Cached compilation still produces independent, serializable snapshots.
        self.assertEqual(sim.compile(a).probability,sim.compile(ref[0]).probability)
        ir.operations[0]['kind']='I'
        with self.assertRaisesRegex(ValueError,'different IR'): sim.probability(a)

    def test_bad_codes_and_duplicate_report_flips(self):
        sim=Simulator(instrument(),Options(cache_entries=0))
        noise=NoiseModel((NoiseSite('m',(),(('REPORT','m'),),Fraction(1,5)),))
        for rows in ([[(0,2)]],[[(1,1)]],[[(-1,1)]],[[(0,True)]]):
            with self.assertRaises(ValueError):list(sim.compact_noise_assignments(noise,rows,(0,)))
        a=next(sim.compact_noise_assignments(noise,[[(0,1),(0,1)]],(0,)))
        self.assertEqual(a.true_word,0);self.assertEqual(a.report_flips,())
        with self.assertRaises(ValueError):CompactAssignment(sim._prepare(),2)

    def test_compact_callbacks_execute_and_batch_snapshot(self):
        calls=[]
        def cb(r): calls.append(dict(r));return [dict(kind='X' if r['m'] else 'I',qubits=[0])]
        ir=InstrumentIR((0,),(0,),(dict(id='m',kind='MZ',qubits=[0],record='m'),
            dict(id='c',kind='CONTROL')),StabilizerKet.positive(1),{'c':ControlNode(('m',),'v1',cb)})
        sim=Simulator(ir,Options(cache_entries=0));noise=NoiseModel(())
        a=next(sim.compact_noise_assignments(noise,[[]],(0,)))
        def values():
            yield a
            ir.initial=StabilizerKet.positive(1,base=1)
            yield a
        self.assertEqual(list(sim.probabilities(values())),[sim.compile_instantiated(
            instantiate_branch(sim._prepared.ir,SampleAssignment({'m':0}),prepared=sim._prepared)).probability]*2)
        self.assertEqual(len(calls),3)
        with self.assertRaises(ValueError):sim.probability(a)

if __name__=='__main__':unittest.main()
