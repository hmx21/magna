"""Current native contract checked with independent small physical matrices."""
import itertools
import random
import unittest
from dataclasses import replace
from unittest.mock import patch
from magna.operations import (Simulator, Options, InstrumentIR, SampleAssignment, StabilizerKet,
                                        ControlNode, PhaseFeedback)
from magna import _native
from magna.scalar import ExactScalar, ONE
from magna.file_api import ResourceLimit
import numpy as np
from magna.ir import instantiate_branch
from magna.normal import normalize_fixed_input, absorb_boundaries
from magna.evaluate import decompose_resource
from test_unified_simulator import physical_vector, expansion_vector




class KernelContractTests(unittest.TestCase):
    def test_no_old_probability_entry_in_production_extension(self):
        self.assertFalse(hasattr(_native, 'evaluate_batch'))
        self.assertFalse(hasattr(_native, 'boundary_trace_reference_test'))
        self.assertTrue(hasattr(_native, 'trace_phase_tensor'))

    def test_public_certificate_facades_keep_budgets_and_zero_semantics(self):
        width = 10
        ir = InstrumentIR(tuple(range(width)), tuple(range(width)),
            tuple(dict(id=f't{q}', kind='T', qubits=(q,)) for q in range(width)),
            StabilizerKet.positive(width, tuple(1 << q for q in range(width))))
        normal = normalize_fixed_input(instantiate_branch(ir, SampleAssignment({})))
        with self.assertRaises(ResourceLimit):
            decompose_resource(normal, max_quotient_rank=0)
        zero, resource = decompose_resource(replace(normal, zero=True))
        self.assertEqual(zero.terms, ())
        self.assertTrue(resource['zero'])
        one = replace(normal, n=1, terms=((1, 1),), power=-1,
                      boundary=((1, 0),), output_count=1)
        for backend in ('database', 'fourier'):
            expansion, resource = decompose_resource(one, backend=backend)
            np.testing.assert_allclose(expansion_vector(expansion, 1),
                np.array([1, np.exp(1j*np.pi/4)])/np.sqrt(2), atol=1e-14)
            self.assertIn('seconds', resource)
            self.assertIn('stabilizer_terms' if resource['database_hit'] else 'amplitude_terms', resource)
        marker = dict(stage='caller-certificate')
        self.assertEqual(absorb_boundaries(replace(one, certificates=(marker,))).certificates[0], marker)

    def test_random_full_records_states_faults_and_moments_against_independent_matrices(self):
        rng = random.Random(26092393)
        for trial in range(24):
            ops = []
            for j in range(10):
                kind = rng.choice(('H', 'S', 'T', 'CX', 'CZ', 'CCZ', 'PHASE'))
                arity = 3 if kind == 'CCZ' else 2 if kind in ('CX','CZ','PHASE') else 1
                op = dict(id=f'g{j}', kind=kind, qubits=tuple(rng.sample(range(3), arity)))
                if kind in ('T','S','PHASE'): op['exponent'] = rng.choice((1,3,5,7))*(2 if kind == 'S' else 1)
                ops.append(op)
                if j == 3: ops.append(dict(id='a',kind='MP',qubits=(0,1),paulis='YX',sign=trial%2,record='a'))
                if j == 6: ops.append(dict(id='reset',kind='RESET_PLUS',qubits=(1,)))
            ops.append(dict(id='b',kind='MX_DESTRUCT',qubits=(2,),record='b'))
            ir = InstrumentIR((0,1,2),(1,0),tuple(ops),StabilizerKet(3,5,(3,6),(1,3),(2,0),-2,7))
            faults = tuple(dict(after='g2',qubit=ops[2]['qubits'][0],pauli=p)
                           for p in ('XZ','ZX','Y','')[trial%4])
            flips = ('a','a','b') if trial%2 else ()
            new = Simulator(ir,Options(cache_entries=0,max_quotient_rank=10,max_terms=1024))
            total = ExactScalar()
            for a,b in itertools.product(range(2),repeat=2):
                assignment = SampleAssignment(dict(a=a,b=b),faults,flips)
                expected = physical_vector(instantiate_branch(ir, assignment))
                actual = new.compile(assignment)
                if trial < 4:
                    normal = normalize_fixed_input(instantiate_branch(ir, assignment))
                    self.assertEqual(normal.environment, ('reset',))
                    for certificate in (normal, absorb_boundaries(normal)):
                        expansion, _ = decompose_resource(certificate, max_quotient_rank=10)
                        np.testing.assert_allclose(expansion_vector(expansion, int(np.log2(len(expected)))), expected, atol=3e-12)
                np.testing.assert_allclose(expansion_vector(actual.expansion, int(np.log2(len(expected)))), expected, atol=3e-12)
                self.assertAlmostEqual(float(new.probability(assignment).real_decimal()), float(np.vdot(expected,expected).real), places=12)
                self.assertEqual(actual.probability,new.probability(assignment))
                total += actual.probability
                for x,z in ((1,0),(1,1),(0,1),(3,2)):
                    want = sum(expected[y^x].conjugate()*v*(1j)**((x&z).bit_count())*(-1)**((z&y).bit_count())
                               for y,v in enumerate(expected))
                    self.assertAlmostEqual(new.evaluate_observable(actual,x=x,z=z).complex(), want, places=12)
            self.assertEqual(total,ONE)

    def test_legacy_flags_cannot_select_an_old_kernel_and_general_callbacks_work(self):
        calls = []
        def callback(reports):
            calls.append(dict(reports))
            return [dict(kind='H',qubits=(0,)),dict(kind='T',qubits=(0,),exponent=1+reports['a'])]
        ops = (dict(id='a',kind='MZ',qubits=(0,),record='a'),dict(id='c',kind='CONTROL'),
               dict(id='b',kind='MX',qubits=(0,),record='b'))
        base = InstrumentIR((0,),(0,),ops,StabilizerKet.positive(1,(1,)),{'c':ControlNode(('a',),'literal',callback)})
        assignments = [SampleAssignment(dict(a=a,b=b),(dict(after='c',qubit=0,pauli='Y'),),('a',))
                       for a,b in itertools.product(range(2),repeat=2)]
        expected = [physical_vector(instantiate_branch(base,a)) for a in assignments]
        for declaration in (None,PhaseFeedback((0,),odd_axes=((0,),)),PhaseFeedback((99,))):
            ir = InstrumentIR(base.inputs,base.outputs,base.operations,base.initial,
                              {'c':replace(base.controls['c'],feedback=declaration)})
            for reuse in (False,True):
                sim = Simulator(ir,Options(reuse_structure=reuse,cache_entries=0))
                calls.clear()
                np.testing.assert_allclose([float(v.real_decimal()) for v in sim.probabilities(assignments)],
                    [np.vdot(v,v).real for v in expected], atol=3e-12)
                self.assertEqual(len(calls),4)
                for a,want in zip(assignments,expected):
                    np.testing.assert_allclose(expansion_vector(sim.compile(a).expansion, int(np.log2(len(want)))), want, atol=3e-12)
                self.assertEqual(len(calls),8)
                self.assertEqual(sim.execution_info['probability_kernel'],'single-sided-trace')

    def test_literal_internal_ids_do_not_collide_with_original_ids(self):
        controlled = InstrumentIR((0,),(0,),
            (dict(id='c',kind='CONTROL'),dict(id='c/0',kind='T',qubits=(0,))),
            StabilizerKet.positive(1),{'c':ControlNode((),'id-collision',
                lambda r:[dict(kind='H',qubits=(0,))])})
        faulted = InstrumentIR((0,),(0,),
            (dict(id='g',kind='H',qubits=(0,)),dict(id='g/fault/0',kind='T',qubits=(0,))),
            StabilizerKet.positive(1))
        for ir,assignment in ((controlled,SampleAssignment({})),
                (faulted,SampleAssignment({},(dict(after='g',qubit=0,pauli='Y'),)))):
            branch = instantiate_branch(ir,assignment)
            expected = physical_vector(branch)
            for options in (Options(),Options(reuse_structure=False),Options(max_structure_bytes=0)):
                sim = Simulator(ir,options)
                self.assertAlmostEqual(float(sim.probability(assignment).real_decimal()), float(np.vdot(expected,expected).real), places=12)
                found = sim.compile(assignment)
                self.assertEqual(found.branch,branch)
                np.testing.assert_allclose(expansion_vector(found.expansion, int(np.log2(len(expected)))), expected, atol=3e-12)

    def test_options_change_revalidates_resource_budget(self):
        ir=InstrumentIR((0,),(),(dict(id='t',kind='T',qubits=(0,)),
            dict(id='m',kind='MX_DESTRUCT',qubits=(0,),record='m')),StabilizerKet.positive(1,(1,)))
        sim=Simulator(ir,Options(max_quotient_rank=1,max_terms=2,max_pair_contractions=4))
        assignment=SampleAssignment({'m':0})
        self.assertTrue(sim.probability(assignment));previous=sim._prepared
        sim.options=replace(sim.options,max_quotient_rank=0)
        with self.assertRaises(ResourceLimit):sim.probability(assignment)
        self.assertIsNot(sim._prepared,previous)

    def test_unexpected_kernel_errors_are_never_retried_on_another_backend(self):
        ir=InstrumentIR((0,),(0,),(),StabilizerKet.positive(1))
        sim=Simulator(ir);sim.prepare()
        with patch.object(_native,'evaluate_full_records',side_effect=ArithmeticError('injected')) as call:
            with self.assertRaisesRegex(ArithmeticError,'injected'):sim.probability(SampleAssignment({}))
            self.assertEqual(call.call_count,1)
