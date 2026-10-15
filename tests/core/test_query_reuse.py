"""Intermediate reuse must preserve exact faults, event RHSs and ownership."""
from tests.support.readout_target import ZERO_TARGET
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import unittest

from magna.operations import InstrumentIR, NativeBoundaryTrace, StabilizerKet
from magna.ir import PreparedInstrument
from magna.circuit_file import ControllerRegistry, parse_circuit
from magna.file_api import NoiseDraw
from magna.file_api import ResourceLimit
from test_boundary_sampling import HAS_NATIVE
from tests.support import file_hooks as hooks


class OwnershipAndDependencies(unittest.TestCase):
    def test_ordered_distinct_dependencies_keep_callback_duplicates(self):
        registry = ControllerRegistry()
        registry.register('wide', lambda bits, config: (0,), version='test')
        c = parse_circuit('M 0 1 2\nDETECTOR rec[-3] rec[-2]\n'
            'DETECTOR rec[-1]\nCONTROL[wide] det[0] det[-1] det[0] -> ctrl[0]\n'
            'DETECTOR rec[-1]\nDETECTOR rec[-3]\nCONTROL[wide] det[-2] det[-1] -> ctrl[1]', controllers=registry)
        self.assertEqual(len(c.analysis.required_parities), 3)
        self.assertEqual(c.analysis.required_parities[0], c.analysis.expression(c.analysis.external_instructions[0].targets[0]))
        self.assertEqual(len(c.analysis.external_instructions[0].targets), 3)


    def test_public_ir_still_copies_arbitrary_graph_and_prepared_snapshot(self):
        shared = [0]
        metadata = {}; metadata['self'] = metadata
        ops = [dict(id='h', kind='H', qubits=shared, extra=metadata),
               dict(id='t', kind='T', qubits=shared)]
        ir = InstrumentIR((0,), (0,), ops, StabilizerKet.positive(1))
        self.assertIs(ir.operations[0]['qubits'], ir.operations[1]['qubits'])
        self.assertIs(ir.operations[0]['extra'], ir.operations[0]['extra']['self'])
        shared[0] = 3
        self.assertEqual(ir.operations[0]['qubits'], [0])
        plain = InstrumentIR((0,), (0,), (dict(id='h',kind='H',qubits=[0]),), StabilizerKet.positive(1))
        prepared = PreparedInstrument(plain)
        plain.operations[0]['qubits'][0] = 7
        self.assertEqual(prepared.ir.operations[0]['qubits'], [0])
        self.assertFalse(prepared.matches(plain))

    def test_deferred_public_graph_isolation_and_owned_transfer(self):
        from magna.deferred_feedback import defer_affine_feedback, _defer_affine_feedback_owned
        annotation = {'values': [1]}; annotation['cycle'] = annotation
        ir = InstrumentIR((0,), (0,), (
            dict(id='h', kind='H', qubits=[0], extra=annotation),
            dict(id='t', kind='T', qubits=[0], extra=annotation)), StabilizerKet.positive(1))
        public = defer_affine_feedback(ir)
        owned = _defer_affine_feedback_owned(ir)
        self.assertIs(owned.ir.operations[0]['extra'], ir.operations[0]['extra'])
        self.assertIs(public.ir.operations[0]['extra'], public.ir.operations[1]['extra'])
        self.assertIs(public.ir.operations[0]['extra']['cycle'], public.ir.operations[0]['extra'])
        ir.operations[0]['extra']['values'].append(2)
        ir.operations[0]['qubits'][0] = 9
        self.assertEqual(public.ir.operations[0]['extra']['values'], [1])
        self.assertEqual(public.ir.operations[0]['qubits'], [0])

    def test_lowered_operations_are_owned_even_for_guarded_noise(self):
        c = parse_circuit('CONTROL[XOR](1) -> ctrl[0]\nX_ERROR(0.1) ctrl[0] 1\n')
        noise_id = c.instructions[1].op_id
        gate = dict(kind='X',qubits=[1],extra={'nested':[2]})
        draw = NoiseDraw({noise_id:(gate,)}, ())
        ir = hooks.lower(c,(0,1),{},draw)
        before = deepcopy(ir.operations)
        gate['qubits'][0] = 99;gate['extra']['nested'].append(7)
        self.assertEqual(ir.operations,before)


@unittest.skipUnless(HAS_NATIVE, 'native extension required')
class QueryReuse(unittest.TestCase):
    def make(self):
        return InstrumentIR((0,1,2),(0,1),(
            dict(id='ccz',kind='CCZ',qubits=(0,1,2)),
            dict(id='cx',kind='CX',qubits=(0,2)),
            dict(id='m',kind='MX_DESTRUCT',qubits=(2,),record='r')),
            StabilizerKet.positive(3,(1,2,4)))

    def test_traced_network_reuse_matches_fresh_plans_and_event_changes(self):
        ir=self.make();engine=NativeBoundaryTrace(ir,max_quotient_rank=8,max_terms=256)
        row,=engine.rows((('r',),))
        # Reuse each traced structure with distinct event RHSs and faults.
        jobs=[]
        for pauli in ('I','X','Y','Z'):
            for bit in (0,1):
                faults=() if pauli=='I' else (dict(after='ccz',qubit=0,pauli=pauli),)
                for x,z in ((0,0),(1,0),(1,1),(0,1)):
                    jobs.append((faults,dict(event_rows=(row,),outcomes=(bit,),x=x,z=z)))
        for faults,kw in jobs+jobs[:4]:
            fresh=NativeBoundaryTrace(ir,max_quotient_rank=8,max_terms=256)
            expected=fresh.query(fresh.pack(faults),**kw)
            first=engine.query(engine.pack(faults),**kw)
            second=engine.query(engine.pack(faults),**kw)
            self.assertEqual(first.raw,expected.raw)
            self.assertEqual(second.raw,expected.raw)
            self.assertTrue(second.info['output_trace_reused'])
            self.assertEqual(second.info['pairing_factorizations'], 0)
            self.assertEqual(second.info['pairing_seconds'], 0)
        # Validation still executes on warm caches.
        with self.assertRaisesRegex(ValueError,'commute'):
            engine.query(engine.pack(),event_rows=(row,),outcomes=(0,),x=row)
        self.assertFalse(engine.query(engine.pack(),event_rows=(row,row),outcomes=(0,1)).total())

    def test_owned_preparation_matches_public_with_actual_controls_and_faults(self):
        c = parse_circuit('H 0 1\nT 0\nCX 0 1\nM 1\nCY rec[-1] 0\n')
        public_ir = hooks.lower(c, (0,), {}, NoiseDraw({}, ()))
        public = NativeBoundaryTrace(public_ir)
        owned = NativeBoundaryTrace._from_owned(hooks.lower(c, (0,), {}, NoiseDraw({}, ())))
        self.assertEqual(public.deferred.ir.operations, owned.deferred.ir.operations)
        # Public callers may change their source after preparing the plan.
        public_ir.operations[0]['qubits'][0] = 99
        self.assertNotEqual(public_ir.operations[0]['qubits'], public.deferred.ir.operations[0]['qubits'])
        anchor = next(op['id'] for op in owned.deferred.ir.operations if op['kind'] == 'PHASE')
        for pauli in ('I', 'X', 'Y', 'Z'):
            faults = (dict(after=anchor, qubit=0, pauli=pauli),)
            for flips in ((), ('m0',)):
                for bit in (0, 1):
                    for x, z in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        kw = dict(event_rows=owned.rows((('m0',),)), outcomes=(bit,), x=x, z=z)
                        self.assertEqual(public.query(public.pack(faults, flips), **kw).raw,
                                         owned.query(owned.pack(faults, flips), **kw).raw)
        for construct in (NativeBoundaryTrace, NativeBoundaryTrace._from_owned):
            with self.assertRaises(ValueError):
                construct(self.make(), max_path_bits=-1)

    def test_shared_plan_threads_and_batched_paulis(self):
        ir=self.make();engine=NativeBoundaryTrace(ir,max_quotient_rank=8,max_terms=256)
        queries=tuple(((),(),x,z) for x,z in ((0,0),(1,0),(0,1),(1,1)))
        assignments=[engine.pack((),()),engine.pack((),('r',)),
            engine.pack((dict(after='ccz',qubit=1,pauli='Z'),),())]
        expected=[[engine.query(a,x=x,z=z).total() for _,_,x,z in queries] for a in assignments]
        with ThreadPoolExecutor(max_workers=3) as pool:
            actual=list(pool.map(lambda a: engine.totals((a,),queries)[0][0], assignments*4))
        self.assertEqual(actual,expected*4)


if __name__ == '__main__':
    unittest.main()


class PreparedBoundaryResources(unittest.TestCase):
    def test_affine_probe_does_not_expand_a_nonaffine_resource(self):
        circuit=parse_circuit('H 0\nT 0\nH 0\nM 0')
        engine=NativeBoundaryTrace(hooks.lower(circuit, (0,), {}, NoiseDraw({}, ())),max_quotient_rank=0)
        rows=engine.rows((('m0',),))
        result=engine.query(engine.pack(),keep_rows=rows,affine_only=True)
        self.assertFalse(result.info['affine'])
        self.assertEqual(result.info['resource_constructions'],0)
        with self.assertRaises(ResourceLimit):
            engine.query(engine.pack(),keep_rows=rows)

    def test_public_prepared_query_validates_rhs_and_boundary(self):
        circuit = parse_circuit('H 0\nT 0\nH 0\nM 0')
        engine = NativeBoundaryTrace(hooks.lower(circuit, (0,), {}, NoiseDraw({}, ())))
        events = engine.rows((('m0',),))
        engine.prepare_queries((((), events, 0, 0),))
        assignment = engine.pack()
        engine.query(assignment, event_rows=events, outcomes=(0,))
        for outcomes in ((), (0, 1)):
            with self.subTest(outcomes=outcomes), self.assertRaisesRegex(ValueError, 'shape'):
                engine.query(assignment, event_rows=events, outcomes=outcomes)
        for bit in (-1, 2):
            with self.subTest(bit=bit), self.assertRaisesRegex(ValueError, 'binary'):
                engine.query(assignment, event_rows=events, outcomes=(bit,))
        for specification in (((1 << 100,), (), 0, 0), ((), (1 << 100,), 0, 0),
                              ((), (), 1 << 100, 0), ((), (), 0, 1 << 100)):
            with self.subTest(specification=specification), self.assertRaisesRegex(ValueError, 'outside'):
                engine.prepare_queries((specification,))
        with self.assertRaisesRegex(ValueError, 'commute'):
            engine.prepare_queries((((), events, events[0], 0),))

    def test_file_compilation_prepares_all_stages_without_decoders_or_draws(self):
        from magna import compile_circuit, run
        from random import Random
        seen = []
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: seen.append(bits) or bits, version='1')
        protocol = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCH ctrl[0] 1', controllers=registry)
        recovery = parse_circuit('H 1\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0', controllers=registry)
        compiled = compile_circuit(protocol, task={'outputs': [1], 'sample_records': [0]},
            recovery=recovery, ideal_state=parse_circuit('H 0'))
        report = compiled.inspect()['compilation']
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(set(report['query_families']), {'recovery','target'})
        self.assertEqual(report['resource_summary']['queries'], 3)
        self.assertGreaterEqual(report['resource_summary']['max_reference_terms'], 1)
        self.assertEqual(seen, [])
        self.assertEqual(compiled.queries, 0)
        self.assertEqual(compiled.compilations, 2)
        resource_summary = deepcopy(report['resource_summary'])
        compiled.sample(Random(6))
        self.assertEqual(len(seen), 2)
        self.assertEqual(compiled.compilations, 2)
        self.assertGreater(compiled.queries, 0)
        self.assertEqual(compiled.inspect()['compilation']['resource_summary'], resource_summary)
        completed = run(protocol, shots=2, seed=7, task={'outputs': [1], 'sample_records': [0]},
                        recovery=recovery, ideal_state=parse_circuit('H 0'))
        self.assertEqual(completed['completed'], 2)
        self.assertNotIn('last_query_diagnostics', completed['resources'])
        self.assertEqual(completed['resources']['compilation']['resource_summary'], resource_summary)
        failed = run(parse_circuit('H 0\nT 0\nH 0\nM 0'), shots=3,
            task={'outputs': [0], 'ideal_state': ZERO_TARGET}, options={'max_quotient_rank': 0, 'max_terms': 0})
        self.assertEqual((failed['completed'], failed['failed'], failed['not_attempted']), (0, 0, 3))
        self.assertEqual(failed['failures'][0]['phase'], 'preparation')
        self.assertEqual(failed['resources']['queries'], 0)

    def test_prepared_entries_and_exact_event_transport(self):
        from magna.scalar import ExactScalar
        circuit = parse_circuit('H 0\nT 0\nH 0\nM 0')
        engine = NativeBoundaryTrace(hooks.lower(circuit, (0,), {}, NoiseDraw({}, ())))
        events = engine.rows((('m0',),))
        reports = engine.prepare_queries(((events, (), 0, 0), ((), events, 0, 0)), include_maps=True)
        self.assertEqual(engine.queries,0)
        self.assertEqual(len(reports),2)
        self.assertTrue(all(r['uniform_term_bound'] for r in reports))
        self.assertTrue(all('event_support_direction' in r for r in reports))
        for flips,sign in (((),1),(('m0',),-1)):
            marginal = engine.query(engine.pack(report_flips=flips),event_rows=events,outcomes=(0,))
            self.assertEqual(marginal.total(),ExactScalar((2,sign,0,-sign),-2))
            self.assertTrue(marginal.info['prepared_query'])
            self.assertTrue(marginal.info['reference_entries_reused'])
            self.assertEqual(marginal.info['resource_constructions'],0)

    def test_eager_and_lazy_preparation_match_after_faults(self):
        circuit = parse_circuit('H 0 1\nT 0\nCX 0 1\nM 1\nCY rec[-1] 0\nT 0\nH 0')
        engine = NativeBoundaryTrace(hooks.lower(circuit, (0,), {}, NoiseDraw({}, ())))
        events = engine.rows((('m0',),))
        observables = ((0, 0), (1, 0), (0, 1), (1, 1))
        reports = engine.prepare_queries(tuple(((), events, x, z) for x, z in observables))
        self.assertEqual(len(reports), len(observables))
        self.assertTrue(all(report['direct_term_bound'] or report['uniform_term_bound'] for report in reports))
        anchor = next(op['id'] for op in engine.deferred.ir.operations if op['kind'] == 'PHASE')
        zero_queries = reused_queries = 0
        for pauli in ('I', 'X', 'Y', 'Z'):
            faults = (dict(after=anchor, qubit=0, pauli=pauli),)
            for flips in ((), ('m0',)):
                for bit in (0, 1):
                    for x, z in observables:
                        with self.subTest(pauli=pauli, flips=flips, bit=bit, x=x, z=z):
                            # A fresh plan prepares this query lazily from the original IR.
                            fresh = NativeBoundaryTrace(hooks.lower(circuit, (0,), {}, NoiseDraw({}, ())))
                            kwargs = dict(event_rows=events, outcomes=(bit,), x=x, z=z)
                            expected = fresh.query(fresh.pack(faults, flips), **kwargs)
                            actual = engine.query(engine.pack(faults, flips), **kwargs)
                            self.assertTrue(expected.info['output_trace_reused'])
                            self.assertTrue(actual.info['prepared_query'])
                            self.assertEqual(actual.total(), expected.total())
                            if actual.info['zero']:
                                self.assertFalse(actual.total())
                                zero_queries += 1
                            else:
                                self.assertTrue(actual.info['reference_entries_reused'])
                                reused_queries += 1
                            self.assertEqual(actual.info['resource_constructions'], 0)
                            self.assertEqual(actual.info['phase_coordinate_constructions'], 0)
        self.assertGreater(zero_queries, 0)
        self.assertGreater(reused_queries, 0)
