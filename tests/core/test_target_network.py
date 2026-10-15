"""Target-attached preparation and query-specific resource reuse."""
from random import Random
import unittest

from magna import compile_circuit, parse_circuit
from magna.operations import NativeBoundaryTrace
from magna.file_api import NoiseDraw
from tests.support import file_hooks as hooks


class TargetNetworkTests(unittest.TestCase):
    def test_linear_scalar_target_prepares_b_and_c(self):
        circuit = parse_circuit('H 0\nT 0\nDEPOLARIZE1(1/3) 0\nH 1\nM(1/4) 1\nCX rec[-1] 0\n'
                                'DETECTOR rec[-1]\nPOSTSELECT det[0]')
        compiled = compile_circuit(circuit, task={'outputs': [0]},
            recovery=parse_circuit('S 0'), ideal_state=parse_circuit('H 0\nT 0'))
        self.assertEqual(compiled.compilations, 2)
        self.assertGreater(compiled._execution.state()['shared_path_gates'], 0)
        before = compiled.inspect()['compilation']
        for noise in hooks.draw_many(compiled.noise_schedule, Random(7), 20):
            actual = compiled.sample(Random(2), noise=noise)
            # Independently lower the physical assignments, preserving the
            # original pre-target acceptance and the explicit target boundary.
            ir = hooks.lower(circuit, (0,), {}, noise)
            ir = hooks.attach_recovery(compiled.recovery, ir, {}, circuit.num_records, circuit.num_qubits)
            acceptance = NativeBoundaryTrace(ir)
            packed = hooks.pack(compiled, acceptance, noise)
            mass = acceptance.query(packed, event_rows=acceptance.rows((('m0',),)), outcomes=(0,)).total()
            target_ir, _ = hooks.attach_target(compiled.ideal_state, ir, circuit.num_records)
            target = NativeBoundaryTrace(target_ir)
            success = target.query(hooks.pack(compiled, target, noise),
                event_rows=target.rows((('m0',), (compiled.target_records[0],))), outcomes=(0, 0)).total()
            self.assertEqual(actual['exact']['acceptance'], mass.to_dict())
            self.assertEqual(actual['exact']['target_success'], success.to_dict())
        self.assertEqual(compiled.compilations, 2)
        self.assertEqual(compiled.inspect()['compilation'], before)

    def test_unit_target_resource_omits_unpaired_construction(self):
        circuit = parse_circuit('H 0\nT 0\nX_ERROR(1/4) 0')
        compiled = compile_circuit(circuit, ideal_state=parse_circuit('H 0\nT 0'))
        report, = compiled.inspect()['compilation']['query_families']['target']
        self.assertEqual(report['direct_term_bound'], 1)
        self.assertIsNone(report['inherited_phase_coordinates'])
        self.assertEqual(report['inherited_limit'], '')
        engine = hooks.engine(compiled, {}, NoiseDraw({}, ()), stage='target')
        anchor = next(i.op_id for i in circuit.instructions if i.name == 'X_ERROR')
        for pauli in ('I', 'X', 'Y', 'Z'):
            packed = engine.pack((dict(after=anchor, qubit=0, pauli=pauli),))
            query = engine.query(packed, event_rows=engine.rows((r,) for r in compiled.target_records), outcomes=(0,))
            self.assertTrue(query.info['prepared_query'])
            self.assertEqual(query.info['resource_constructions'], 0)
            if not query.info['zero']:
                self.assertTrue(query.info['reference_entries_reused'])

    def test_many_outputs_trace_before_any_resource_expansion(self):
        # The term/rank caps apply after internal reduction. Enumerating the
        # output amplitudes would already require 2**64 entries in the last case.
        from magna.scalar import ExactScalar
        for width in (1, 4, 16, 64):
            wires=' '.join(map(str, range(width)))
            circuit=parse_circuit(f'H {wires}\nT {wires}\nZ_ERROR(0) 0\nH {width}\nM {width}\n'
                                  'DETECTOR rec[-1]\nPOSTSELECT det[0]')
            compiled=compile_circuit(circuit, task={'outputs': list(range(width))},
                ideal_state=parse_circuit(f'H {wires}\nT {wires}'),
                options={'max_quotient_rank': 0, 'max_terms': 1})
            self.assertEqual(compiled.compilations, 2)
            self.assertGreater(compiled._execution.state()['shared_path_gates'], 0)
            anchor=next(op.op_id for op in circuit.instructions if op.name=='Z_ERROR')
            for stage in ('protocol', 'target'):
                engine=hooks.engine(compiled, {}, NoiseDraw({}, ()), stage=stage)
                events=[('m0',)]
                if stage=='target': events.extend((r,) for r in compiled.target_records)
                rows=engine.rows(events)
                for fault in (False, True):
                    packed=engine.pack((dict(after=anchor,qubit=0,pauli='Z'),) if fault else ())
                    result=engine.query(packed,event_rows=rows,outcomes=(0,)*len(rows))
                    expected=ExactScalar() if stage=='target' and fault else ExactScalar((1,0,0,0),-1)
                    self.assertEqual(result.total(),expected)
                    self.assertTrue(result.info['output_trace_reused'])
                    self.assertEqual(result.info['pairing_seconds'],0)
                    self.assertEqual(result.info['phase_coordinate_constructions'],0)
                    self.assertEqual(result.info['resource_constructions'],0)
                    self.assertLessEqual(result.info['constructive_terms'],1)


class WholeTaskRouteTests(unittest.TestCase):
    def test_terminal_decoder_selects_nonlinear_route(self):
        from magna import ControllerRegistry, run, enumerate_errors
        registry = ControllerRegistry()
        seen = []
        registry.register('and', lambda bits, config: seen.append(bits) or (bits[0] & bits[1],), version='1')
        recovery = parse_circuit('H 1 2\nM 1 2\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\nCX ctrl[0] 0', controllers=registry)
        physical = parse_circuit('I 0')
        kwargs = dict(ideal_state=parse_circuit('I 0'), recovery=recovery)
        compiled = compile_circuit(physical, **kwargs)
        self.assertEqual(physical.analysis.route, 'linear')
        self.assertEqual(compiled.inspect()['route'], 'nonlinear')
        self.assertEqual(seen, [])
        self.assertIn('recovery', compiled.inspect()['compilation']['query_families'])
        self.assertIn('target', compiled.inspect()['compilation']['query_families'])
        result = run(physical, shots=4, seed=3, **kwargs)
        self.assertEqual(result['route'], 'nonlinear')
        self.assertEqual(len(seen), 4)
        exact = enumerate_errors(physical, max_weight=0, **kwargs)
        self.assertEqual(exact['route'], 'nonlinear')
        self.assertEqual(exact['totals']['error'], .25)

    def test_target_is_unitary_on_selected_outputs_only(self):
        target = parse_circuit('H 0\nT 0\nCX 0 1')
        compiled = compile_circuit(parse_circuit('H 4\nT 4\nCX 4 2'),
            task={'outputs': [4, 2]}, ideal_state=target)
        self.assertEqual(compiled.sample(Random(0))['error'], 0)
        # Inverse gates map directly to selected wires; the target module adds
        # no independent quantum ancilla. Private readout records are effects.
        
        ir = hooks.lower(compiled.circuit, (4, 2), {}, NoiseDraw({}, ()))
        attached, _ = hooks.attach_target(compiled.ideal_state, ir, 0)
        for op in attached.operations[len(ir.operations):]:
            self.assertTrue(set(op['qubits']).issubset({4, 2}))
        for invalid in ('H 2', 'M 0', 'R 0', 'X_ERROR(1/10) 0'):
            with self.subTest(target=invalid), self.assertRaises(ValueError):
                compile_circuit(parse_circuit('I 0 1'), ideal_state=parse_circuit(invalid))


if __name__ == '__main__':
    unittest.main()
