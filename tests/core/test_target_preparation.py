"""Independent target densities and exact normalization of auxiliary branches."""
from random import Random
import unittest

from magna import compile_circuit, enumerate_errors, parse_circuit
from tests.support import file_hooks as hooks


class TargetPreparationTests(unittest.TestCase):
    def projected(self, branch):
        return dict(circuit=parse_circuit('H 0 1 2 3 4\nCCZ 0 3 4\nCCZ 1 2 4\nH 4'),
                    outputs=[0, 1, 2, 3], postselection={4: branch})

    def test_quadratic_branches_against_basis_probabilities(self):
        for branch, count in ((0, 10), (1, 6)):
            for word in range(16):
                gates = ['I 0 1 2 3']+[f'X {q}' for q in range(4) if word >> q & 1]
                compiled = compile_circuit(parse_circuit('\n'.join(gates)), ideal_state=self.projected(branch))
                draw = compiled.sample(Random(0))
                parity = ((word >> 0 & 1)*(word >> 3 & 1)) ^ ((word >> 1 & 1)*(word >> 2 & 1))
                self.assertEqual(draw['acceptance'], 1)
                self.assertAlmostEqual(draw['error'], 1-(parity == branch)/count)
                self.assertEqual(compiled._execution.state()['reference_network_resource_constructions'], 0)

    def test_trace_and_encoded_output_order(self):
        # Tracing one half of a Bell pair gives I/2, so every pure output has overlap 1/2.
        target = dict(circuit=parse_circuit('H 0\nCX 0 1'), outputs=[1])
        for physical in ('I 0', 'X 0', 'H 0\nT 0'):
            draw = compile_circuit(parse_circuit(physical), ideal_state=target).sample(Random(1))
            self.assertAlmostEqual(draw['error'], .5)
        # A traced, independent ancilla must contribute identity, including its factor two.
        target = dict(circuit=parse_circuit('H 0 1\nT 1\nCX 1 2'), outputs=[2, 1])
        draw = compile_circuit(parse_circuit('H 1\nT 1\nCX 1 0'), ideal_state=target).sample(Random(2))
        self.assertAlmostEqual(draw['error'], 0)

    def test_full_output_target_does_not_change_acceptance(self):
        physical = parse_circuit('I 0 1 2 3\nH 4')
        target = dict(circuit=parse_circuit('H 0 1 2 3 5\nCCZ 0 3 5\nCCZ 1 2 5\nH 5'),
                      outputs=[0, 1, 2, 3, 4], postselection={5: 0})
        result = enumerate_errors(physical, ideal_state=target, max_weight=0)
        self.assertEqual(result['status'], 'complete', result)
        self.assertEqual(result['totals']['acceptance'], 1)
        self.assertAlmostEqual(result['totals']['error'], .95)
        # A fault crosses the quadratic target boundary. Check both the
        # complete error probability and the omitted-sector normalization.
        noisy = parse_circuit('X 3\nX_ERROR(1/4) 0')
        complete = enumerate_errors(noisy, ideal_state=self.projected(0), max_weight=1)
        self.assertAlmostEqual(complete['totals']['error'], .925)
        partial = enumerate_errors(noisy, ideal_state=self.projected(0), max_weight=0)
        self.assertAlmostEqual(partial['totals']['error'], .675)
        self.assertEqual(partial['totals']['full_distribution_bounds']['error'], [.675, .925])

    def test_invalid_and_zero_target_boundaries(self):
        physical = parse_circuit('I 0')
        for target in (dict(circuit=parse_circuit('I 0 1'), outputs=[0], postselection={1: 1}),
                       dict(circuit=parse_circuit('I 0 1'), outputs=[0], postselection={0: 0}),
                       dict(circuit=parse_circuit('I 0 1'), outputs=[2]),
                       dict(circuit=parse_circuit('H 0\nM 0'), outputs=[0])):
            with self.assertRaises(ValueError):
                compile_circuit(physical, ideal_state=target)

    def test_encoded_target_includes_full_physical_output(self):
        target=parse_circuit('H 0\nCX 0 1')
        physical=parse_circuit('H 0\nCX 0 1\nX_ERROR(1/4) 1')
        result=enumerate_errors(physical,ideal_state=target,max_weight=1)
        self.assertEqual(result['status'],'complete',result)
        self.assertEqual(result['totals']['acceptance'],1)
        self.assertEqual(result['totals']['error'],.25)
        for gates,error in [('',0),('X 1',1),('Z 0\nZ 1',0)]:
            compiled=compile_circuit(parse_circuit('H 0\nCX 0 1\n'+gates),ideal_state=target)
            self.assertEqual(compiled.sample(Random(1))['error'],error)
        with self.assertRaisesRegex(ValueError,'unknown task fields'):
            compile_circuit(physical,task={'target_checks':[[0]]},ideal_state=target)

    def test_cancelled_magic_keeps_one_sided_pauli_updates(self):
        # Identical ideal protocol/target phases cancel, but a physical Pauli
        # on only the protocol side can leave a nontrivial Clifford phase.
        cases = ((1, 'H 0\nT 0', (1, .5, .5, 0)),
                 (2, 'H 0 1\nCS 0 1', (1, .25, .25, 0)),
                 (3, 'H 0 1 2\nCCZ 0 1 2', (1, .25, 0, 0)))
        for width, preparation, overlaps in cases:
            for history in (False, True):
                for pauli, overlap in zip('IXYZ', overlaps):
                    with self.subTest(width=width, history=history, pauli=pauli):
                        physical = preparation
                        if pauli != 'I':
                            physical += f'\n{pauli}_ERROR(1) 0'
                        task = {'outputs': list(range(width))}
                        if history:
                            physical += f'\nH {width}\nM {width}'
                            task['sample_records'] = [0]
                        compiled = compile_circuit(parse_circuit(physical), task=task,
                            ideal_state=parse_circuit(preparation),
                            options={'max_quotient_rank': 0, 'max_terms': 1})
                        for draw in hooks.samples(compiled, Random(5), 4):
                            self.assertEqual(draw['acceptance'], 1)
                            self.assertEqual(draw['error'], 1-overlap)
                        parents = compiled.inspect()['compilation']['query_families']
                        self.assertTrue(all(p['reference_phase_coordinates'] == 0
                                            for rows in parents.values() for p in rows))
                        state = compiled._execution.state()
                        self.assertEqual(state['reference_network_coordinate_constructions'], 0)
                        self.assertEqual(state['reference_network_resource_constructions'], 0)


if __name__ == '__main__':
    unittest.main()
