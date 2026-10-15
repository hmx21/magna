"""Quantum target readouts preserve exact events and integrate terminal draws."""
from contextlib import redirect_stdout
from io import StringIO
from random import Random
from pathlib import Path
import json
import tempfile
import unittest

from tests.support.readout_target import ZERO_TARGET, quantum_readout
from magna import ControllerRegistry, compile_circuit, parse_circuit, run, enumerate_errors
from magna.file_api import NoiseDraw, SCHEMA_VERSION, main
from magna.ir import fingerprint
from magna.scalar import ExactScalar, ONE


def lookbacks(num_records, row):
    return ''.join(f' rec[{record-num_records}]' for record in row)


def detector_selection(num_records, rows, outcomes=None):
    """Require each absolute-record parity row to equal its outcome (default zero)."""
    outcomes = [0]*len(rows) if outcomes is None else outcomes
    lines = ['DETECTOR'+lookbacks(num_records, row) for row in rows]
    lines += [f'POSTSELECT {"!" if bit else ""}det[{i}]' for i, bit in enumerate(outcomes)]
    return ''.join(line+'\n' for line in lines)


def observable_checks(num_records, rows, first=0):
    """Declare each absolute-record parity row as one fresh OBSERVABLE_INCLUDE index."""
    lines = [f'OBSERVABLE_INCLUDE({first+k})'+lookbacks(num_records, row) for k, row in enumerate(rows)]
    return ''.join(line+'\n' for line in lines), list(range(first, first+len(rows)))


class QuantumOutputTests(unittest.TestCase):
    def test_report_parities_equal_independent_selected_event(self):
        text = 'H 0\nT 0\nH 0\nM(1/3) 0\nH 1\nM 1\n'
        for rows in ([], [[0]], [[0, 1]], [[0], [1]], [[0, 0]], [[], [0]]):
            rewritten, outputs = quantum_readout(text, rows)
            checks, indices = observable_checks(2, rows)
            target = compile_circuit(parse_circuit(rewritten+detector_selection(2, [[1]])),
                task={'outputs': outputs, 'ideal_state': ZERO_TARGET})
            native = compile_circuit(parse_circuit(text+detector_selection(2, [[1]])+checks),
                task={'error_observables': indices})
            reference = compile_circuit(parse_circuit(text+detector_selection(2, [[1], *rows])),
                task={'outputs': []})
            for flips in ((), ('m0',), ('m1',), ('m0', 'm1')):
                noise = NoiseDraw({}, flips)
                actual = target.sample(Random(1), noise=noise)
                self.assertEqual(actual['exact'], native.sample(Random(1), noise=noise)['exact'])
                expected = reference.sample(Random(1), noise=noise)
                accepted = ExactScalar(**actual['exact']['acceptance'])
                error = ExactScalar(**actual['exact']['error'])
                self.assertEqual(accepted-error, ExactScalar(**expected['exact']['acceptance']))
                self.assertEqual(target.compilations, 2 if rows else 1)
                state = target._execution.state()
                self.assertEqual(state['reference_network_preparations'], 2)
                if rows:
                    self.assertGreater(state['target_frame_reuses'], 0)

    def test_causal_callbacks_and_both_history_methods(self):
        calls = []
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: calls.append(bits) or bits, version='1')
        text = 'H 0\nM(1/3) 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 1\nM 1\n'
        rewritten, outputs = quantum_readout(text, [[0, 1]], controllers=registry)
        for method in ('affine_support', 'exact_marginal'):
            target = compile_circuit(parse_circuit(rewritten, controllers=registry),
                task={'outputs': outputs, 'ideal_state': ZERO_TARGET}, options={'history_sampler': method})
            native = compile_circuit(parse_circuit(text+'OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]\n', controllers=registry),
                task={'error_observables': [0]}, options={'history_sampler': method})
            for flips in ((), ('m0',)):
                result = target.sample(Random(19), noise=NoiseDraw({}, flips))
                self.assertEqual(result['exact'], native.sample(Random(19), noise=NoiseDraw({}, flips))['exact'])
                self.assertEqual((result['acceptance'], result['error'], result['decoder_calls']), (1, 0, 1))
            self.assertEqual(target._execution.state()['reference_network_preparations'], 3)

    def test_unsampled_quantum_output_is_terminal_conditional_expectation(self):
        quantum = compile_circuit(parse_circuit('H 0\nT 0'), ideal_state=parse_circuit('H 0'))
        integrated = quantum.sample(Random(1))
        measured = compile_circuit(parse_circuit('H 0\nT 0\nMX 0'),
            task={'sample_records': [0]}, ideal_state=parse_circuit('H 0'),
            options={'history_sampler': 'exact_marginal'})
        branches = {}
        for seed in range(64):
            value = measured.sample(Random(seed))
            branches[value['sampled_records'][0]] = value
            if len(branches) == 2:
                break
        self.assertEqual(set(branches), {0, 1})
        masses = [ExactScalar(**branches[b]['exact']['history_mass']) for b in (0, 1)]
        errors = [ExactScalar(**branches[b]['exact']['error']) for b in (0, 1)]
        self.assertEqual(sum(masses, ExactScalar()), ONE)
        self.assertEqual(sum(errors, ExactScalar()), ExactScalar(**integrated['exact']['error']))
        self.assertEqual([branches[b]['error'] for b in (0, 1)], [0, 1])
        self.assertEqual(integrated['sampled_records'], [])
        self.assertTrue(0 < integrated['error'] < 1)

    def test_annotations_manual_faults_and_source_snapshot(self):
        text = ('X_ERROR(1/5) 0\nM(1/7) 0\nOBSERVABLE_INCLUDE(4) rec[-1]\n'
                'H 1\nM 1\nOBSERVABLE_INCLUDE(4) rec[-1]\nOBSERVABLE_INCLUDE(9)\n'
                'OBSERVABLE_INCLUDE(0) rec[-2] rec[-2]\n')
        original = parse_circuit(text)
        task = {'outputs': [], 'error_observables': [0, 4, 9, 4]}
        native = compile_circuit(original, task=task)
        rewritten, outputs = quantum_readout(text, [[0, 0], [0, 1], [], [0, 1]])
        explicit = compile_circuit(parse_circuit(rewritten),
            task={'outputs': outputs, 'ideal_state': ZERO_TARGET})
        self.assertEqual(original.to_text(), text)
        self.assertEqual(original.num_qubits, 2)
        self.assertEqual(native.circuit.instructions[:len(original.instructions)], original.instructions)
        self.assertEqual(native.task['raw'], task)
        self.assertNotIn('error_observables', native.task.keys() - {'raw'})
        self.assertEqual(native.circuit.num_records, original.num_records)
        self.assertEqual(native.noise_schedule.actions, explicit.noise_schedule.actions)
        for noise in (NoiseDraw({}, ()), NoiseDraw({'op0': ({'kind': 'X', 'qubits': [0]},)}, ('m1',))):
            self.assertEqual(native.sample(Random(5), noise=noise)['exact'],
                             explicit.sample(Random(5), noise=noise)['exact'])
        for method in ('affine_support', 'exact_marginal'):
            a = compile_circuit(original, task=dict(task, sample_records=[0]), options={'history_sampler': method})
            b = compile_circuit(parse_circuit(rewritten), task={'outputs': outputs,
                'ideal_state': ZERO_TARGET, 'sample_records': [0]}, options={'history_sampler': method})
            for seed in range(8):
                x, y = a.sample(Random(seed)), b.sample(Random(seed))
                self.assertEqual(x['sampled_records'], y['sampled_records'])
                self.assertEqual(x['exact'], y['exact'])

    def test_public_run_enumeration_and_task_file(self):
        text = 'X_ERROR(1/5) 0\nM(1/7) 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n'
        rewritten, outputs = quantum_readout(text, [[0]])
        reference_task = {'outputs': outputs, 'ideal_state': ZERO_TARGET}
        expected = enumerate_errors(parse_circuit(rewritten), task=reference_task, max_weight=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'circuit.magna').write_text(text)
            (root/'task.json').write_text(json.dumps({'error_observables': [0]}))
            actual = enumerate_errors(root/'circuit.magna', task=root/'task.json', max_weight=2)
            self.assertEqual(actual['totals'], expected['totals'])
            a = run(root/'circuit.magna', task=root/'task.json', shots=40, seed=8)
            b = run(parse_circuit(rewritten), task=reference_task, shots=40, seed=8)
            self.assertEqual(a['status'], 'complete')
            self.assertEqual(a['estimates'], b['estimates'])
            # Reports describe the circuit as written; the internal extension stays internal.
            evaluation = dict(ideal_state=None, recovery=None, output_mapping=[],
                              error_metric='error_observables [0]: wrong when any listed OBSERVABLE_INCLUDE parity is one',
                              observables='physical output')
            for result in (a, actual):
                self.assertEqual(result['schema_version'], SCHEMA_VERSION)
                self.assertEqual(result['evaluation'], evaluation)
                self.assertEqual(result['fingerprints']['circuit'], fingerprint(text))
            self.assertIsNone(a['fingerprints']['ideal_state'])
            inspected = compile_circuit(root/'circuit.magna', task=root/'task.json').inspect()
            self.assertEqual((inspected['num_qubits'], len(inspected['instructions'])), (1, 3))
            self.assertEqual(inspected['fingerprint'], parse_circuit(text).fingerprint)
            self.assertEqual(inspected['evaluation'], evaluation)
            output = StringIO()
            with redirect_stdout(output):
                code = main(['validate', str(root/'circuit.magna'), '--task', str(root/'task.json'), '--json'])
            validated = json.loads(output.getvalue())
            self.assertEqual((code, validated['schema_version'], validated['num_qubits']), (0, SCHEMA_VERSION, 1))
            self.assertNotIn('<record-success>', json.dumps(validated))
        self.assertAlmostEqual(actual['totals']['error'], 2/7)
        zero = run(parse_circuit('M 0\n'+detector_selection(1, [[]], [1])), task={'error_observables': []}, shots=2)
        self.assertEqual(zero['estimates']['error'], 0)
        self.assertIsNone(zero['estimates']['conditional_error'])

    def test_entry_validation(self):
        circuit = parse_circuit('M 0\nOBSERVABLE_INCLUDE(0) rec[-1]')
        for removed in ({'error_checks': [[-1]]}, {'error_checks': [[1]]},
                        {'error_checks': [[True]]}, {'error_checks': [['0']]},
                        {'error_checks': []}, {'decoded_error': {}}):
            field = next(iter(removed))
            with self.subTest(removed=removed), \
                    self.assertRaisesRegex(ValueError, f"task field '{field}' is no longer supported"):
                compile_circuit(circuit, task=removed)
        # Record parities are declared in the circuit, so their record references are checked there.
        for text in ('M 0\nOBSERVABLE_INCLUDE(0) rec[-2]', 'M 0\nOBSERVABLE_INCLUDE(0) rec[0]'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'does not name an earlier measurement'):
                parse_circuit(text)
        invalid = ({'error_observables': [1]}, {'error_observables': [-1]},
                   {'error_observables': [0.5]}, {'error_observables': [True]},
                   {'error_observables': ['0']}, {'error_observables': [0], 'typo': 1},
                   {'error_observables': [0], 'outputs': None}, {'error_observables': [0], 'observables': False})
        for task in invalid:
            with self.subTest(task=task), self.assertRaises((ValueError, TypeError)):
                compile_circuit(circuit, task=task)
        for conflict in ({'outputs': [0]}, {'observables': [[0, 1]]},
                         {'ideal_state': ZERO_TARGET}, {'recovery': ZERO_TARGET}):
            with self.subTest(conflict=conflict), \
                    self.assertRaisesRegex(ValueError, 'error_observables asks a classical question'):
                compile_circuit(circuit, task=dict(error_observables=[], **conflict))
