"""Public conveniences: compile once and run many times, CLI summaries and actionable errors."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from magna import compile_circuit, parse_circuit, run
from magna.file_api import PACKAGE_VERSION, main

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT/'tests/fixtures/importance-sampling'


class CompiledRunTests(unittest.TestCase):
    def test_compiled_runs_match_fresh_runs_and_report_their_own_work(self):
        circuit = parse_circuit('H 0\nT 0\nDEPOLARIZE1(1/20) 0\nM(1/50) 0\nCX rec[-1] 1\nH 1\nT 1')
        kwargs = dict(task={'outputs': [1]}, ideal_state=parse_circuit('H 0\nT 0'))
        fresh = run(circuit, shots=64, seed=11, **kwargs)
        compiled = compile_circuit(circuit, **kwargs)
        first = run(compiled, shots=64, seed=11)
        second = run(compiled, shots=64, seed=11)
        for result in (first, second):
            self.assertEqual(result['status'], 'complete')
            for key in ('estimates', 'standard_errors', 'estimates_scientific', 'fingerprints', 'route'):
                self.assertEqual(result[key], fresh[key], key)
            # Each call reports only the queries it made, not the circuit's history.
            self.assertEqual(result['resources']['queries'], first['resources']['queries'])
        self.assertEqual(first['timings']['preparation_seconds'], second['timings']['preparation_seconds'])
        self.assertLessEqual(first['resources']['queries'], fresh['resources']['queries'])

    def test_compiled_run_rejects_settings_it_already_fixes(self):
        compiled = compile_circuit(parse_circuit('X_ERROR(1/10) 0'), ideal_state=parse_circuit('I 0'))
        with self.assertRaisesRegex(ValueError, 'already fixes its task, sampling; pass them to compile_circuit'):
            run(compiled, task={}, sampling={'method': 'physical'})

    def test_budgets_count_from_the_start_of_each_run(self):
        from time import sleep
        from magna import ControllerRegistry
        registry = ControllerRegistry()
        registry.register('majority', lambda bits, config: (int(sum(bits) >= 2),), version='1')
        circuit = parse_circuit('H 0 1 2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                                'CONTROL[majority] det[-3] det[-2] det[-1] -> ctrl[0]\nCX ctrl[0] 3', controllers=registry)
        per_run = run(compile_circuit(circuit), shots=5, seed=1)['resources']['queries']
        compiled = compile_circuit(circuit, options={'max_queries': per_run+1})
        for seed in range(4):
            result = run(compiled, shots=5, seed=seed)
            self.assertEqual((result['status'], result['completed']), ('complete', 5), result['failures'])
        timed = compile_circuit(circuit, options={'max_seconds': .5})
        self.assertEqual(run(timed, shots=2)['status'], 'complete')
        sleep(.6)
        self.assertEqual(run(timed, shots=2)['status'], 'complete')

    def test_compiled_circuits_are_not_compiled_again(self):
        from magna import compare_samplers
        compiled = compile_circuit(parse_circuit('X_ERROR(1/10) 0'), ideal_state=parse_circuit('I 0'))
        with self.assertRaisesRegex(TypeError, 'already compiled; pass it to run'):
            compile_circuit(compiled)
        with self.assertRaisesRegex(TypeError, 'pass the circuit or its path, not a compiled circuit'):
            compare_samplers(compiled, pilot_shots=4)

    def test_option_and_task_errors_name_the_key_and_suggest_a_fix(self):
        circuit = parse_circuit('X_ERROR(1/10) 0\nM 0')
        with self.assertRaisesRegex(ValueError, r"unknown options: 'max_term' \(did you mean 'max_terms'\?\)\. Valid: .*max_terms"):
            compile_circuit(circuit, options={'max_term': 4})
        with self.assertRaisesRegex(ValueError, r"unknown task fields: 'output' \(did you mean 'outputs'\?\)"):
            compile_circuit(circuit, task={'output': [0]})
        with self.assertRaisesRegex(ValueError, r'outputs must be distinct qubit indices in 0\.\.0; invalid: \[3\]'):
            compile_circuit(circuit, task={'outputs': [3]})
        with self.assertRaisesRegex(ValueError, r"max_plans must be a positive integer, not 0"):
            compile_circuit(circuit, options={'max_plans': 0})
        with self.assertRaisesRegex(ValueError, r"sampling method must be 'physical' or 'importance', not 'tilted'"):
            compile_circuit(circuit, sampling={'method': 'tilted'})


class CommandLineTests(unittest.TestCase):
    def cli(self, *args):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try: code = main([str(a) for a in args])
            except SystemExit as exc: code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_summary_by_default_and_full_json_on_request(self):
        args = ('run', FIXTURES/'physical.magna', '--task', FIXTURES/'task.json', '--shots', 200, '--seed', 7)
        code, out, _ = self.cli(*args)
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith('run: complete, 200 of 200 draws, seed 7'), out)
        self.assertRegex(out, r'acceptance\s+1 ± 0')
        self.assertIn('conditional error', out)
        code, out, _ = self.cli(*args, '--json')
        self.assertEqual(json.loads(out)['completed'], 200)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'result.json'
            code, out, _ = self.cli(*args, '--output', target)
            self.assertEqual(json.loads(target.read_text())['completed'], 200)
            self.assertIn(f'Full result written to {target}.', out)

    def test_each_command_summarizes_and_accepts_only_its_options(self):
        circuit, task = FIXTURES/'physical.magna', FIXTURES/'task.json'
        code, out, _ = self.cli('validate', circuit, '--task', task)
        self.assertEqual(code, 0); self.assertTrue(out.startswith('valid: '), out)
        code, out, _ = self.cli('inspect', circuit, '--task', task)
        self.assertEqual(code, 0); self.assertIn('prepared queries', out)
        code, out, _ = self.cli('enumerate', circuit, '--task', task, '--max-weight', 1)
        self.assertEqual(code, 0); self.assertRegex(out, r'covered probability\s+1')
        for args in (('validate', circuit, '--shots', 8), ('enumerate', circuit, '--max-weight', 1,
                      '--history-sampler', 'exact_marginal'), ('enumerate', circuit, '--sampling', task),
                     ('run', circuit, '--max-weight', 1), ('enumerate', circuit)):
            with self.subTest(args=args):
                code, _, err = self.cli(*args)
                self.assertEqual(code, 2); self.assertIn('usage: magna', err)

    def test_version_and_command_help(self):
        code, out, _ = self.cli('--version')
        self.assertEqual((code, out.strip()), (0, f'magna {PACKAGE_VERSION}'))
        code, out, _ = self.cli('run', '--help')
        self.assertEqual(code, 0)
        self.assertIn('--shots N', out); self.assertIn('number of draws', out)

    def test_invalid_input_reports_on_stderr_or_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            options = Path(directory)/'options.json'; options.write_text('{"max_term": 4}')
            code, out, err = self.cli('run', FIXTURES/'physical.magna', '--options', options)
            self.assertEqual((code, out), (2, ''))
            self.assertIn("magna run: error: unknown options: 'max_term' (did you mean 'max_terms'?)", err)
            code, out, _ = self.cli('run', FIXTURES/'physical.magna', '--options', options, '--json')
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)['status'], 'invalid')


    def test_unimportable_decoder_binding_is_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            circuit = Path(directory)/'c.magna'
            circuit.write_text('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[dec] det[-1] -> ctrl[0]\nCX ctrl[0] 1\n')
            bindings = Path(directory)/'controllers.json'
            for target in ('no_such_package.module:decode', 'json:no_such_function'):
                bindings.write_text(json.dumps({'bindings': {'dec': {'import': target, 'version': '1'}}}))
                code, out, _ = self.cli('run', circuit, '--controllers', bindings, '--json')
                self.assertEqual((code, json.loads(out)['status']), (2, 'invalid'))
                self.assertIn("controller binding 'dec' cannot import", json.loads(out)['error'])

    def test_budget_failure_is_a_failed_command_not_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            circuit = Path(directory)/'t.magna'; circuit.write_text('H 0\nT 0\nH 0\nT 0\nH 0\nM 0\n')
            options = Path(directory)/'options.json'; options.write_text('{"max_terms": 0}')
            code, out, err = self.cli('inspect', circuit, '--options', options)
            self.assertEqual((code, out), (1, ''))
            self.assertIn('magna inspect: failed: ResourceLimit:', err)
            self.assertIn('(max_terms = 0, needs at least 1)', err)
            code, out, _ = self.cli('inspect', circuit, '--options', options, '--json')
            self.assertEqual((code, json.loads(out)['status']), (1, 'failed'))


class NamespaceTests(unittest.TestCase):
    def test_top_level_is_the_file_interface_and_operations_is_separate(self):
        import magna
        from magna import operations
        self.assertEqual(set(magna.__all__), {
            'load_circuit', 'parse_circuit', 'load_controllers', 'compile_circuit', 'run', 'enumerate_errors',
            'compare_samplers', 'CompiledCircuit', 'CircuitFile', 'ControllerRegistry', 'ResourceLimit',
            'CircuitSyntaxError', 'SourceLocation'})
        for name in magna.__all__:
            self.assertTrue(hasattr(magna, name), name)
        for name in operations.__all__:
            self.assertTrue(hasattr(operations, name), name)
            self.assertFalse(hasattr(magna, name), name)
        self.assertTrue(issubclass(magna.ResourceLimit, RuntimeError))

    def test_first_parameters_have_their_documented_names(self):
        from inspect import signature
        import magna
        from magna.operations import Simulator
        for function in (magna.run, magna.enumerate_errors, magna.compare_samplers):
            self.assertEqual(next(iter(signature(function).parameters)), 'circuit', function.__name__)
        self.assertEqual(list(signature(Simulator).parameters), ['ir', 'options'])
        self.assertEqual(run(circuit=parse_circuit('I 0'), shots=2)['status'], 'complete')


class MissingExtensionTests(unittest.TestCase):
    def test_import_without_native_extension_explains_how_to_build(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)/'magna'
            shutil.copytree(ROOT/'core/magna', package,
                            ignore=shutil.ignore_patterns('_native*', 'native', '__pycache__', '*.json'))
            code = f'import sys; sys.path.insert(0, {directory!r}); import magna'
            process = subprocess.run([sys.executable, '-I', '-c', code],
                                     cwd=directory, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("magna's compiled extension is missing", process.stderr)
        self.assertIn('scripts/build_native_core.py', process.stderr)
        self.assertNotIn('circular import', process.stderr)


if __name__ == '__main__':
    unittest.main()
