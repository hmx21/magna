"""Pauli feedback assignments versus fresh literal whole-instrument queries."""
from itertools import product
from random import Random
import unittest

from magna import ControllerRegistry, compile_circuit, parse_circuit
from magna.operations import NativeBoundaryTrace
from magna.file_api import NoiseDraw
from magna.scalar import ExactScalar
from tests.support import file_hooks as hooks
from tests.support import native_testing


def literal_engine(compiled, values, noise, *, stage='protocol', recovery_values=None):
    """Independent uncached lowering with every reached feedback gate explicit."""
    ir = hooks.lower(compiled.circuit, compiled.task['outputs'], values, noise)
    if stage != 'protocol' and compiled.recovery is not None:
        ir = hooks.attach_recovery(compiled.recovery, ir, recovery_values or {},
            compiled.circuit.num_records, compiled.circuit.num_qubits)
    if stage == 'target':
        ir, _ = hooks.attach_target(compiled.ideal_state, ir, compiled.circuit.num_records+
            (compiled.recovery.circuit.num_records if compiled.recovery else 0))
    return NativeBoundaryTrace(ir)


class PauliFeedbackReuseTests(unittest.TestCase):
    def registry(self):
        registry = ControllerRegistry()
        registry.register('and', lambda bits, config: (bits[0] & bits[1],), version='1')
        registry.register('echo', lambda bits, config: bits, version='1')
        return registry

    def test_all_paulis_mixed_affine_guards_and_original_faults_share_one_plan(self):
        circuit = parse_circuit('H 0 1 2\nT 2\nM 0 1\n'
            'DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\n'
            'DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[echo] det[-2] det[-1] -> ctrl[1] ctrl[2]\n'
            'CONTROL[XOR] !ctrl[0] rec[-1] -> ctrl[3]\n'
            'CX ctrl[3] 2\nY_ERROR(1/3) 2\nT 2\nCY ctrl[1] 2\n'
            'H 2\nCZ 2 ctrl[2]\nM(1/4) 2\nR 2\nCX ctrl[0] 2\nH 2\nT 2',
            controllers=self.registry())
        compiled = compile_circuit(circuit, task={'outputs': [2]})
        terms = tuple(t for inst in circuit.analysis.external_instructions for t in inst.outputs)
        noise_id = next(i.op_id for i in circuit.instructions if i.name == 'Y_ERROR')
        for word in product((0, 1), repeat=3):
            values = dict(zip(terms, word))
            for noisy in (False, True):
                noise = NoiseDraw({noise_id: ({'kind': 'Y', 'qubits': [2]},)} if noisy else {},
                                  ('m0', 'm2') if noisy else ())
                actual = hooks.engine(compiled, values, noise)
                expected = literal_engine(compiled, values, noise)
                a = hooks.pack(compiled, actual, noise); b = hooks.pack(compiled, expected, noise)
                # Compare the entire joint record distribution, independently
                # of its affine origin/basis and the sampler's seeded word.
                names = (('m0',), ('m1',), ('m2',))
                left = actual.query(a, keep_rows=actual.rows(names)).raw
                right = expected.query(b, keep_rows=expected.rows(names)).raw
                difference = [(ExactScalar(*c), k) for c, k in left]
                difference += [(-ExactScalar(*c), k) for c, k in right]
                squared = ExactScalar()
                for c, k in difference:
                    for d, l in difference:
                        squared += c.conjugate()*d*ExactScalar(*native_testing.load().overlap(k, l))
                self.assertFalse(squared)
                for outcomes in product((0, 1), repeat=2):
                    rows = actual.rows((('m0', 'm1'), ('m2',)))
                    reference_rows = expected.rows((('m0', 'm1'), ('m2',)))
                    for x, z in ((0, 0), (1, 0), (0, 1), (1, 1)):
                        self.assertEqual(actual.query(a, event_rows=rows, outcomes=outcomes, x=x, z=z).total(),
                            expected.query(b, event_rows=reference_rows, outcomes=outcomes, x=x, z=z).total())
        self.assertEqual(compiled.compilations, 1)
        self.assertEqual(len(compiled.plans), 1)

    def test_unreached_mixed_feedback_keeps_earlier_causal_marginal(self):
        circuit = parse_circuit('H 0 1\nT 1\nM 0\n'
            'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\nCY ctrl[1] 1\n'
            'H 1\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[2]\nCX ctrl[2] 0',
            controllers=self.registry())
        compiled = compile_circuit(circuit)
        term = circuit.analysis.external_instructions[0].outputs[0]
        for values, row in (({}, ('m0',)), ({term: 0}, ('m1',)), ({term: 1}, ('m1',))):
            for flips in ((), ('m0',)):
                noise = NoiseDraw({}, flips)
                actual = hooks.engine(compiled, values, noise)
                expected = literal_engine(compiled, values, noise)
                for bit in (0, 1):
                    self.assertEqual(actual.query(hooks.pack(compiled, actual, noise),
                        event_rows=actual.rows((row,)), outcomes=(bit,)).total(),
                        expected.query(hooks.pack(compiled, expected, noise),
                        event_rows=expected.rows((row,)), outcomes=(bit,)).total())
        self.assertEqual(compiled.compilations, 1)

    def test_physical_and_recovery_callbacks_match_literal_sampling(self):
        registry = ControllerRegistry(); observed = []
        registry.register('and', lambda bits, config: observed.append(tuple(bits)) or (bits[0] & bits[1],), version='1')
        registry.register('echo', lambda bits, config: observed.append(tuple(bits)) or bits, version='1')
        circuit = parse_circuit('H 0 1 2\nT 2\nM(1/3) 0 1\n'
            'DETECTOR rec[-2]\nDETECTOR rec[-1]\nCONTROL[and] det[-2] det[-1] -> ctrl[0]\nCY ctrl[0] 2\n'
            'X_ERROR(1/3) 2\nH 2\nM 2\n'
            'DETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[1]\nCZ ctrl[1] 2\nH 2\nT 2\n'
            # Accept when physical record 0 is zero.
            'DETECTOR rec[-3]\nPOSTSELECT det[3]', controllers=registry)
        recovery = parse_circuit('H 1\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CONTROL[XOR] ctrl[0] rec[-1] -> ctrl[1]\nCY ctrl[1] 0\n'
            'CZ ctrl[0] 0\nT 0', controllers=registry)
        kwargs = dict(task={'outputs': [2], 'sample_records': [0, 1, 2],
                           'observables': [[1, 0], [0, 1], [1, 1]]},
                      recovery=recovery, ideal_state=parse_circuit('H 0\nT 0'))
        actual = compile_circuit(circuit, **kwargs)
        expected = compile_circuit(circuit, **kwargs)
        from tests.support.file_history import ReferenceHistory
        reference = ReferenceHistory(expected, lambda *a, **kw: literal_engine(expected, *a, **kw))
        for seed in range(24):
            noise = hooks.draw(actual.noise_schedule, Random(seed)); observed.clear()
            draw = actual.sample(Random(seed+100), noise=noise)
            # Reference-sector sampling changes the RNG schedule. Replay the
            # reached physical inputs through an independently lowered circuit.
            words = iter(sum(bit << j for j, bit in enumerate(bits)) for bits in observed)
            exact = reference.sample(Random(0), noise=noise,
                                     _branch_selector=lambda terms, width: next(words))
            fields = [draw['exact'][name] for name in ('history_mass', 'acceptance', 'error')]
            fields += draw['exact']['moments']
            for value, raw in zip(fields, exact):
                self.assertEqual(ExactScalar(tuple(value['coefficients']), value['power']), ExactScalar(*raw))
        self.assertEqual(actual.compilations, 2)

    def test_no_decoder_value_cache_and_clifford_template_reuse(self):
        calls = []
        registry = self.registry()
        registry.register('count', lambda bits, config: calls.append(bits) or bits, version='1')
        circuit = parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[count] det[-1] -> ctrl[0]\n'
            'CX ctrl[0] 1\nH 1\nT 1', controllers=registry)
        compiled = compile_circuit(circuit)
        for seed in range(12): compiled.sample(Random(seed))
        self.assertEqual(len(calls), 12)
        self.assertEqual(set(calls), {(0,), (1,)})
        mixed = parse_circuit(circuit.text+'\nCS ctrl[0] 1\nCH ctrl[0] 1', controllers=registry)
        compiled = compile_circuit(mixed)
        term = mixed.analysis.external_instructions[0].outputs[0]
        for bit in (0, 1):
            values = {term: bit}; noise = NoiseDraw({}, ())
            actual = hooks.engine(compiled, values, noise); expected = literal_engine(compiled, values, noise)
            for x, z in ((1, 0), (0, 1), (1, 1)):
                self.assertEqual(actual.query(actual.pack(), x=x << 1, z=z << 1).total(),
                                 expected.query(expected.pack(), x=x << 1, z=z << 1).total())
        self.assertEqual(compiled.compilations, 1)


if __name__ == '__main__':
    unittest.main()
