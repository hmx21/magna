"""Exact small physical instruments for the default capped history sampler."""
from fractions import Fraction
from itertools import product
from random import Random
import math
import unittest

from magna import ControllerRegistry, compile_circuit, parse_circuit, run, enumerate_errors
from magna.file_api import NoiseDraw
from tests.support import file_hooks as hooks


def scalar(raw):
    if 'numerator' in raw: return Fraction(raw['numerator'], raw['denominator'])
    a, b, c, d = raw['coefficients']
    assert b == c == d == 0
    return Fraction(a) * Fraction(2)**raw['power']


class CappedAffineTests(unittest.TestCase):
    def test_complete_support_weights_and_encoded_target(self):
        registry = ControllerRegistry()
        seen = []
        def echo(bits, config):
            seen.append(tuple(bits))
            return bits
        registry.register('echo', echo, version='1')
        circuit = parse_circuit('H 0 1\nM(1/5) 0\nDETECTOR rec[-1]\n'
            'CONTROL[echo] det[-1] -> ctrl[0]\nCS ctrl[0] 1\n'
            'MX(1/7) 1\nZ 1\nMX 1\nDETECTOR rec[-1]\n'
            'CONTROL[echo] det[-1] -> ctrl[1]\nCX ctrl[1] 2\nCX 2 3\n'
            'X_ERROR(1/3) 0\nM 0\nDETECTOR rec[-4] rec[-3]\nPOSTSELECT det[2]', controllers=registry)
        anchor = next(i.op_id for i in circuit.instructions if i.name == 'X_ERROR')
        task = {'outputs': [2, 3], 'sample_records': [0, 1, 2, 3]}
        target = dict(circuit=parse_circuit('I 0 1\nH 2 3'), outputs=[0, 1], postselection={2: 1})
        compiled = compile_circuit(circuit, task=task, ideal_state=target,
                                   options={'history_sampler': 'affine_support'})
        support_report = next(r for rows in compiled._execution.state()['query_families'].values()
                              for r in rows if r['purpose'] == 'sampling_capped')
        self.assertEqual(support_report['retained_labels'], 0)
        self.assertGreater(support_report['transport_labels'], 0)
        self.assertEqual(compiled._execution.state()['persistent_networks'], 3)
        total_q = acceptance = success = Fraction(0)
        for flip0, flip1, fault in product((0, 1), repeat=3):
            flips = tuple(f'm{j}' for j, f in enumerate((flip0, flip1)) if f)
            noise = NoiseDraw({anchor: ({'kind': 'X', 'qubits': [0]},)} if fault else {}, flips, (3, 2))
            # Independent measurement oracle. S|+> has fair X outcomes;
            # |+> has outcome zero. Z flips the next true X result.
            oracle = {}
            for true0 in (0, 1):
                a = true0 ^ flip0
                for true1 in range(2 if a else 1):
                    b, c, d = true1 ^ flip1, true1 ^ 1, true0 ^ fault
                    oracle[a, b, c, d] = Fraction(1, 4 if a else 2)
            observed = {}
            for seed in range(64):
                seen.clear()
                draw = compiled.sample(Random(seed), noise=noise)
                word = tuple(draw['sampled_records'])
                self.assertIn(word, oracle)
                self.assertEqual(seen, [(word[0],), (word[2],)])
                p = oracle[word]
                self.assertEqual(scalar(draw['exact']['history_mass']), p)
                self.assertEqual(scalar(draw['exact']['proposal_probability']), p)
                self.assertEqual(scalar(draw['exact']['target_normalization']), Fraction(1, 2))
                accepted = int(word[0] == word[1])
                self.assertEqual(draw['acceptance'], 1.5*accepted)
                self.assertEqual(draw['error'], 1.5*accepted*word[2])
                self.assertEqual(math.ldexp(*draw['weight_scientific']), 1.5)
                observed[word] = draw
            self.assertEqual(set(observed), set(oracle))
            self.assertEqual(sum(scalar(d['exact']['proposal_probability']) for d in observed.values()), 1)
            noise_p = Fraction(1 if flip0 else 4, 5)*Fraction(1 if flip1 else 6, 7)*Fraction(1 if fault else 2, 3)
            total_q += noise_p*sum(oracle.values())
            acceptance += noise_p*sum(p*(w[0] == w[1]) for w, p in oracle.items())
            success += noise_p*sum(p*(w[0] == w[1])*(1-w[2]) for w, p in oracle.items())
        self.assertEqual(total_q, 1)
        self.assertEqual(acceptance, Fraction(19, 28))
        self.assertEqual(success, Fraction(3, 14))
        state = compiled._execution.state()
        self.assertEqual(state['reference_network_evaluations'].get('sampling', 0), 0)
        self.assertEqual(state['reference_network_evaluations']['sampling_capped'], 512)
        self.assertGreater(state['support_updates'], 0)
        self.assertEqual(state['reference_network_resource_constructions'], 0)
        self.assertEqual(state['reference_network_coordinate_constructions'], 0)

    def test_biased_t_full_support(self):
        compiled = compile_circuit(parse_circuit('H 0\nT 0\nMX 0\nMX 0'),
            task={'outputs': [0], 'sample_records': [0, 1]},
            ideal_state=dict(circuit=parse_circuit('H 0 1'), outputs=[0], postselection={1:1}),
            options={'history_sampler': 'affine_support'})
        found = {}
        for seed in range(32):
            draw = compiled.sample(Random(seed));word = tuple(draw['sampled_records'])
            self.assertEqual(word[0], word[1])
            self.assertEqual(scalar(draw['exact']['proposal_probability']), Fraction(1, 2))
            expected = 1+(1-2*word[0])/math.sqrt(2)
            self.assertAlmostEqual(draw['acceptance'], expected)
            self.assertAlmostEqual(draw['error'], word[0]*expected)
            self.assertAlmostEqual(math.ldexp(*draw['weight_scientific']), expected)
            found[word] = draw['acceptance']
        self.assertEqual(set(found), {(0, 0), (1, 1)})
        self.assertAlmostEqual(sum(found.values())/2, 1)
        state = compiled._execution.state()
        self.assertEqual(state['support_method'], 'affine_envelope')
        self.assertEqual(state['reference_network_evaluations'].get('sampling', 0), 0)
        self.assertEqual(state['support_tensor_expansions'], 32)

    def test_general_transport_reuse_respects_actual_feedback(self):
        registry = ControllerRegistry()
        seen = []
        def echo(bits, config):
            seen.append(tuple(bits))
            return bits
        registry.register('echo', echo, version='1')
        circuit = parse_circuit('H 0\nT 0\nMX 0\nDETECTOR rec[-1]\n'
            'CONTROL[echo] det[-1] -> ctrl[0]\nCZ ctrl[0] 0\nMX 0', controllers=registry)
        compiled = compile_circuit(circuit, task={'sample_records':[0,1]},
                                   options={'history_sampler':'affine_support'})
        self.assertEqual(compiled._execution.state()['support_method'],'affine_envelope')
        found = set()
        for seed in range(32):
            before = compiled._execution.state()['support_tensor_expansions']
            seen.clear();draw = compiled.sample(Random(seed));a,b = draw['sampled_records']
            self.assertEqual(seen,[(a,)])
            self.assertEqual(b,0)  # Z feedback flips the negative X branch to positive.
            self.assertEqual(scalar(draw['exact']['proposal_probability']),Fraction(1,2))
            self.assertAlmostEqual(math.ldexp(*draw['weight_scientific']),1+(1-2*a)/math.sqrt(2))
            after = compiled._execution.state()['support_tensor_expansions']
            self.assertEqual(after-before,1+a)
            found.add(a)
        self.assertEqual(found,{0,1})

    def test_nonaffine_support_keeps_interference_zeros(self):
        triples = [(0, 1, 2), (0, 1, 3), (0, 2, 3)]
        amplitudes = [sum((-1)**(sum(((x>>i)&1)*((x>>j)&1)*((x>>k)&1)
            for i,j,k in triples)+(x&y).bit_count()) for x in range(16)) for y in range(16)]
        oracle = {y: Fraction(a*a, 256) for y,a in enumerate(amplitudes) if a}
        self.assertEqual(len(oracle), 10)
        source = 'H 0 1 2 3\n'+''.join(f'CCZ {i} {j} {k}\n' for i,j,k in triples)+'H 0 1 2 3\nM 0 1 2 3'
        compiled = compile_circuit(parse_circuit(source), task={'sample_records': [0,1,2,3]},
                                   options={'history_sampler': 'affine_support'})
        observed = {}
        for seed in range(128):
            d = compiled.sample(Random(seed));word=sum(b<<j for j,b in enumerate(d['sampled_records']))
            self.assertEqual(scalar(d['exact']['history_mass']), oracle.get(word, 0))
            self.assertEqual(scalar(d['exact']['proposal_probability']), Fraction(1,16))
            self.assertEqual(d['acceptance'], float(16*oracle.get(word, 0)))
            observed[word]=d
        self.assertEqual(set(observed), set(range(16)))
        self.assertEqual(sum(d['acceptance'] == 0 for d in observed.values()), 6)
        self.assertEqual(sum(oracle.values()), 1)
        self.assertAlmostEqual(sum(d['acceptance']/16 for d in observed.values()),1)
        self.assertEqual(compiled._execution.state()['reference_network_evaluations']['sampling_capped'],128)
        registry=ControllerRegistry();registry.register('echo',lambda bits,config:bits,version='1')
        adaptive=compile_circuit(parse_circuit(source+'\nDETECTOR rec[-3]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 4'
                                               '\nDETECTOR rec[-4]\nPOSTSELECT det[1]',controllers=registry),
            task={'outputs':[4], 'sample_records':[0,1,2,3]},
            ideal_state=parse_circuit('I 0'), options={'history_sampler':'affine_support'})
        found={}
        for seed in range(128):
            d=adaptive.sample(Random(seed),noise=NoiseDraw({},('m1',)))
            word=sum(b<<j for j,b in enumerate(d['sampled_records'])); true=word^2
            branch=(word>>1)&1
            # The exact injection decomposition certifies a smaller affine
            # continuation after reported m1=0: m2=m3, with m0 still free.
            # The other sector retains eight proposals, including two zeros.
            q=Fraction(1,16) if word&2 else Fraction(1,8)
            self.assertEqual(scalar(d['exact']['proposal_probability']),q)
            self.assertEqual(scalar(d['exact']['history_mass']),oracle.get(true,0))
            self.assertEqual(d['acceptance'],float(oracle.get(true,0)/q)*(1-(word&1)))
            self.assertEqual(d['error'],d['acceptance']*branch)
            found[word]=q
        self.assertEqual(set(found), {w for w in range(16) if w&2 or ((w>>2)&1)==((w>>3)&1)})
        self.assertTrue({w^2 for w in oracle}.issubset(found))
        self.assertEqual(sum(oracle.get(w^2, 0)==0 for w in found), 2)
        self.assertEqual(sum(found.values()),1)
        state = adaptive._execution.state()
        self.assertEqual(state['support_method'], 'affine_envelope')
        self.assertEqual(state['reference_network_resource_constructions'], 0)
        self.assertEqual(state['reference_network_coordinate_constructions'], 0)

    def test_large_affine_envelope_without_record_enumeration(self):
        # A nonuniform magic measurement plus 80 free bits and a correlated copy.
        # The support has 2^81 words; only the mother's terms may be traversed.
        source = 'H '+ ' '.join(map(str, range(81)))+'\nT 0\nMX 0\nMX 0\nM '+ ' '.join(map(str,range(1,81)))
        compiled = compile_circuit(parse_circuit(source),task={'sample_records':list(range(82))},
                                   options={'history_sampler':'affine_support'})
        for draw in hooks.samples(compiled, Random(15), 8):
            word=draw['sampled_records']
            self.assertEqual(word[0],word[1])
            self.assertEqual(scalar(draw['exact']['proposal_probability']),Fraction(1,2**81))
            self.assertAlmostEqual(draw['acceptance'],1+(1-2*word[0])/math.sqrt(2))
        state=compiled._execution.state()
        self.assertEqual(state['support_method'],'affine_envelope')
        bound=next(row['direct_term_bound'] for rows in state['query_families'].values()
            for row in rows if row['purpose']=='sampling_capped')
        self.assertLessEqual(state['support_term_visits'],8*bound)

    def test_nonuniform_dense_feedback_and_guarded_noise(self):
        # Independent dense instrument sums all true/report/noise branches.
        from test_conditional_noise import dense_fixture
        for mode, feedback in product(('affine', 'external', 'mixed'), ('CX', 'CS')):
            with self.subTest(mode=mode, feedback=feedback):
                circuit, task, target, expected = dense_fixture(mode, feedback)
                exact = enumerate_errors(circuit, task=task, ideal_state=target,
                    max_weight=7, options={'history_sampler': 'affine_support'})
                self.assertEqual(exact['status'], 'complete', exact)
                self.assertAlmostEqual(exact['totals']['covered_probability'], expected[0], places=12)
                self.assertAlmostEqual(exact['totals']['acceptance'], expected[1], places=12)
                self.assertAlmostEqual(exact['totals']['error'], expected[2], places=12)
                compiled = compile_circuit(circuit, task=task, ideal_state=target,
                    options={'history_sampler': 'affine_support'})
                draws = list(hooks.samples(compiled, Random(291), 1024))
                self.assertAlmostEqual(sum(d['acceptance'] for d in draws)/len(draws), expected[1], delta=.065)
                self.assertAlmostEqual(sum(d['error'] for d in draws)/len(draws), expected[2], delta=.065)
                state = compiled._execution.state()
                self.assertEqual(state['persistent_networks'], 3)
                self.assertEqual(state['reference_network_resource_constructions'], 0)
                self.assertEqual(state['reference_network_coordinate_constructions'], 0)
                self.assertNotIn('sampling', state['reference_network_evaluations'])

    def test_reconfigure_prepares_optional_route(self):
        circuit = parse_circuit('H 0\nM 0')
        compiled = compile_circuit(circuit, task={'sample_records': [0]}, options={'history_sampler': 'exact_marginal'})
        compiled.options['history_sampler'] = 'affine_support'
        draw = compiled.sample(Random(1))
        self.assertEqual(scalar(draw['exact']['proposal_probability']), Fraction(1, 2))
        self.assertEqual(draw['acceptance'], 1)

    def test_terminal_recovery_controller_and_target(self):
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        circuit = parse_circuit('H 0')
        recovery = parse_circuit('M 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0', controllers=registry)
        compiled = compile_circuit(circuit, recovery=recovery, ideal_state=parse_circuit('I 0'),
                                   options={'history_sampler': 'affine_support'})
        for draw in hooks.samples(compiled, Random(9), 16):
            self.assertEqual(draw['acceptance'], 1)
            self.assertEqual(draw['error'], 0)
            self.assertEqual(draw['recovery_decoder_calls'], 1)
            self.assertEqual(scalar(draw['exact']['proposal_probability']), Fraction(1, 2))

    def test_conditional_noise_and_pauli_feedback(self):
        registry = ControllerRegistry()
        registry.register('echo', lambda bits, config: bits, version='1')
        circuit = parse_circuit('H 0 1\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\n'
            'CS ctrl[0] 1\nMX 1\nX_ERROR(1/3) rec[-1] 2\nM 2\nCX rec[-1] 3', controllers=registry)
        options = {'history_sampler': 'affine_support'}
        result = run(circuit, task={'outputs': [3], 'sample_records': [0, 1, 2]},
                     ideal_state=parse_circuit('I 0'), options=options, shots=512, seed=100)
        self.assertEqual(result['status'], 'complete')
        self.assertAlmostEqual(result['estimates']['error'], 1/12, delta=.04)
        self.assertEqual(result['estimates']['acceptance'], 1)


if __name__ == '__main__':
    unittest.main()
