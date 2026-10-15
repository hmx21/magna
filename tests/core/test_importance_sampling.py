"""Exact finite importance identities and public weighted-estimator contracts."""
from tests.support.readout_target import ZERO_TARGET
from fractions import Fraction
from itertools import product
from random import Random
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import math
import statistics
import tempfile
import unittest

from magna import _native, compile_circuit, parse_circuit, run, ControllerRegistry
from magna.file_api import NoiseDraw, _NoiseSchedule, main
from tests.support import file_hooks as hooks
from tests.support import native_testing


class ImportanceSamplingTests(unittest.TestCase):
    def test_native_draws_return_exact_likelihood_products(self):
        categories = (((1, 2),), ((1, 4), (1, 4)))
        factors = (((9, 5), (1, 5)), ((8, 5), (2, 5)))
        for seed in range(64):
            selected, ratio = native_testing.load().sample_file_categories(categories, seed, factors)
            expected = Fraction(1)
            for choice, pair in zip(selected, factors): expected *= Fraction(*pair[choice >= 0])
            self.assertEqual(Fraction(*ratio), expected)
            self.assertEqual(selected, native_testing.load().sample_file_categories(categories, seed))
        self.assertEqual(native_testing.load().sample_file_categories((), 1, ()), ((), (1, 1)))
        with self.assertRaises(ValueError): native_testing.load().sample_file_categories(categories, 0, ())

    def test_channel_proportions_correlations_endpoints_and_guard_policy(self):
        circuit = parse_circuit('PAULI_CHANNEL_1(1/20,1/10,0) 0\n'
            'CORRELATED_ERROR(1/10) X0\nELSE_CORRELATED_ERROR(2/9) Z0\n'
            'M(1/100) 0\nX_ERROR(1/5) rec[-1] 0\nX_ERROR(0) 0\nZ_ERROR(1) 0')
        physical = _NoiseSchedule(circuit)
        proposal = _NoiseSchedule(circuit, {'method': 'importance', 'proposal_scale': '2'})
        for index, (praw, qraw, factors) in enumerate(zip(physical.probabilities, proposal.probabilities, hooks.weight_factors(proposal))):
            ps = [Fraction(*v) for v in praw]; qs = [Fraction(*v) for v in qraw]
            self.assertEqual(len(ps), len(qs))
            # Exhaustive one-category identity, including the residual no-error outcome.
            for p, q, factor in zip([1-sum(ps)]+ps, [1-sum(qs)]+qs,
                                    [Fraction(*factors[0])]+[Fraction(*factors[1])]*len(ps)):
                self.assertEqual(q*factor, p)
            if index < 3: self.assertEqual(qs, [2*p for p in ps])
            else: self.assertEqual(ps, qs)
        self.assertEqual([Fraction(*v) for v in proposal.probabilities[0]], [Fraction(1,10),Fraction(1,5),0])
        self.assertEqual([Fraction(*v) for v in proposal.probabilities[1]], [Fraction(1,5),Fraction(2,5)])
        self.assertEqual([Fraction(*v) for v in proposal.probabilities[2]], [Fraction(1,50)])
        self.assertEqual(proposal.to_dict()['proposal_scale'], '2')
        self.assertEqual(proposal.to_dict()['tilted_categories'], 3)
        self.assertEqual(proposal.to_dict()['guarded_categories_unchanged'], 1)

    def test_all_fault_configurations_recover_exact_physical_acceptance_and_error(self):
        p0, p1 = Fraction(1, 10), Fraction(1, 5)
        circuit = parse_circuit('X_ERROR(1/10) 0\nM 0\nX_ERROR(1/5) 1\nDETECTOR rec[-1]\nPOSTSELECT det[0]')
        noise_ids = [i.op_id for i in circuit.instructions if i.name == 'X_ERROR']
        for scale in (Fraction(1, 2), Fraction(1), Fraction(2)):
            with self.subTest(scale=scale):
                compiled = compile_circuit(circuit, task={'outputs': [1]},
                    ideal_state=parse_circuit('I 0'), sampling={'method': 'importance', 'proposal_scale': str(scale)})
                a = e = 0.; norm = Fraction(0)
                for bits in product((0, 1), repeat=2):
                    weight = proposal_mass = Fraction(1); gates = {}
                    for index, (bit, p) in enumerate(zip(bits, (p0, p1))):
                        q = scale*p
                        weight *= p/q if bit else (1-p)/(1-q)
                        proposal_mass *= q if bit else 1-q
                        if bit: gates[noise_ids[index]] = (dict(kind='X', qubits=[index]),)
                    draw = compiled.sample(Random(2), noise=NoiseDraw(gates, (), (weight.numerator, weight.denominator)))
                    a += float(proposal_mass)*draw['acceptance']
                    e += float(proposal_mass)*draw['error']
                    norm += proposal_mass*weight
                self.assertEqual(norm, 1)
                self.assertAlmostEqual(a, float(1-p0))
                self.assertAlmostEqual(e, float((1-p0)*p1))
                self.assertAlmostEqual(e/a, float(p1))

    def test_weighted_summary_uses_joint_means_and_paired_ratio_errors(self):
        weights = [2, Fraction(1, 2), Fraction(1, 3), 1]
        acceptance = [1, Fraction(1, 2), 0, 1]
        error = [Fraction(1, 2), 0, 0, Fraction(1, 4)]
        draws = []
        for w,a,e in zip(weights, acceptance, error):
            values = [float(w*a), float(w*e), float(w*(a-2*e))]
            draws.append(dict(scientific=[math.frexp(v) for v in values], exact={}))
        result = hooks.summary(draws)
        av = [float(w*a) for w,a in zip(weights,acceptance)]
        ev = [float(w*e) for w,e in zip(weights,error)]
        ratio = sum(ev)/sum(av)
        self.assertAlmostEqual(result['estimates']['acceptance'], statistics.mean(av))
        self.assertAlmostEqual(result['estimates']['error'], statistics.mean(ev))
        self.assertAlmostEqual(result['estimates']['conditional_error'], ratio)
        self.assertAlmostEqual(result['standard_errors']['conditional_error'],
            statistics.stdev([e-ratio*a for a,e in zip(av,ev)])/2/statistics.mean(av))
        ws = native_testing.load().summarize_file_weights(tuple(math.frexp(float(w)) for w in weights))
        self.assertAlmostEqual(ws['ess'], float(sum(weights)**2/sum(w*w for w in weights)))

    def test_huge_and_tiny_weights_remain_range_safe(self):
        one = ((1,0,0,0),0)
        for weight, sign in (((1 << 2000, 1), 1), ((1, 1 << 2000), -1)):
            values = native_testing.load().file_scalar_ratios(((one,one),), weight)
            self.assertEqual(values, ((.5, sign*2000+1),))
            result = hooks.summary([dict(scientific=[values[0],values[0]],exact={})]*2)
            self.assertIsNone(result['estimates']['acceptance'])
            self.assertEqual(result['estimates']['conditional_error'], 1)
            field = 'overflow_fields' if sign > 0 else 'underflow_fields'
            self.assertIn('estimates.acceptance',result['numerics'][field])
            self.assertEqual(native_testing.load().summarize_file_weights(values*2)['ess'],2)
        with self.assertRaises(ValueError): native_testing.load().file_scalar_ratios(((one,one),), (-1,1))

    def test_reproducibility_and_physical_mode_unchanged(self):
        circuit = parse_circuit('X_ERROR(1/10) 0\nY_ERROR(1/5) 0\nM(1/20) 0\nCX rec[-1] 1')
        task = {'outputs': [1], 'ideal_state': ZERO_TARGET}
        a = run(circuit,task=task,shots=32,seed=4)
        b = run(circuit,task=task,shots=32,seed=4,sampling={'method':'physical'})
        c = run(circuit,task=task,shots=32,seed=4,sampling={'method':'importance','proposal_scale':'1'})
        self.assertEqual(a['estimates'],b['estimates'])
        self.assertEqual(a['estimates'],c['estimates'])
        self.assertEqual(c['weights']['ess'],32)
        self.assertEqual(c['sampling']['tilted_categories'],0)
        physical = _NoiseSchedule(circuit)
        unit_scale = _NoiseSchedule(circuit, {'method':'importance','proposal_scale':'1'})
        left, right = Random(51), Random(51)
        self.assertEqual(hooks.draw_many(physical, left, 64), hooks.draw_many(unit_scale, right, 64))
        self.assertEqual(left.getstate(), right.getstate())
        kwargs = dict(task=task,shots=32,seed=4,sampling={'method':'importance','proposal_scale':'2'})
        d = run(circuit,**kwargs); e = run(circuit,**kwargs)
        self.assertEqual(d['estimates'],e['estimates'])
        self.assertNotEqual(a['fingerprints']['sampling'],d['fingerprints']['sampling'])

    def test_complex_terminal_recovery_keeps_true_born_sampling(self):
        registry=ControllerRegistry(); seen=[]
        def echo(bits,config): seen.append(bits[0]); return bits
        registry.register('echo',echo,version='1')
        recovery=parse_circuit('H 1\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0',controllers=registry)
        compiled=compile_circuit(parse_circuit('X_ERROR(1/10) 0'),ideal_state=parse_circuit('I 0'),
            recovery=recovery,sampling={'method':'importance','proposal_scale':'5'})
        noise_id=compiled.circuit.instructions[0].op_id
        branches=[]
        for seed in range(12):
            draw=compiled.sample(Random(seed),noise=NoiseDraw({noise_id:(dict(kind='X',qubits=[0]),)},(),(1,5)))
            branches.append(seen[-1])
            self.assertAlmostEqual(draw['acceptance'],.2)
            self.assertAlmostEqual(draw['error'],.2*(1-seen[-1]))
            self.assertEqual(draw['recovery_decoder_calls'],1)
            self.assertEqual(draw['weight_scientific'],math.frexp(.2))
        self.assertEqual(set(branches),{0,1})

    def test_invalid_proposals_and_cli_config(self):
        physical=parse_circuit('I 0')
        for settings in ({'method':'bad'}, {'method':'physical','proposal_scale':'2'},
                         {'method':'importance','proposal_rate':'1/2'},
                         {'method':'importance'}, *({'method':'importance','proposal_scale':x}
                         for x in (0,-1,'nan','inf',True,'1/0'))):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                compile_circuit(physical,sampling=settings)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'c.magna').write_text('X_ERROR(1/1000) 0\nM 0')
            (root/'task.json').write_text(json.dumps({'outputs': [0], 'ideal_state': ZERO_TARGET}))
            (root/'sampling.json').write_text('{"method":"importance","proposal_scale":"100"}')
            for command in ('validate','inspect','run'):
                out=StringIO()
                with redirect_stdout(out):
                    code=main([command,str(root/'c.magna'),'--task',str(root/'task.json'),
                               '--sampling',str(root/'sampling.json'),'--json',
                               *(['--shots','16'] if command=='run' else [])])
                self.assertEqual(code,0,out.getvalue())
                self.assertEqual(json.loads(out.getvalue())['sampling']['method'],'importance')
                self.assertEqual(json.loads(out.getvalue())['sampling']['proposal_scale'],'100')

    def test_scaled_channel_totals_preserve_full_support_or_fail(self):
        for source, boundary in (
                ('X_ERROR(1/5) 0', Fraction(5)),
                ('PAULI_CHANNEL_1(1/10,1/10,0) 0', Fraction(5)),
                ('CORRELATED_ERROR(1/10) X0\nELSE_CORRELATED_ERROR(2/9) Z0', Fraction(10, 3)),
                ('M(1/4) 0', Fraction(4))):
            for scale in (boundary, boundary+1):
                with self.subTest(source=source, scale=scale), self.assertRaises(ValueError):
                    compile_circuit(parse_circuit(source),
                        sampling={'method':'importance','proposal_scale':str(scale)})
        # Deterministic and guarded categories are never inflated, even when
        # their nominal rate times the scale exceeds one.
        circuit = parse_circuit('M 0\nX_ERROR(1/3) rec[-1] 0\nX_ERROR(0) 0\nZ_ERROR(1) 0')
        physical = _NoiseSchedule(circuit)
        scaled = _NoiseSchedule(circuit, {'method':'importance','proposal_scale':'100'})
        self.assertEqual(scaled.probabilities, physical.probabilities)
        self.assertEqual(scaled.tilted, 0)


if __name__ == '__main__': unittest.main()
