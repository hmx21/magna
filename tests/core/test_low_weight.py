"""Exhaustive finite-support checks against analytic probabilities."""
from tests.support.readout_target import ZERO_TARGET
import json
import math
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction
from contextlib import redirect_stdout
from io import StringIO
from magna import parse_circuit, enumerate_errors, ControllerRegistry
from magna.file_api import main
from tests.support import native_testing


def rational(encoded):
    coeff=[int(v,16) for v in encoded['coefficients_hex']]
    assert coeff[1:]==[0,0,0]
    return Fraction(coeff[0],int(encoded['denominator_hex'],16))*Fraction(2)**encoded['exponent2']


class LowWeightTests(unittest.TestCase):
    def evaluate(self,text,weight=1,**kwargs):
        return enumerate_errors(parse_circuit(text),max_weight=weight,ideal_state=parse_circuit('I 0'),**kwargs)

    def test_distinct_sites_and_exact_truncated_mass(self):
        r=self.evaluate('X_ERROR(1/10) 0\nX_ERROR(1/5) 0',include_patterns=True)
        self.assertEqual(r['status'],'complete',r)
        self.assertEqual(r['completed_patterns'],3)
        t=r['totals'];self.assertEqual(rational(t['exact']['acceptance']),Fraction(49,50))
        self.assertEqual(rational(t['exact']['error']),Fraction(13,50))
        self.assertEqual(t['full_distribution_bounds']['error'],[.26,.28])
        full=self.evaluate('X_ERROR(1/10) 0\nX_ERROR(1/5) 0',2)
        self.assertEqual(full['completed_patterns'],4)
        self.assertEqual(rational(full['totals']['exact']['error']),Fraction(13,50))
        self.assertEqual(full['totals']['covered_probability'],1)
        self.assertEqual([p['weight'] for p in r['patterns']].count(1),2)
        self.assertEqual(len(r['category_definitions']),2)
        self.assertEqual(r['category_definitions'][0]['channels'][0]['paulis'],({'kind':'X','qubits':[0]},))
        self.assertEqual(len(r['fingerprints']['circuit']),64)

    def test_mutually_exclusive_paulis_and_correlated_chain(self):
        r=self.evaluate('PAULI_CHANNEL_1(1/10,1/5,3/10) 0')
        self.assertEqual(r['completed_patterns'],4)
        self.assertEqual(rational(r['totals']['exact']['error']),Fraction(3,10))
        r=self.evaluate('CORRELATED_ERROR(1/10) X0\nELSE_CORRELATED_ERROR(2/9) Z0')
        self.assertEqual(r['completed_patterns'],3)
        self.assertEqual(rational(r['totals']['exact']['error']),Fraction(1,10))
        self.assertEqual(r['totals']['covered_probability'],1)

    def test_endpoints_empty_sector_and_zero_noise(self):
        for text in ('I 0','X_ERROR(0) 0'):
            r=self.evaluate(text,0);self.assertEqual(r['completed_patterns'],1)
            self.assertEqual(r['totals']['error'],0)
        r=self.evaluate('X_ERROR(1) 0',0)
        self.assertEqual(r['completed_patterns'],0)
        self.assertEqual(r['totals']['covered_probability'],0)
        self.assertIsNone(r['totals']['conditional_error_within_enumerated_sector'])
        self.assertEqual(r['totals']['full_distribution_bounds']['conditional_error'],[0,1])
        self.assertEqual(self.evaluate('X_ERROR(1) 0')['totals']['error'],1)
        self.assertEqual(self.evaluate('X_ERROR(1) 0',1<<64)['totals']['error'],1)

    def test_report_noise_and_postselection_are_joint_contributions(self):
        r=enumerate_errors(parse_circuit('X_ERROR(1/10) 0\nM(1/5) 0\nDETECTOR rec[-1]\nPOSTSELECT det[0]'),max_weight=2,
                           task={'observables':[[0,1]]},ideal_state=parse_circuit('I 0'))
        t=r['totals'];self.assertEqual(r['status'],'complete',r)
        self.assertEqual(rational(t['exact']['acceptance']),Fraction(37,50))
        self.assertEqual(rational(t['exact']['error']),Fraction(1,50))
        self.assertAlmostEqual(t['conditional_error_within_enumerated_sector'],1/37)
        self.assertEqual(rational(t['exact']['moments'][0]),Fraction(7,10))

    def test_nonlinear_magic_feedback_exactly_sums_irrational_branches(self):
        registry=ControllerRegistry();calls=[]
        def decode(bits,config):calls.append(tuple(bits));return (bits[0]&bits[1],)
        registry.register('and',decode,version='1')
        physical=parse_circuit('H 0\nT 0\nH 0\nCX 0 1\nM 0 1\n'
                              'DETECTOR rec[-2]\nDETECTOR rec[-1]\n'
                              'CONTROL[and] det[-2] det[-1] -> ctrl[0]\nCX ctrl[0] 2',controllers=registry)
        r=enumerate_errors(physical,max_weight=0,task={'outputs':[2]},ideal_state=parse_circuit('I 0'))
        self.assertEqual(r['status'],'complete',r)
        self.assertEqual(r['feedback_branches'],2)
        self.assertEqual(set(calls),{(0,0),(1,1)})
        self.assertAlmostEqual(r['totals']['error'],(2-math.sqrt(2))/4)
        self.assertEqual(r['totals']['covered_probability'],1)

    def test_terminal_feedback_and_guarded_noise(self):
        registry=ControllerRegistry();registry.register('echo',lambda bits,config:bits,version='1')
        recovery=parse_circuit('H 1\nM 1\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0',controllers=registry)
        r=enumerate_errors(parse_circuit('X_ERROR(1/10) 0'),max_weight=1,
                           recovery=recovery,ideal_state=parse_circuit('I 0'))
        self.assertEqual(r['status'],'complete',r)
        self.assertEqual(rational(r['totals']['exact']['error']),Fraction(1,2))
        r=enumerate_errors(parse_circuit('H 1\nM 1\nX_ERROR(1/5) rec[-1] 0'),
                           max_weight=1,task={'outputs':[0]},ideal_state=parse_circuit('I 0'))
        self.assertEqual(r['status'],'complete',r)
        self.assertEqual(rational(r['totals']['exact']['error']),Fraction(1,10))

    def test_failure_does_not_return_complete_estimates(self):
        r=self.evaluate('X_ERROR(1/10) 0\nX_ERROR(1/10) 0',2,max_patterns=3)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['completed_patterns'],0)
        self.assertIsNone(r['totals'])
        registry=ControllerRegistry();registry.register('echo',lambda bits,config:bits,version='1')
        c=parse_circuit('H 0\nM 0\nDETECTOR rec[-1]\nCONTROL[echo] det[-1] -> ctrl[0]\nCX ctrl[0] 0',controllers=registry)
        r=enumerate_errors(c,max_weight=0,max_branches=1,ideal_state=parse_circuit('I 0'))
        self.assertEqual(r['status'],'failed');self.assertIsNone(r['totals'])

    def test_mandatory_suffix_prunes_impossible_prefixes(self):
        from magna import _native
        categories=tuple([((9,1),10)]*200 + [((0,1),1)]*4)
        patterns,denominator=native_testing.load().enumerate_noise_categories(categories,3,1)
        self.assertEqual(patterns,[])
        patterns,denominator=native_testing.load().enumerate_noise_categories(categories,4,1)
        self.assertEqual(len(patterns),1)
        self.assertEqual(patterns[0][0],tuple((j,0) for j in range(200,204)))

    def test_invalid_budgets_and_cli(self):
        for value in (-1,True,1.5,None):
            with self.assertRaises(ValueError):self.evaluate('I 0',value)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'c.magna').write_text('X_ERROR(1/10) 0\nM 0')
            (p/'task.json').write_text(json.dumps({'outputs': [0], 'ideal_state': ZERO_TARGET}))
            out=StringIO()
            with redirect_stdout(out):code=main(['enumerate',str(p/'c.magna'),'--task',str(p/'task.json'),'--max-weight','1','--json'])
            self.assertEqual(code,0,out.getvalue())
            self.assertEqual(json.loads(out.getvalue())['totals']['error'],.1)


if __name__=='__main__':unittest.main()
