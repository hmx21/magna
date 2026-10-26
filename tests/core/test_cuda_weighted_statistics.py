"""Paper importance estimates use weighted joint means, not self-normalization."""
from fractions import Fraction
from types import SimpleNamespace
import unittest

from tests.support.cuda import require_cuda

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('optional NumPy required')

from magna.cuda_sampler import CudaSampler, cpu_moments, summarize_moments
from tests.core.test_cuda_sampler import exact_arrays, independent_moments


class WeightedCpuStatisticsTests(unittest.TestCase):
    def test_likelihoods_multiply_joint_contributions_without_clipping(self):
        c,p=exact_arrays([1,Fraction(1,2),0,Fraction(1,4)],
                         [Fraction(1,2),Fraction(1,4),0,Fraction(1,8)])
        w=np.array([2.,.25,4.,0.])
        actual=cpu_moments(c,p,0,w)
        np.testing.assert_allclose(actual,independent_moments([2.,.125,0.,0.],[1.,.0625,0.,0.]))
        summary=summarize_moments(actual,4)
        self.assertEqual(summary['estimates']['acceptance'],2.125/4)
        self.assertNotEqual(summary['estimates']['acceptance'],2.125/w.sum())
        self.assertEqual(summary['estimates']['conditional_error'],.5)

    def test_zero_likelihood_and_weighted_values_above_one(self):
        c,p=exact_arrays([1,1],[0,0])
        summary=summarize_moments(cpu_moments(c,p,0,np.array([2.,2.])),2)
        self.assertEqual(summary['estimates']['acceptance'],2.)
        self.assertEqual(summary['standard_errors']['acceptance'],0.)
        summary=summarize_moments(cpu_moments(c,p,0,np.zeros(2)),2)
        self.assertEqual(summary['estimates']['acceptance'],0.)
        self.assertIsNone(summary['estimates']['conditional_error'])

    def test_invalid_weights_and_range_are_rejected(self):
        c,p=exact_arrays([Fraction(1,2)],[Fraction(1,4)])
        for weights in (np.array([-1.]),np.array([np.nan]),np.array([np.inf]),
                        np.array([1e308]),np.array([2.**481]),np.array([2.**-1074]),np.array([[1.]]),
                        np.array([1.],dtype=np.float32)):
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                cpu_moments(c,p,0,weights)


class WeightedGpuStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = require_cuda()

    def fake(self):
        sampler=object.__new__(CudaSampler)
        sampler.noise=SimpleNamespace(cp=self.cp)
        sampler.norm_power=0
        sampler._statistics_module=None
        return sampler

    def test_weighted_gpu_moments_match_independent_mean(self):
        c,p=exact_arrays([1,Fraction(1,2),0,Fraction(1,4)]*65,
                         [Fraction(1,2),Fraction(1,4),0,Fraction(1,8)]*65)
        w=np.array([2.,.25,4.,0.]*65)
        exact=SimpleNamespace(coefficients=self.cp.asarray(c),powers=self.cp.asarray(p))
        gpu=self.fake().gpu_moments(exact,self.cp.asarray(w))
        cpu=cpu_moments(c,p,0,w)
        np.testing.assert_allclose(gpu,cpu,rtol=1e-14,atol=1e-16)
        for section in ('estimates','standard_errors'):
            a,b=summarize_moments(gpu,len(w))[section],summarize_moments(cpu,len(w))[section]
            for key in a:
                np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-16)

    def test_gpu_invalid_weights_are_rejected(self):
        c,p=exact_arrays([Fraction(1,2)],[Fraction(1,4)])
        exact=SimpleNamespace(coefficients=self.cp.asarray(c),powers=self.cp.asarray(p))
        for weights in ([-1.],[np.nan],[np.inf],[1e308],[2.**481],[2.**-1074]):
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                self.fake().gpu_moments(exact,self.cp.asarray(weights,dtype=self.cp.float64))
        with self.assertRaises(ValueError):
            self.fake().gpu_moments(exact,np.array([1.]))

    def test_paper_bh2_complete_sampler_matches_same_philox_cpu(self):
        from pathlib import Path
        from magna import compile_circuit
        root=Path(__file__).resolve().parents[2]
        compiled=compile_circuit(root/'examples/bh2-T/physical.magna',
            task=root/'examples/bh2-T/task.json',sampling={'method':'importance','proposal_scale':'1000/7'})
        sampler=CudaSampler(compiled)
        self.assertTrue(sampler.noise.has_importance_weights)
        gpu=sampler.run(2048,seed=2026100800,batch_size=511)
        cpu=sampler.run(2048,seed=2026100800,batch_size=383,backend='cpu')
        np.testing.assert_allclose(gpu['moments'],cpu['moments'],rtol=2e-14,atol=1e-16)
        for section in ('estimates','standard_errors'):
            for key,value in gpu[section].items():
                np.testing.assert_allclose(value,cpu[section][key],rtol=3e-12,atol=1e-16)
        self.assertEqual(gpu['likelihood_weighting'],'native homogeneous count table')
        self.assertGreater(gpu['estimates']['error'],0.)


if __name__=='__main__':
    unittest.main()
