"""Dense independent verification of completed-pair record trace and phases."""
import cmath
import random
import unittest
from magna.evaluate import StabilizerExpansion
from magna.stabilizer import StabilizerKet
from tests.support.quadratic import qvalue
from magna.scalar import ONE
from tests.support.operation_hooks import scalar
from magna.native_records import NativeRecordMarginals


def dense(expansion,n):
    out=[0j]*(1<<n)
    for c,k in expansion.terms:
        for v in range(1<<len(k.columns)):
            word=k.base
            for j, column in enumerate(k.columns):
                if (v >> j) & 1: word ^= column
            out[word]+=c.complex()*2**(k.power/2)*cmath.exp(1j*cmath.pi*(k.phase+2*qvalue(v,k.linear,k.upper))/4)
    return out


class NativeRecordTests(unittest.TestCase):
    def test_dense_events_and_hermitian_moments(self):
        a=StabilizerKet(3,0,(1,2,4),(1,2,0),(2,4,0),-3,3)
        b=StabilizerKet(3,1,(3,4),(3,1),(2,0),-2,6)
        e=StabilizerExpansion(((scalar((1,0),1,-1),a),(scalar((1,0),-2,-1),b)))
        psi=dense(e,3)
        for event in (0,1):
            plan=NativeRecordMarginals((1,2),event_rows=(3,),outcomes=(event,))
            for x,z in ((0,0),(0,4),(4,0),(4,4)):
                m=plan.evaluate(e,x=x,z=z);expected=[0j]*4
                for word in range(8):
                    if (word&3).bit_count()%2!=event:continue
                    expected[word&3]+=psi[word].conjugate()*(1j**((x&z).bit_count()))*((-1)**((z&(word^x)).bit_count()))*psi[word^x]
                for word,v in enumerate(expected):self.assertAlmostEqual(m.value(word).complex(),v,places=12)
                self.assertAlmostEqual(m.total().complex(),sum(expected),places=12)
                self.assertAlmostEqual(m.total((1,),(1,)).complex(),expected[1]+expected[3],places=12)

    def test_sampling_completed_marginal(self):
        # The measured T state has irrational biased X outcomes; exercise the
        # signed-pair fallback instead of only a stabilizer distribution.
        plus=StabilizerKet.positive(1,(1,));minus=StabilizerKet(1,0,(1,),(2,),(0,),-1,0)
        e=StabilizerExpansion(((scalar((1,0),0,-1),plus),(scalar((1,0),1,-1),minus)))
        m=NativeRecordMarginals((1,)).evaluate(e);self.assertEqual(m.total(),ONE)
        rng=random.Random(739);count=0
        for _ in range(2048):
            word,method=m.sample(rng);count+=word
        expected=m.value(1).ratio(ONE)
        self.assertLess(abs(count/2048-expected),.04)

    def test_reject_noncommuting_pauli(self):
        e=StabilizerExpansion(((ONE,StabilizerKet.positive(1,(1,))),))
        with self.assertRaises(ValueError):NativeRecordMarginals((1,)).evaluate(e,x=1)


if __name__=='__main__':unittest.main()
