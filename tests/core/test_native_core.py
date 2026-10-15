"""Independent dense oracles for the current exact C++ primitives."""
import random
import unittest
from tests.support.exact_scalar import ExactScalar
from tests.support.exact_scalar import scalar
from tests.support import native_testing
try:
    from magna import _native
except ImportError:
    _native=None


def dense(k):
    if k is None:return {}
    width,base,columns,linear,upper,power,phase=k
    out={}
    for x in range(1<<len(columns)):
        y=base;q=0
        for j,c in enumerate(columns):
            if x>>j&1:
                y^=c;q+=linear[j]+2*(upper[j]&x).bit_count()
        out[y]=out.get(y,ExactScalar())+scalar((1,0),phase+2*(q%4),power)
    return {y:v for y,v in out.items() if v}


def exact(v):return ExactScalar(tuple(v[0]),v[1])


@unittest.skipIf(_native is None,'optional extension not installed')
class NativeTests(unittest.TestCase):
    def test_random_contractions_against_independent_dense_sum(self):
        rng=random.Random(26091461)
        for n in range(9):
            for _ in range(18):
                width=rng.randrange(5)
                k=(width,rng.randrange(1<<width),tuple(rng.randrange(1<<width) for j in range(n)),
                   tuple(rng.randrange(4) for j in range(n)),
                   tuple(rng.randrange(1<<n)&~((1<<(j+1))-1) for j in range(n)),rng.randrange(-15,3),rng.randrange(8))
                self.assertEqual(dense(_native.contract(k)),dense(k))

    def test_overlap_cancellation_complex_phase_and_large_integers(self):
        rng=random.Random(26091462)
        for _ in range(100):
            ks=[]
            for side in range(2):
                n=rng.randrange(6)
                raw=(3,rng.randrange(8),tuple(rng.randrange(8) for j in range(n)),
                     tuple(rng.randrange(4) for j in range(n)),
                     tuple(rng.randrange(1<<n)&~((1<<(j+1))-1) for j in range(n)),rng.randrange(-10,10),rng.randrange(8))
                k=_native.contract(raw)
                if k is None:break
                ks.append(k)
            if len(ks)!=2:continue
            a,b=map(dense,ks)
            expected=sum((v.conjugate()*b.get(y,ExactScalar()) for y,v in a.items()),ExactScalar())
            self.assertEqual(exact(native_testing.load().overlap(*ks)),expected)
        # Gauss sums beyond fixed-width machine integers, plus cancellation.
        for n in (63,64,65,127,128,129,257):
            for l in (0,1,2,3):
                k=(0,0,(0,)*n,(l,)*n,(0,)*n,0,3)
                found=_native.contract(k)
                v=ExactScalar() if found is None else scalar((1,0),found[6],found[5])
                unit=ExactScalar((1,0,0,0))+scalar((1,0),2*l,0)
                expected=scalar((1,0),3,0)
                for _ in range(n):expected=expected*unit
                self.assertEqual(v,expected)
        for _ in range(100):
            a=ExactScalar(tuple(rng.randrange(-(1<<300),1<<300) for j in range(4)),rng.randrange(-200,200))
            b=ExactScalar(tuple(rng.randrange(-(1<<300),1<<300) for j in range(4)),rng.randrange(-200,200))
            s,m=native_testing.load().scalar_arithmetic((a.coefficients,a.power),(b.coefficients,b.power))
            self.assertEqual(exact(s),a+b);self.assertEqual(exact(m),a*b)

    def test_packed_width_boundaries_and_dependent_columns(self):
        rng=random.Random(26091463)
        for width in (0,1,63,64,65,127,128,129,257):
            for _ in range(10):
                n=6
                cols=[rng.randrange(1<<width) for j in range(n)]
                cols[4]=cols[0]^cols[1];cols[5]=0
                k=(width,rng.randrange(1<<width),tuple(cols),tuple(rng.randrange(4) for j in range(n)),
                   tuple(rng.randrange(1<<n)&~((1<<(j+1))-1) for j in range(n)),-4,5)
                self.assertEqual(dense(_native.contract(k)),dense(k))







    def test_malformed_native_data_and_initialization_errors_are_not_fallbacks(self):
        for k in ((1,0,(2,),(0,),(0,),0,0),(1,0,(1,),(0,),(1,),0,0),
                  (1,-1,(),(),(),0,0),(1,0,(1,),(),(),0,0)):
            with self.assertRaises(ValueError):_native.contract(k)

if __name__=='__main__':unittest.main()
