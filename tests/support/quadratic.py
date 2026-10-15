"""Independent finite polynomial arithmetic for tests only."""

def qvalue(x,linear,upper):
    result=0
    while x:
        bit=x&-x;i=bit.bit_length()-1;x^=bit
        result+=linear[i]+2*(upper[i]&x).bit_count()
    return result%4



def multiply(a,b):
    result=[0]*4
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            k=i+j;result[k%4]+=x*y*(-1 if k>=4 else 1)
    return tuple(result)
