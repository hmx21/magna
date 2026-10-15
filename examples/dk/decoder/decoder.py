"""Total Steane CSS decoder from the maintained manuscript task."""
DK_H7 = tuple(sum(1 << (q-1) for q in row) for row in ((1,2,6,7), (2,3,4,7), (4,5,6,7)))
COLUMNS=[tuple((h>>q)&1 for h in DK_H7) for q in range(7)]

def decode(bits,config):
 """Total CSS minimum-weight correction from six measured syndrome bits."""
 answer=[0]*14
 for offset,syndrome in [(0,tuple(bits[:3])),(7,tuple(bits[3:]))]:
  if any(syndrome):answer[offset+COLUMNS.index(syndrome)]=1
 return tuple(answer)
