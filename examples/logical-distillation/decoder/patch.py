"""Rotated d-by-d patches and Chen et al.'s literal half-cycle fold.

Geometry, CNOT ordering and fold support follow the MIT-licensed author code
Zihan-Chen-PhMA/Dynamical-S-gate-decoding, c3b7a1e2f95d49a5853d5534a4596854317890e9.
See CHEN_LICENSE. All synthesis here is experiment-side, outside the kernel.
"""
from functools import cached_property

import stim


class Patch:
    def __init__(self, d):
        if type(d) is not int or d < 3 or d % 2 != 1:
            raise ValueError("distance must be an odd integer at least 3")
        self.d = d
        self.n = d*d
        self.data = [(x, y) for x in range(d) for y in range(d)]
        self.xpos = ([(2*x+.5, 2*y+.5) for x in range(d//2) for y in range(d//2)]
                     + [(2*x+1.5, 2*y+1.5) for x in range(d//2) for y in range(d//2)]
                     + [(-.5, 2*y+1.5) for y in range(d//2)]
                     + [(d-.5, 2*y+.5) for y in range(d//2)])
        self.zpos = ([(2*x+.5, -.5) for x in range(d//2)]
                     + [(2*x+.5, 2*y+1.5) for x in range(d//2) for y in range(d//2)]
                     + [(2*x+1.5, 2*y+.5) for x in range(d//2) for y in range(d//2)]
                     + [(2*x+1.5, d-.5) for x in range(d//2)])
        self.positions = self.data + self.xpos + self.zpos
        self.index = {pos: i for i, pos in enumerate(self.positions)}
        self.width = len(self.positions)
        self.xanc = list(range(self.n, self.n + len(self.xpos)))
        self.zanc = list(range(self.n + len(self.xpos), self.width))
        self.anc = self.xanc + self.zanc
        self.xchecks = [self.support(pos) for pos in self.xpos]
        self.zchecks = [self.support(pos) for pos in self.zpos]
        self.lx = [self.index[x, d//2] for x in range(d)]
        self.lz = [self.index[d//2, y] for y in range(d)]

    def support(self, pos):
        return [self.index[p] for dx, dy in [(-.5,-.5),(.5,-.5),(-.5,.5),(.5,.5)]
                if (p := (pos[0]+dx, pos[1]+dy)) in self.index and self.index[p] < self.n]

    def pauli(self, basis, support):
        p = stim.PauliString(self.n)
        for q in support:
            p[q] = basis
        return p

    @cached_property
    def encoding_tableau(self):
        # The unconstrained logical X sign/support is fixed explicitly.
        stabilizers = ([self.pauli('Z', self.lz)]
                      + [self.pauli('X', s) for s in self.xchecks]
                      + [self.pauli('Z', s) for s in self.zchecks])
        t = stim.Tableau.from_stabilizers(stabilizers)
        # Fix both logical axes, including their signs. Adjust the other
        # destabilizers to commute with the chosen logical X representative.
        lx = self.pauli('X',self.lx)
        xs = [lx]
        zs = [t.z_output(i) for i in range(self.n)]
        for i in range(1,self.n):
            x = t.x_output(i)
            if not x.commutes(lx):
                x *= zs[0]
            xs.append(x)
        return stim.Tableau.from_conjugated_generators(xs=xs,zs=zs)

    @cached_property
    def encoder(self):
        return self.encoding_tableau.to_circuit(method='elimination')

    def layers(self):
        directions = {'nw':(-.5,-.5), 'ne':(.5,-.5), 'sw':(-.5,.5), 'se':(.5,.5)}
        for zi, xi in zip(('nw','ne','sw','se'), ('nw','sw','ne','se')):
            pairs = []
            for positions, ancillas, name, is_x in ((self.zpos,self.zanc,zi,False),
                                                    (self.xpos,self.xanc,xi,True)):
                dx, dy = directions[name]
                for (x,y), a in zip(positions,ancillas):
                    q = self.index.get((x+dx,y+dy))
                    if q is not None and q < self.n:
                        pairs.append((a,q) if is_x else (q,a))
            yield pairs

    def fold(self, inverse=False):
        circuit = stim.Circuit()
        for q,(x,y) in enumerate(self.positions):
            if not (0 <= x <= self.d-1 and 0 <= y <= self.d-1):
                continue
            if x+y < self.d-1:
                circuit.append('CZ',[q,self.index[self.d-1-y,self.d-1-x]])
            elif x+y == self.d-1:
                dag = (x != int(x)) ^ inverse
                circuit.append('S_DAG' if dag else 'S',[q])
        return circuit

    def cycle(self, fold=0):
        circuit = stim.Circuit()
        circuit.append('R',self.anc)
        circuit.append('H',self.xanc)
        for i, pairs in enumerate(self.layers()):
            circuit.append('CX',[q for pair in pairs for q in pair])
            if i == 1 and fold:
                circuit += self.fold(inverse=fold < 0)
        circuit.append('H',self.xanc)
        circuit.append('M',self.anc)
        return circuit
