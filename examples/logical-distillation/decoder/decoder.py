"""Small, total, independent-patch MWPM controllers.

Ordinary d-round memory windows use their circuit-level detector error models,
separately for X and Z checks, with an open final time boundary. A depolarizing
entrance prior represents errors inherited from the preceding operation.
Cross-window, cross-patch and fold correlations are not inferred. This is a
local MWPM baseline, not an assertion that the full algorithm DEM factorizes.
No fault or ideal-outcome information is accepted by a controller.
"""
from functools import lru_cache
from fractions import Fraction

import numpy as np
import pymatching
import stim

from .patch import Patch


@lru_cache(maxsize=32)
def matching(d, basis, rounds, final_open):
    patch = Patch(d)
    checks = patch.xchecks if basis == 'X' else patch.zchecks
    h = np.zeros((len(checks), patch.n), dtype=np.uint8)
    for i, support in enumerate(checks):
        h[i, support] = 1
    graph = pymatching.Matching(h, repetitions=rounds)
    if final_open:
        for i in range(len(checks)):
            graph.add_boundary_edge((rounds-1)*len(checks)+i, weight=1,
                                    merge_strategy='smallest-weight')
    return graph


@lru_cache(maxsize=32)
def memory_model(d, basis, rounds, rate):
    """Construct a *decoder model*, including a virtual final frame readout.

    The final ideal inverse encoder and readouts label the Pauli correction
    on model edges. They are never added to the physical distillation circuit
    and never supply detector data to a controller.
    """
    from .generate import Builder
    import re
    patch = Patch(d)
    builder = Builder(d,rate,0)
    builder.active = {0}
    circuit = stim.Circuit()
    circuit.append('R',range(patch.width))
    if basis == 'X':
        circuit.append('H',[0])
    circuit += patch.encoder
    circuit.append('DEPOLARIZE1',range(patch.n),float(rate))
    start = 0 if basis == 'X' else len(patch.xanc)
    count = len(patch.xanc) if basis == 'X' else len(patch.zanc)
    previous = None
    for _ in range(rounds):
        records = builder.cycle(0)[start:start+count]
        for i,r in enumerate(records):
            terms = [r] + ([] if previous is None else [previous[i]])
            builder.lines.append('DETECTOR '+' '.join(f'rec[{v-builder.records}]' for v in terms))
        previous = records
    # Stim accepts decimal probabilities; magna circuit files also accept fractions.
    source = '\n'.join(builder.lines)
    source = re.sub(r'\(([0-9]+/[0-9]+)\)',lambda m:'('+str(float(Fraction(m[1])))+')',source)
    circuit += stim.Circuit(source)
    circuit += patch.encoder.inverse()
    if basis == 'X':
        circuit.append('H',[0])
    circuit.append('M',range(patch.n))
    indices = [0] + list(range(1+start,1+start+count))
    corrections = [patch.encoding_tableau.z_output(0) if basis == 'X'
                   else patch.encoding_tableau.x_output(0)]
    corrections += [patch.encoding_tableau.x_output(i) for i in indices[1:]]
    for i,q in enumerate(indices):
        circuit.append('OBSERVABLE_INCLUDE',[stim.target_rec(q-patch.n)],i)
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem), corrections, dem


def decode(bits, config):
    p = Patch(config['distance'])
    mode = config['mode']
    if mode == 'ideal_recovery':
        if len(bits) != len(p.anc):
            raise ValueError('ideal recovery requires one complete reported syndrome')
        nx = len(p.xanc)
        x = matching(p.d, 'Z', 1, False).decode(bits[nx:])
        z = matching(p.d, 'X', 1, False).decode(bits[:nx])
        return tuple(map(int, x)) + tuple(map(int, z))
    if mode == 'readout':
        basis = config['basis']
        if len(bits) != p.n:
            raise ValueError('readout requires exactly d*d reported data bits')
        checks = p.xchecks if basis == 'X' else p.zchecks
        syndrome = [sum(bits[q] for q in s) % 2 for s in checks]
        correction = matching(p.d,basis,1,False).decode(syndrome)
        logical = p.lx if basis == 'X' else p.lz
        return (sum(int(bits[q]) ^ int(correction[q]) for q in logical) % 2,)
    if mode in ('gauge', 'gauge_zero'):
        # Initial |+> has random Z stabilizers. Gauge fixing commutes with X_L.
        basis = 'Z' if mode == 'gauge' else 'X'
        checks = p.zchecks if mode == 'gauge' else p.xchecks
        if len(bits) != len(checks):
            raise ValueError('gauge input length mismatch')
        return tuple(map(int,matching(p.d,basis,1,False).decode(bits)))
    if mode != 'memory':
        raise ValueError('unknown matching controller mode')
    rounds = config['rounds']
    if len(bits) != rounds * len(p.anc):
        raise ValueError('memory input length mismatch')
    history = np.array(bits,dtype=np.uint8).reshape(rounds,len(p.anc))
    differences = history.copy()
    differences[1:] ^= history[:-1]
    nx = len(p.xanc)
    frame = stim.PauliString(p.n)
    for basis, syndrome in [('X',differences[:,:nx]),('Z',differences[:,nx:])]:
        graph, corrections, _ = memory_model(p.d,basis,rounds,float(Fraction(config['model_rate'])))
        labels = graph.decode(syndrome.ravel())
        for bit,pauli in zip(labels,corrections):
            if bit:
                frame *= pauli
    x,z = frame.to_numpy()
    return tuple(map(int,x)) + tuple(map(int,z))
