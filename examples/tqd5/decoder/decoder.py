"""Observation-only adapter to the pinned author's classical matching routines.

Cached objects contain only geometry and matching graphs. All per-shot state
is an explicit input/output bit vector; no outcome or correction cache is used.
"""
from functools import lru_cache
import numpy as np
from pymatching import Matching
from .author import load, REVISION

BINDING_VERSION = 'physical-2-' + REVISION


@lru_cache(None)
def model(size, depth):
    author = load()
    full = author.lattice.build_incidence_matrix(size, depth)
    prefixes = tuple(author.lattice.build_incidence_matrix(size, t+1, open_end_node=t<depth-1)
                     for t in range(depth))
    return full, Matching(full), prefixes, tuple(Matching(h) for h in prefixes)


def step_from_syndrome(syndrome, prior, size, depth, step):
    """Author jit_decode_step after replacing its hidden-noise syndrome line.

    syndrome contains exactly step*size**2 real prefix vertices, with the
    artificial future node (if present) computed here from their total parity.
    Returns the whole joined increment, not a silently truncated correction.
    """
    full, matching, prefixes, prefix_matchings = model(size, depth)
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    prior = np.asarray(prior, dtype=np.uint8)
    if not 1 <= step <= depth or syndrome.shape != (step*size**2,) or prior.shape != (full.shape[1],):
        raise ValueError('JIT input shape does not match the declared layout')
    if np.any(syndrome > 1) or np.any(prior > 1):
        raise ValueError('JIT inputs must be bits')
    if step < depth:
        syndrome = np.append(syndrome, np.sum(syndrome) % 2).astype(np.uint8)
    elif int(np.sum(syndrome)) % 2:
        # A closed matching graph cannot pair odd reported parity. Retain
        # prior decisions and make no new proposal; this is an observable
        # decoder convention, not an inference about actual logical success.
        return np.zeros_like(prior)
    estimate = prefix_matchings[step-1].decode(syndrome)
    joined = prior.copy()
    joined[:len(estimate)] ^= estimate
    return matching.decode(full @ joined % 2)


def decode(bits, config):
    """Inputs: reported B differences, then cumulative prior spacetime state.

    Outputs: current physical corrections (node-major, axes 0/1), then new
    cumulative spacetime state. Future spatial proposals are carried explicitly
    and only committed at their declared time, including previous proposals.
    """
    size, depth, step = (config[k] for k in ('size', 'depth', 'step'))
    if config.get('revision') != REVISION:
        raise ValueError('decoder revision is not pinned')
    width = size**2
    n = 3*width*depth
    if len(bits) != step*width+n:
        raise ValueError('wrong observation/state width')
    observed = np.array(bits[:step*width], dtype=np.uint8).reshape(step,width)
    prior = np.array(bits[step*width:], dtype=np.uint8)
    full, _, _, _ = model(size,depth)
    # Remove the known effect of the physical correction immediately before
    # each difference. Geometry is static; no sampled fault is inspected.
    for t in range(1,step):
        spatial = prior[3*width*(t-1):3*width*t].copy()
        spatial[2::3] = 0
        block = full[:width,:3*width]
        observed[t] ^= np.asarray(block @ spatial % 2, dtype=np.uint8)
    delta = step_from_syndrome(observed.ravel(),prior,size,depth,step)
    if np.any(delta[:3*width*(step-1)]):
        raise ValueError('author increment would revise an already committed slice')
    state = prior ^ delta
    current = state[3*width*(step-1):3*width*step].reshape(width,3)[:,:2].ravel()
    return tuple(map(int, np.concatenate((current,state))))


def exit_decode(bits, config):
    """Apply the rooted-tree gauge correction for every reported green word.

    On a coboundary this is exactly the source's ideal filling. Otherwise
    non-tree residual edges remain in the physical output. No winding test,
    hidden fault, success flag or selection affects the returned action.
    """
    size=config['size']; w=size**2
    if len(bits)!=2*w: raise ValueError('green exit width mismatch')
    if any(bit not in (0,1) for bit in bits): raise ValueError('green exit inputs must be bits')
    potential=[None]*w;potential[0]=0; pending=[0]
    while pending:
        v=pending.pop(); x,y=v%size,v//size
        neighbors=[((x+1)%size+y*size,2*v), (x+((y+1)%size)*size,2*v+1),
                   ((x-1)%size+y*size,2*((x-1)%size+y*size)),
                   (x+((y-1)%size)*size,2*(x+((y-1)%size)*size)+1)]
        for u,e in neighbors:
            value=potential[v]^bits[e]
            if potential[u] is None: potential[u]=value;pending.append(u)
    return tuple(potential)


def charge_decode(bits, config):
    """Naive author final charge matching from observed A differences only.

    No simulator residual loops or hidden success flag are available here.
    If the reported parity is odd, this closed matching problem has no
    solution. Return identity, leaving the actual residual state to the
    quantum evaluator. The decoder does not know whether that state is good.
    """
    size,depth=config['size'],config['depth'];w=size**2
    if len(bits)!=w*depth:raise ValueError('charge history width mismatch')
    if any(bit not in (0,1) for bit in bits):raise ValueError('charge inputs must be bits')
    if sum(bits)%2:return (0,)*(2*w)
    full,matching,_,_=model(size,depth)
    prediction=matching.decode(np.array(bits,dtype=np.uint8))
    correction=prediction.reshape(depth,w,3)[:,:,:2].sum(axis=0)%2
    return tuple(map(int,correction.ravel()))
