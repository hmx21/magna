"""Fused physical-noise sampling and full Pauli-frame accumulation on CUDA.

Noise probabilities use MAGNA's compiled uint64 RationalCoin thresholds,
including sequential conditional coins for nonuniform categorical channels.
The GPU uses Philox4x32-10, not MAGNA's historical mt19937_64 random stream.
Seed reproducibility therefore belongs to this backend, not across backends.
Every random block is addressed by (global shot ID, per-shot block counter).

``sample`` returns device-resident frames and performs no result transfer.
Validation can request site-indexed action IDs, reconstruct the exact original
noise draw and evaluate it through the original native sampler.

Importance proposals are supported only when the native compiler certifies a
single complete count table. Use ``sample_weighted`` for those plans: raw
conditional probabilities must be multiplied by the returned likelihoods.
"""
from operator import index as integer_index

import numpy as np

from ._cuda_source import _read_cuda_source

from . import _native
from .cuda_experiment import NoiseBatch


MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def philox4x32(counter, key):
    """Independent Python arithmetic oracle for the standard 10-round PRNG.

    Known-answer vectors: DEShawResearch/random123 tests/kat_vectors.
    https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    """
    a, b, c, d = map(int, counter)
    k0, k1 = map(int, key)
    if any(v < 0 or v > MASK32 for v in (a, b, c, d, k0, k1)):
        raise ValueError('Philox counter/key words must be uint32')
    for _ in range(10):
        p0, p1 = 0xD2511F53 * a, 0xCD9E8D57 * c
        a, b, c, d = ((p1 >> 32) ^ b ^ k0, p1 & MASK32,
                      (p0 >> 32) ^ d ^ k1, p0 & MASK32)
        k0, k1 = (k0 + 0x9E3779B9) & MASK32, (k1 + 0xBB67AE85) & MASK32
    return a, b, c, d


class _PhiloxStream:
    def __init__(self, seed, shot):
        self.seed, self.shot, self.block, self.saved = seed, shot, 0, None

    def __call__(self):
        if self.saved is not None:
            value, self.saved = self.saved, None
            return value
        a, b, c, d = philox4x32((self.shot & MASK32, self.shot >> 32,
                                self.block & MASK32, self.block >> 32),
                               (self.seed & MASK32, self.seed >> 32))
        self.block += 1
        self.saved = c | (d << 32)
        return a | (b << 32)


def _uint64(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer')
    try:
        value = integer_index(value)
    except TypeError:
        raise TypeError(f'{name} must be an integer') from None
    if value < 0 or value > MASK64:
        raise ValueError(f'{name} must fit uint64')
    return value


_SOURCE = _read_cuda_source("noise_sampling.cu")


class FusedCudaNoise:
    """GPU noise RNG and complete-frame XOR in one kernel, with an exact CPU control.

    ``sample(shots, seed, offset=0)`` is asynchronous and returns a CuPy uint64
    array shaped ``(shots, experiment.width)``. It can feed the GPU query kernel
    directly. Requesting ``return_actions`` additionally allocates a diagnostic
    uint32 array ``(shots, sites)`` (UINT32_MAX means no error); this mode is not
    used for production timing. Diagnostic arrays are limited to 512 MiB.
    """
    _types = dict(thresholds='<u8', certain='u1', block_offsets='<u4', block_sites='<u4',
                  choice_offsets='<u4', choices='<u4', uniform='u1',
                  conditional_thresholds='<u8', conditional_certain='u1')

    def __init__(self, experiment):
        experiment.validate()
        self.experiment = experiment
        self.metadata = _native.cuda_noise_metadata(experiment.handle)
        self.tables = {name: np.frombuffer(self.metadata[name], dtype=dtype)
                       for name, dtype in self._types.items()}
        self.width = experiment.width
        self.sites, self.blocks = int(self.metadata['sites']), int(self.metadata['blocks'])
        self.has_importance_weights = bool(self.metadata['has_importance_weights'])
        self.weighted_sites = np.frombuffer(self.metadata['weighted_sites'], dtype='u1')
        self.likelihood_values = np.frombuffer(self.metadata['likelihood_values'], dtype='<f8')
        self.likelihood_mantissas = np.frombuffer(self.metadata['likelihood_mantissas'], dtype='<f8')
        self.likelihood_exponents = np.frombuffer(self.metadata['likelihood_exponents'], dtype='<i4')
        # Keep GPU initialization lazy: metadata and CPU controls require no CUDA.
        self._cp = None

    @property
    def cp(self):
        if self._cp is None:
            import cupy as cp
            self._cp = cp
            self._columns = cp.asarray(self.experiment.columns)
            self._base = cp.asarray(self.experiment.base_frame)
            self._tables = {name: cp.asarray(value) for name, value in self.tables.items()}
            self._weighted_sites = cp.asarray(self.weighted_sites)
            self._likelihood_values = cp.asarray(self.likelihood_values)
            module = cp.RawModule(code=_SOURCE, options=('--std=c++11',))
            self._kernel = module.get_function('fused_noise')
            self._rng_kernel = module.get_function('philox_words')
            self._dummy = cp.empty(1, dtype=cp.uint32)
            self._dummy_weights = cp.empty(1, dtype=cp.float64)
        return self._cp

    def _arguments(self, shots, seed, offset):
        shots, seed, offset = (_uint64(shots, 'shots'), _uint64(seed, 'seed'),
                               _uint64(offset, 'offset'))
        if shots > self.experiment.metadata['max_shots']:
            raise ValueError('noise batch exceeds the native shot limit')
        if shots and offset > MASK64 - (shots - 1):
            raise ValueError('global shot ID overflows uint64')
        self.experiment.validate()
        return shots, seed, offset

    def sample(self, shots, seed, *, offset=0, return_actions=False):
        """Enqueue device noise/frames; does not synchronize or copy to host."""
        if self.has_importance_weights:
            raise ValueError('importance draws require sample_weighted so likelihoods are not discarded')
        return self._sample(shots, seed, offset=offset, return_actions=return_actions, weighted=False)

    def sample_weighted(self, shots, seed, *, offset=0, return_actions=False):
        """Return device ``(frames, likelihoods[, actions])`` without a host transfer.

        The float64 likelihood is the native count table's rounded exact
        target/proposal ratio. It is not a probability and can exceed one.
        Physical/no-op proposals return unit weights.
        """
        return self._sample(shots, seed, offset=offset, return_actions=return_actions, weighted=True)

    def _sample(self, shots, seed, *, offset, return_actions, weighted):
        shots, seed, offset = self._arguments(shots, seed, offset)
        if return_actions and shots * self.sites * 4 > 512 * 1024**2:
            raise ValueError('diagnostic action array exceeds 512 MiB; validate in smaller chunks')
        cp = self.cp
        frames = cp.empty((shots, self.width), dtype=cp.uint64)
        actions = cp.empty((shots, self.sites), dtype=cp.uint32) if return_actions else self._dummy
        likelihoods = cp.empty(shots, dtype=cp.float64) if weighted else self._dummy_weights
        if shots:
            args = (self._columns, self._base, *(self._tables[k] for k in self._types),
                    self._weighted_sites, self._likelihood_values, frames, actions, likelihoods,
                    np.uint64(shots), np.uint64(seed), np.uint64(offset),
                    np.uint32(self.width), np.uint32(self.blocks), np.uint32(self.sites),
                    np.int32(bool(return_actions)), np.int32(weighted))
            self._kernel(((shots + 127) // 128,), (128,), args)
        if weighted:
            return (frames, likelihoods, actions) if return_actions else (frames, likelihoods)
        return (frames, actions) if return_actions else frames

    def cpu_sample(self, shots, seed, *, offset=0, return_noise=False):
        """Compiled C++ scalar implementation of the same Philox/fusion algorithm."""
        if self.has_importance_weights:
            raise ValueError('importance draws require cpu_sample_weighted so likelihoods are not discarded')
        return self._cpu_sample(shots, seed, offset=offset, return_noise=return_noise, weighted=False)

    def cpu_sample_weighted(self, shots, seed, *, offset=0, return_noise=False):
        """Return host ``(frames, likelihoods[, NoiseBatch])`` from the C++ control."""
        return self._cpu_sample(shots, seed, offset=offset, return_noise=return_noise, weighted=True)

    def _cpu_sample(self, shots, seed, *, offset, return_noise, weighted):
        shots, seed, offset = self._arguments(shots, seed, offset)
        raw = _native.cuda_noise_cpu_sample(self.experiment.handle, shots, seed, offset, return_noise, weighted)
        if weighted:
            if return_noise:
                frames, weights, starts, indices = raw
                batch = NoiseBatch(self.experiment, np.frombuffer(starts, dtype='<u8'),
                                   np.frombuffer(indices, dtype='<u4'))
            else:
                frames, weights = raw
            output = np.frombuffer(frames, dtype='<u8').reshape(shots, self.width)
            likelihoods = np.frombuffer(weights, dtype='<f8')
            return (output, likelihoods, batch) if return_noise else (output, likelihoods)
        if return_noise:
            frames, starts, indices = raw
            batch = NoiseBatch(self.experiment, np.frombuffer(starts, dtype='<u8'),
                               np.frombuffer(indices, dtype='<u4'))
        else:
            frames = raw
        output = np.frombuffer(frames, dtype='<u8').reshape(shots, self.width)
        return (output, batch) if return_noise else output

    def weights_for_noise(self, batch):
        """Count selected weighted sites and apply the native likelihood table.

        This is a host validation helper; production GPU sampling accumulates
        the count inside its fused kernel without a CSR round trip.
        """
        offsets, indices = self.experiment._csr(batch)
        flags = self.weighted_sites[self.experiment._action_sites[indices]]
        prefix = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(flags, dtype=np.uint64)])
        counts = prefix[offsets[1:].astype(np.intp)] - prefix[offsets[:-1].astype(np.intp)]
        return self.likelihood_values[counts.astype(np.intp)]

    def actions_to_noise(self, actions):
        """Copy diagnostic actions if necessary and reconstruct original CSR draws."""
        if hasattr(actions, 'get'):
            actions = actions.get()
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != self.sites or actions.dtype != np.uint32:
            raise ValueError('expected uint32 diagnostic action array with one column per site')
        selected = actions != MASK32
        counts = selected.sum(axis=1, dtype=np.uint64)
        offsets = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64)])
        indices = np.ascontiguousarray(actions[selected], dtype=np.uint32)
        batch = NoiseBatch(self.experiment, offsets, indices)
        self.experiment._csr(batch)
        if indices.size and not np.array_equal(self.experiment._action_sites[indices],
                                               np.nonzero(selected)[1]):
            raise ValueError('diagnostic action ID does not belong to its site column')
        return batch

    def oracle_sample(self, shots, seed, *, offset=0):
        """Slow independent Python Philox/coin oracle, only for bounded validation."""
        shots, seed, offset = self._arguments(shots, seed, offset)
        if shots > 4096:
            raise ValueError('Python oracle is limited to 4096 shots; use the C++ CPU control')
        t = self.tables
        frames = np.tile(self.experiment.base_frame, (shots, 1)).copy()
        offsets, indices = [0], []
        for shot in range(shots):
            rng = _PhiloxStream(seed, offset + shot)
            chosen = []
            def coin(threshold, certain):
                return bool(certain) or (bool(threshold) and rng() < int(threshold))
            def select(site):
                first, end = map(int, t['choice_offsets'][site:site+2])
                if t['uniform'][site]:
                    n = end - first
                    at = 0
                    if n != 1:
                        reject = ((1 << 64) - n) % n
                        value = rng()
                        while value < reject:
                            value = rng()
                        at = value % n
                    action = int(t['choices'][first + at])
                else:
                    at = end - 1
                    for j in range(first, end - 1):
                        if coin(t['conditional_thresholds'][j], t['conditional_certain'][j]):
                            at = j
                            break
                    action = int(t['choices'][at])
                chosen.append(action)
                frames[shot] ^= self.experiment.columns[action]
            for block in range(self.blocks):
                first, end = map(int, t['block_offsets'][block:block+2])
                threshold, certain = int(t['thresholds'][block]), bool(t['certain'][block])
                if end - first < 8:
                    for j in range(first, end):
                        if coin(threshold, certain):
                            select(int(t['block_sites'][j]))
                    continue
                pending = (1 << (end - first)) - 1
                selected = pending if certain else 0
                if not certain:
                    remaining = threshold
                    while pending and remaining:
                        bits = rng()
                        if remaining & (1 << 63):
                            selected |= pending & ~bits
                            pending &= bits
                        else:
                            pending &= ~bits
                        remaining = (remaining << 1) & MASK64
                while selected:
                    bit = (selected & -selected).bit_length() - 1
                    selected &= selected - 1
                    select(int(t['block_sites'][first + bit]))
            indices.extend(sorted(chosen))
            offsets.append(len(indices))
        return frames, NoiseBatch(self.experiment, np.asarray(offsets, dtype=np.uint64),
                                  np.asarray(indices, dtype=np.uint32))

    def rng_words(self, shots, words, seed, *, offset=0):
        """Diagnostic device PRNG stream for known-answer and chunk tests."""
        shots, seed, offset = self._arguments(shots, seed, offset)
        words = _uint64(words, 'words')
        if words > 1024:
            raise ValueError('diagnostic word count exceeds 1024')
        cp = self.cp
        result = cp.empty((shots, words), dtype=cp.uint64)
        if shots and words:
            self._rng_kernel(((shots + 127) // 128,), (128,),
                             (result, np.uint64(shots), np.uint32(words),
                              np.uint64(seed), np.uint64(offset)))
        return result
