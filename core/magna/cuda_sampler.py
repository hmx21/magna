"""Optional device-batched Pauli sampler with exact query kernels.

Compilation stays in MAGNA. Philox noise, frame accumulation, exact scalar
queries and batch statistics run on the device. Only eleven numbers per chunk
are copied back. This is separate from the default public CPU sampler.

Physical sampling and certified homogeneous count-table importance proposals
are supported. Likelihoods multiply joint contributions; means are divided by
the number of attempted draws, never by the sum of likelihoods.

Individual probabilities are exact cyclotomic scalars. Statistics, as in the
CPU public API, are floating-point estimates. The aggregation contract accepts
dyadic target normalization and bounded real scalar coefficients; unsupported
ranges raise an error rather than silently clipping or underflowing.
"""
import math
from operator import index
from time import perf_counter

import numpy as np

from ._cuda_source import _read_cuda_source

from .cuda_experiment import CudaExperiment
from .cuda_noise import FusedCudaNoise, MASK64, _uint64

_MAX_LIKELIHOOD = math.ldexp(1., 480)

_STATISTICS_SOURCE = _read_cuda_source("statistics.cu")


class CenteredMoments(np.ndarray):
    """Five legacy raw diagnostics plus stable means and a covariance factor.

    The upper-triangular factor ``[r00, r01, r11]`` satisfies
    ``R.T @ R = sum((x-mean)(x-mean).T)`` for ``x=(acceptance,error)``.
    Raw moments remain useful for exact dyadic replay checks, but they are not
    subtracted to obtain variances in the production sampler.
    """
    def __new__(cls, raw, count, means, factor):
        obj = np.asarray(raw, dtype=np.float64).view(cls)
        obj.sample_count = int(count)
        obj.centered_mean = np.asarray(means, dtype=np.float64)
        obj.covariance_root = np.asarray(factor, dtype=np.float64)
        return obj

    def __array_finalize__(self, original):
        if original is None:
            return
        self.sample_count = getattr(original, 'sample_count', None)
        self.centered_mean = getattr(original, 'centered_mean', None)
        self.covariance_root = getattr(original, 'covariance_root', None)

    def centered_dict(self):
        # Keep the square root representation: explicitly squaring it here
        # could underflow even though the standard error remains representable.
        return dict(mean=self.centered_mean.tolist(), covariance_root=self.covariance_root.tolist())


def _fma(a, b, c):
    if hasattr(math, 'fma'):
        return math.fma(a, b, c)
    # Python 3.10-3.12 lack math.fma. This exact host fallback is used only
    # for a few covariance-factor operations per batch, never per shot.
    from fractions import Fraction
    return float(Fraction(a)*Fraction(b)+Fraction(c))


def _add_covariance_row(factor, x, y):
    """Scalar QR rank update; called per chunk, never once per shot on CPU."""
    r00, r01, r11 = map(float, factor)
    if x == 0:
        return np.array([r00, r01, math.hypot(r11, y)])
    radius = math.hypot(r00, x)
    cosine, sine = r00/radius, x/radius
    xy = sine*r01
    residual = _fma(cosine,y,-xy)-_fma(sine,r01,-xy)
    yy = sine*y
    middle = _fma(cosine,r01,yy)+_fma(sine,y,-yy)
    return np.array([radius, middle, math.hypot(r11, residual)])


def merge_moments(chunks):
    """Chan mean/covariance merge, with stable square-root rank updates."""
    chunks = list(chunks)
    totals = [math.fsum(float(chunk[k]) for chunk in chunks) for k in range(5)]
    count, means, factor = 0, np.zeros(2), np.zeros(3)
    for chunk in chunks:
        if not isinstance(chunk, CenteredMoments) or chunk.sample_count is None:
            raise TypeError('stable merging requires centered batch statistics')
        other_n = chunk.sample_count
        if not other_n:
            continue
        if not count:
            count = other_n
            means = chunk.centered_mean.copy()
            factor = chunk.covariance_root.copy()
            continue
        delta = chunk.centered_mean-means
        total = count+other_n
        weight = math.sqrt(count*other_n/total)
        other = chunk.covariance_root
        factor = _add_covariance_row(factor, float(other[0]), float(other[1]))
        factor = _add_covariance_row(factor, 0., float(other[2]))
        factor = _add_covariance_row(factor, weight*float(delta[0]), weight*float(delta[1]))
        means += delta*(other_n/total)
        count = total
    return CenteredMoments(totals, count, means, factor)


def _real_values(coefficients, powers):
    """Vectorized CPU equivalent of the bounded, stable real conversion."""
    c = np.asarray(coefficients, dtype=np.int64)
    if np.any(c < -2147483647) or np.any(c > 2147483647):
        raise ValueError('statistics coefficient exceeds signed 31-bit conversion bound')
    if np.any(c[..., 2] != 0) or np.any(c[..., 1] != -c[..., 3]):
        raise ValueError('statistics require real cyclotomic probabilities')
    a, b = c[..., 0], c[..., 1]
    value = a.astype(np.float64) + b.astype(np.float64) * math.sqrt(2)
    opposite = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
    if np.any(opposite):
        aa, bb = a[opposite], b[opposite]
        value[opposite] = (aa*aa - 2*bb*bb).astype(np.float64) / (
            aa.astype(np.float64) - bb.astype(np.float64)*math.sqrt(2))
    with np.errstate(over='ignore', under='ignore'):
        result = np.ldexp(value, powers)
    if np.any(~np.isfinite(result)) or np.any((value != 0) & (result == 0)):
        raise ValueError('statistics probability overflows or underflows float64')
    return result


def cpu_moments(coefficients, powers, norm_power, weights=None):
    """Compiled NumPy array operations for the CPU pipeline statistics control."""
    c, p = np.asarray(coefficients), np.asarray(powers)
    if c.dtype != np.int64 or p.dtype not in (np.dtype('int32'), np.dtype('int64')):
        raise TypeError('exact-query results require int64 coefficients and signed integer powers')
    if np.any(p < -(1 << 31)) or np.any(p > (1 << 31)-1):
        raise ValueError('query powers exceed int32 range')
    p = p.astype(np.int64, copy=False)
    if c.ndim != 3 or c.shape[-1] != 4 or c.shape[1] not in (1, 2) or p.shape != c.shape[:2]:
        raise ValueError('invalid exact-query result shape')
    a = _real_values(c[:, 0], p[:, 0])
    e = np.zeros_like(a)
    if c.shape[1] == 2:
        ca, ct = c[:, 0], c[:, 1]
        safe = (1 << 61)-1
        if np.any(ct < -safe) or np.any(ct > safe):
            raise ValueError('statistics target coefficient exceeds int64 sign bound')
        if np.any(ct[:, 2] != 0) or np.any(ct[:, 1] != -ct[:, 3]):
            raise ValueError('statistics require a real target probability')
        ta, tb = ct[:, 0], ct[:, 1]
        if np.any((ta <= 0) & (tb <= 0) & ((ta != 0) | (tb != 0))):
            raise ValueError('query result has a negative target probability')
        opposite = ((ta < 0) & (tb > 0)) | ((ta > 0) & (tb < 0))
        if np.any(opposite):
            x, y = ta[opposite], tb[opposite]
            if np.any(np.abs(x) > 2147483647) or np.any(np.abs(y) > 2147483647):
                raise ValueError('target sign exceeds the exact statistics bound')
            delta = x*x-2*y*y
            if np.any(((x > 0) & (delta < 0)) | ((x < 0) & (delta > 0))):
                raise ValueError('query result has a negative target probability')
        pa, pt = p[:, 0].copy(), p[:, 1] - norm_power
        pa[np.all(ca == 0, axis=1)] = pt[np.all(ca == 0, axis=1)]
        pt[np.all(ct == 0, axis=1)] = pa[np.all(ct == 0, axis=1)]
        common = np.minimum(pa, pt)
        da, dt = pa-common, pt-common
        if np.any(da > 61) or np.any(dt > 61):
            raise ValueError('statistics exact error alignment exceeds int64 bound')
        bound_a, bound_t = ((1 << 61)-1) >> da, ((1 << 61)-1) >> dt
        if (np.any(ca > bound_a[:, None]) or np.any(ca < -bound_a[:, None]) or
                np.any(ct > bound_t[:, None]) or np.any(ct < -bound_t[:, None])):
            raise ValueError('statistics exact error alignment exceeds int64 bound')
        error = (ca << da[:, None]) - (ct << dt[:, None])
        e = _real_values(error, common)
    if np.any(a < 0) or np.any(e < 0) or np.any(a > 1+1e-13) or np.any(e > a+1e-13):
        raise ValueError('query result violates physical probability bounds')
    if weights is not None:
        weights = np.asarray(weights)
        if weights.dtype != np.float64 or weights.shape != a.shape:
            raise ValueError('likelihoods must be float64 with one value per shot')
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > _MAX_LIKELIHOOD):
            raise ValueError('likelihoods must be finite, nonnegative and at most 2^480')
        old_a, old_e = a, e
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            a, e = a*weights, e*weights
            invalid = np.any(~np.isfinite(a)) or np.any(~np.isfinite(e)) or np.any(~np.isfinite(a*a)) or np.any(~np.isfinite(e*e))
        if invalid or np.any((weights != 0) & (((old_a != 0) & (a == 0)) | ((old_e != 0) & (e == 0)))):
            raise ValueError('weighted statistics overflow or underflow float64')
    if np.any((a != 0) & (a*a == 0)) or np.any((e != 0) & (e*e == 0)):
        raise ValueError('statistics second moment underflows float64')
    raw = np.array([a.sum(), e.sum(), (a*a).sum(), (e*e).sum(), (a*e).sum()], dtype=np.float64)
    if not a.size:
        return CenteredMoments(raw, 0, np.zeros(2), np.zeros(3))
    # Anchoring before averaging makes a constant irrational input exactly
    # constant even when raw sum/n would move by a rounding bit.
    da, de = a-a[0], e-e[0]
    ma, me = da.mean(), de.mean()
    da, de = da-ma, de-me
    scale_a, scale_e = float(np.max(np.abs(da))), float(np.max(np.abs(de)))
    def norm(values):
        scale = float(np.max(np.abs(values)))
        return scale*math.sqrt(float(((values/scale)**2).sum())) if scale else 0.
    if scale_a:
        scaled_a = da/scale_a
        variance_scaled = float((scaled_a*scaled_a).sum())
        slope = ((scale_e/scale_a)*float((scaled_a*(de/scale_e)).sum())/variance_scaled) if scale_e else 0.
        r00 = scale_a*math.sqrt(variance_scaled)
        residual = de-slope*da
        factor = [r00, slope*r00, norm(residual)]
    else:
        factor = [0., 0., norm(de)]
    return CenteredMoments(raw, a.size, [float(a[0])+ma, float(e[0])+me], factor)


def summarize_moments(moments, count, has_error=True):
    """IID errors from centered covariance; raw-only input is legacy diagnostic.

    A bare list/array cannot recover variance already lost from its raw sums.
    Production ``run`` always supplies ``CenteredMoments``. The raw-only form
    remains for compatibility and is not reliable for nearly constant values.
    """
    sa, se, saa, see, sae = map(float, moments)
    if not count:
        return dict(estimates=dict(acceptance=None, error=None, conditional_error=None),
                    standard_errors=dict(acceptance=None, error=None, conditional_error=None))
    centered = isinstance(moments, CenteredMoments) and moments.sample_count is not None
    if centered:
        if moments.sample_count != count:
            raise ValueError('centered statistics count does not match the supplied count')
        a, e = map(float, moments.centered_mean)
    else:
        a, e = sa/count, se/count
    ratio = e/a if has_error and a else None
    sa_error = se_error = ratio_error = None
    if count > 1:
        denominator = count*(count-1)
        scale = math.sqrt(denominator)
        if centered:
            r00, r01, r11 = map(float, moments.covariance_root)
            sa_error = r00/scale
            if has_error:
                se_error = math.hypot(r01, r11)/scale
                if ratio is not None:
                    ratio_error = math.hypot(_fma(-ratio, r00, r01), r11)/scale/abs(a)
        else:
            sa_error = math.sqrt(max(0., saa-sa*sa/count))/scale
            if has_error:
                se_error = math.sqrt(max(0., see-se*se/count))/scale
                if ratio is not None:
                    ratio_error = math.sqrt(max(0., see-2*ratio*sae+ratio*ratio*saa))/scale/a
    return dict(estimates=dict(acceptance=a, error=e if has_error else None, conditional_error=ratio),
                standard_errors=dict(acceptance=sa_error, error=se_error, conditional_error=ratio_error))


class CudaSampler:
    """GPU backend for physical or supported homogeneous importance tasks."""
    def __init__(self, compiled, **query_options):
        from .cuda_queries import GpuQueries
        self.experiment = CudaExperiment(compiled)
        self.noise = FusedCudaNoise(self.experiment)
        if np.any(self.noise.likelihood_values > _MAX_LIKELIHOOD):
            raise ValueError('importance likelihood table exceeds the 2^480 statistics bound')
        self.queries = GpuQueries(self.experiment, **query_options)
        coefficients, power = self.experiment.metadata['target_normalization']
        if tuple(coefficients) != (1, 0, 0, 0):
            raise ValueError('GPU summary currently requires a positive power-of-two target normalization')
        self.norm_power = int(power)
        if abs(self.norm_power) > 1_000_000:
            raise ValueError('target normalization exponent exceeds summary bound')
        self.has_error = compiled.ideal_state is not None
        self._statistics_module = None

    def gpu_moments(self, exact, weights=None):
        cp = self.noise.cp
        c, p = exact.coefficients, exact.powers
        if c.ndim != 3 or c.shape[2] != 4 or c.shape[1] not in (1, 2) or p.shape != c.shape[:2]:
            raise ValueError('invalid exact GPU query result shape')
        if c.dtype != cp.int64 or p.dtype != cp.int32 or not c.flags.c_contiguous or not p.flags.c_contiguous:
            raise ValueError('GPU query arrays must be contiguous int64 coefficients and int32 powers')
        query_flags = getattr(exact, 'status', None)
        if query_flags is None:
            query_flags = cp.zeros(p.shape, dtype=cp.int32)
        if query_flags.dtype != cp.int32 or query_flags.shape != p.shape or not query_flags.flags.c_contiguous:
            raise ValueError('GPU query status must be contiguous int32 with one flag per query')
        if weights is not None:
            if (not isinstance(weights, cp.ndarray) or weights.dtype != cp.float64 or
                    weights.shape != (c.shape[0],) or not weights.flags.c_contiguous):
                raise ValueError('device likelihoods must be contiguous float64[shots]')
            likelihoods = weights
        else:
            if getattr(self, '_unit_weight_pointer', None) is None:
                self._unit_weight_pointer = cp.empty(1, dtype=cp.float64)
            likelihoods = self._unit_weight_pointer
        if not c.shape[0]:
            return CenteredMoments(np.zeros(5), 0, np.zeros(2), np.zeros(3))
        if self._statistics_module is None:
            self._statistics_module = cp.RawModule(code=_STATISTICS_SOURCE, options=('--std=c++11',))
        first = self._statistics_module.get_function('scalar_moments')
        finish = self._statistics_module.get_function('finish_moments')
        blocks = (c.shape[0]+127)//128
        partial = cp.empty((blocks, 10), dtype=cp.float64)
        flags = cp.zeros(1, dtype=cp.uint32)
        output = cp.empty(11, dtype=cp.float64)
        first((blocks,), (128,), (c, p, query_flags, likelihoods, np.int32(weights is not None),
                                  np.uint64(c.shape[0]), np.int32(c.shape[1]),
                                  np.int32(self.norm_power), partial, flags))
        finish((1,), (128,), (partial, np.uint64(blocks), np.uint64(c.shape[0]), flags, output))
        host = output.get()
        if host[10]:
            raise ValueError(f'GPU probability summary rejected values (flags={int(host[10])}: '
                             '1=nonreal, 2=integer bound, 4=probability bound, 8=float range, 16=query contract, 32=likelihood)')
        return CenteredMoments(host[:5], c.shape[0], host[5:7], host[7:10])

    def run(self, shots, *, seed=0, offset=0, batch_size=65536, backend='gpu'):
        self.experiment.validate()
        shots, seed, offset = _uint64(shots, 'shots'), _uint64(seed, 'seed'), _uint64(offset, 'offset')
        if shots and offset > MASK64-(shots-1):
            raise ValueError('global shot ID overflows uint64')
        if isinstance(batch_size, bool):
            raise TypeError('batch_size must be an integer')
        batch_size = index(batch_size)
        if not 1 <= batch_size <= self.experiment.metadata['max_shots']:
            raise ValueError('batch_size outside supported range')
        if backend not in ('gpu', 'cpu'):
            raise ValueError('backend must be gpu or cpu')
        if shots > (1 << 53):
            raise ValueError('summary shot count exceeds exact float64 count range')
        start = perf_counter()
        chunks = []
        weighted = bool(getattr(self.noise, 'has_importance_weights', False))
        for first in range(0, shots, batch_size):
            size = min(batch_size, shots-first)
            weights = None
            if backend == 'gpu':
                if weighted:
                    frames, weights = self.noise.sample_weighted(size, seed, offset=offset+first)
                else:
                    frames = self.noise.sample(size, seed, offset=offset+first)
                exact = self.queries.evaluate(frames)
                moments = self.gpu_moments(exact, weights)
            else:
                if weighted:
                    frames, weights = self.noise.cpu_sample_weighted(size, seed, offset=offset+first)
                else:
                    frames = self.noise.cpu_sample(size, seed, offset=offset+first)
                exact = self.queries.cpu_evaluate(frames)
                moments = cpu_moments(exact.coefficients, exact.powers, self.norm_power, weights)
            chunks.append(moments)
        totals = merge_moments(chunks)
        return dict(status='complete', backend=backend, requested=shots, completed=shots,
                    seed=seed, offset=offset, batch_size=batch_size, rng='philox4x32-10-shot-counter',
                    timings=dict(sampling_seconds=perf_counter()-start),
                    statistics_arithmetic='float64 centered covariance factor of exact cyclotomic probabilities',
                    likelihood_weighting='native homogeneous count table' if weighted else 'unity',
                    normalization='mean joint contributions over attempted draws; conditional ratio of aggregate means',
                    statistics_transfer_bytes_per_batch=88 if backend == 'gpu' else 0,
                    moments=totals.tolist(), centered_statistics=totals.centered_dict(),
                    **summarize_moments(totals, shots, self.has_error))
