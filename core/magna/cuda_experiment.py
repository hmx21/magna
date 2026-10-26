"""Low-level compiled-plan bridge and validation controls for optional CUDA.

The bridge exports exact noise maps and full physical/magic Pauli frames.
Its explicit noise-draw and probability-evaluation methods use the original
native CPU implementation as a validation oracle; ``CudaFrameMapper`` also
offers standalone device frame transport. The complete device sampling path
is provided by ``magna.cuda_sampler.CudaSampler``.

NumPy is needed when importing this module. CuPy is imported only when
constructing a :class:`CudaFrameMapper` or using ``gpu_frames``.
"""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from ._cuda_source import _read_cuda_source

from . import _native


_SOURCE = _read_cuda_source("frame_transport.cu")


def _unsigned(values, dtype, ndim, name):
    """Convert integer input without silently wrapping negatives or floats."""
    if isinstance(values, (list, tuple)):
        # NumPy can infer float64 for a Python sequence mixing small integers
        # and values above INT64_MAX.  Inspect objects before any conversion,
        # otherwise those uint64 RNG seeds lose their low bits.
        array = np.asarray(values, dtype=object)
        if array.ndim != ndim:
            raise ValueError(f'{name} must have {ndim} dimensions')
        maximum = np.iinfo(dtype).max
        for value in array.flat:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f'{name} must contain unsigned integers')
            if int(value) < 0 or int(value) > maximum:
                raise ValueError(f'{name} integer outside {np.dtype(dtype).name}')
        return np.ascontiguousarray(array, dtype=dtype)
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions')
    if array.dtype.kind not in 'iu':
        raise TypeError(f'{name} must contain unsigned integers')
    if array.size:
        if array.dtype.kind == 'i' and np.any(array < 0):
            raise ValueError(f'{name} contains a negative integer')
        if int(array.max()) > np.iinfo(dtype).max:
            raise ValueError(f'{name} integer exceeds {np.dtype(dtype).name}')
    return np.ascontiguousarray(array, dtype=dtype)


def validate_csr(offsets, indices, actions):
    """Validate and canonicalize sparse shot rows before CPU or GPU access."""
    offsets = _unsigned(offsets, np.uint64, 1, 'offsets')
    indices = _unsigned(indices, np.uint32, 1, 'indices')
    if not offsets.size or offsets[0] != 0:
        raise ValueError('offsets must start with zero')
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError('offsets must be nondecreasing')
    if int(offsets[-1]) != indices.size:
        raise ValueError('final offset must equal the number of indices')
    if indices.size and int(indices.max()) >= actions:
        raise ValueError('action index outside the prepared map')
    return offsets, indices


class CudaFrameMapper:
    """Prepared, reusable GPU map of sparse actions to packed X/Z frames.

    ``columns`` has shape ``(actions, 2 * words)`` and dtype uint64.  Each
    row stores the X words followed by the Z words, in little-endian word
    order.  ``base_frame`` is the same-width affine offset from the compiler.
    Duplicate action indices cancel under XOR.  ``map`` includes all input
    transfers and a synchronized result transfer back to the host.
    """

    def __init__(self, columns, base_frame=None):
        columns = _unsigned(columns, np.uint64, 2, 'columns')
        if columns.shape[1] % 2:
            raise ValueError('columns must contain equally many X and Z words')
        self.actions, self.width = columns.shape
        if base_frame is None:
            base_frame = np.zeros(self.width, dtype=np.uint64)
        base_frame = _unsigned(base_frame, np.uint64, 1, 'base_frame')
        if base_frame.size != self.width:
            raise ValueError('base_frame width differs from columns')
        if self.width > np.iinfo(np.uint32).max:
            raise ValueError('frame width exceeds the CUDA kernel limit')
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError('CUDA experiment requires optional CuPy; install cupy-cuda12x[ctk]') from exc
        self.cp = cp
        self.columns = cp.asarray(columns)
        self.base = cp.asarray(base_frame)
        self.kernel = cp.RawKernel(_SOURCE, 'sparse_frames', options=('--std=c++11',))

    def prepare_device(self, offsets, indices):
        """Validate/upload CSR once for a device-resident timing experiment.

        Returned buffers belong to this mapper and should be treated as opaque.
        Only the mapping kernel is omitted from this preparation operation.
        """
        offsets, indices = validate_csr(offsets, indices, self.actions)
        shots = offsets.size - 1
        return _DeviceBatch(self, self.cp.asarray(offsets), self.cp.asarray(indices),
                            self.cp.empty((shots, self.width), dtype=self.cp.uint64), shots)

    def launch(self, batch):
        """Enqueue one mapping into the prepared output; caller synchronizes.

        This lower-level method is intended for CUDA-event timings.  Use
        ``map`` when a completed host result is required.
        """
        if not isinstance(batch, _DeviceBatch) or batch.owner is not self:
            raise ValueError('device batch belongs to a different mapper')
        size = batch.shots * self.width
        if size:
            self.kernel(((size + 255) // 256,), (256,),
                        (self.columns, self.base, batch.offsets, batch.indices,
                         batch.output, np.uint64(batch.shots), np.uint32(self.width)))
        return batch.output

    def map(self, offsets, indices):
        """Return packed host frames, including H2D/kernel/D2H costs."""
        batch = self.prepare_device(offsets, indices)
        return self.cp.asnumpy(self.launch(batch))


@dataclass(frozen=True)
class _DeviceBatch:
    owner: object = field(repr=False)
    offsets: object = field(repr=False)
    indices: object = field(repr=False)
    output: object = field(repr=False)
    shots: int


@dataclass(frozen=True)
class NoiseBatch:
    """Sparse native noise draws tied to one experimental compiled plan."""
    owner: object = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    indices: np.ndarray = field(repr=False)

    @property
    def shots(self):
        return self.offsets.size - 1


class CudaExperiment:
    """Export a compiled circuit's fixed plan for the optional CUDA backend.

    Creation rejects models not supported by the native experimental bridge,
    including sampled histories, conditional noise and importance proposals
    without a complete certified homogeneous native count-weight table.
    This class's explicit draw/evaluate methods are CPU validation controls;
    CudaSampler supplies the fused device execution path. The exact rows
    returned by the controls are ``(history, acceptance, error_or_None)`` and use
    MAGNA's ``((c0,c1,c2,c3), power)`` scalar representation.  These are raw
    masses: divide ``error`` by ``metadata['target_normalization']`` to obtain
    the physical error contribution for a postselected target preparation.
    The conditional infidelity is that contribution divided by acceptance.
    These low-level values do not include importance likelihoods. For tilted
    draws use CudaSampler, or multiply by the corresponding weights explicitly.
    """

    def __init__(self, compiled):
        if compiled.options != compiled._execution_options:
            raise ValueError('compiled options changed; recompile before creating a CUDA experiment')
        self.compiled = compiled
        self._options_snapshot = deepcopy(compiled.options)
        self.handle, self.metadata = _native.make_cuda_experiment(compiled._execution.handle)
        self.words = int(self.metadata['words'])
        self.width = 2 * self.words
        self.actions = int(self.metadata['actions'])
        self.columns = np.frombuffer(self.metadata['columns'], dtype='<u8').reshape(self.actions, self.width)
        self.base_frame = np.frombuffer(self.metadata['base_frame'], dtype='<u8')
        site_offsets = np.asarray(self.metadata['offsets'], dtype=np.uint64)
        self._action_sites = np.searchsorted(site_offsets, np.arange(self.actions, dtype=np.uint64),
                                             side='right') - 1
        self._mapper = None

    def validate(self):
        """Reject option edits or native reconfiguration of this fixed plan.

        CUDA plans snapshot a completed compilation. Unlike the default CPU
        runner they do not apply mutable option dictionaries at run time.
        Recompile and construct a new experiment after changing options.
        """
        if self.compiled.options != self._options_snapshot:
            raise ValueError('compiled options changed; recompile and recreate the CUDA experiment')
        _native.cuda_noise_validate(self.handle)

    @property
    def mapper(self):
        self.validate()
        if self._mapper is None:
            self._mapper = CudaFrameMapper(self.columns, self.base_frame)
        return self._mapper

    def draw(self, seeds):
        """Draw native noise with an explicit independent uint64 seed per shot."""
        self.validate()
        seeds = _unsigned(seeds, np.uint64, 1, 'seeds')
        offsets, indices = _native.cuda_draw_noise(self.handle, seeds.tolist())
        return NoiseBatch(self, np.frombuffer(offsets, dtype='<u8'), np.frombuffer(indices, dtype='<u4'))

    def _csr(self, batch):
        self.validate()
        if not isinstance(batch, NoiseBatch) or batch.owner is not self:
            raise ValueError('noise batch belongs to a different experiment')
        offsets, indices = validate_csr(batch.offsets, batch.indices, self.actions)
        if offsets.size - 1 > self.metadata['max_shots']:
            raise ValueError('noise batch exceeds the native shot limit')
        if indices.size > 1:
            sites = self._action_sites[indices]
            invalid = sites[1:] <= sites[:-1]
            # Comparisons straddling separate shots are unconstrained.  Empty
            # shots may repeat an offset, which is harmless here.
            starts = offsets[1:-1]
            starts = starts[(starts > 0) & (starts < indices.size)]
            invalid[(starts - 1).astype(np.intp)] = False
            if np.any(invalid):
                raise ValueError('actions must be site-ordered with at most one outcome per site')
        return offsets, indices

    def cpu_frames(self, batch):
        """Map the identical noise batch with the native C++ XOR kernel."""
        offsets, indices = self._csr(batch)
        frames = _native.cuda_map_frames(self.handle, offsets.tobytes(), indices.tobytes())
        return np.frombuffer(frames, dtype='<u8').reshape(batch.shots, self.width)

    def gpu_frames(self, batch):
        offsets, indices = self._csr(batch)
        return self.mapper.map(offsets, indices)

    def evaluate_frames(self, frames):
        """Evaluate complete frames with MAGNA's native exact T-gate kernel."""
        self.validate()
        frames = _unsigned(frames, np.uint64, 2, 'frames')
        if frames.shape[1] != self.width:
            raise ValueError('frame width differs from the compiled plan')
        return _native.cuda_evaluate_frames(self.handle, frames.tobytes())

    def evaluate_noise(self, batch):
        """Original native execution from the same noise, for exact comparison."""
        offsets, indices = self._csr(batch)
        return _native.cuda_evaluate_noise(self.handle, offsets.tobytes(), indices.tobytes())
