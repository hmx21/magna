# Experimental CUDA sampling

`magna.cuda_sampler.CudaSampler` is an explicit, optional backend for supported
fixed Pauli-noise tasks. It reuses `compile_circuit` and its exact query plans,
then runs noise sampling, Pauli-frame accumulation, conditional probability
queries and batch statistics on an NVIDIA GPU. It does not change `magna.run`,
the `magna` command, the default CPU random stream, or existing decoder APIs.

This backend is experimental and supports a subset of the general simulator.
Unsupported structures and numerical ranges raise errors; there is no hidden
per-draw CPU fallback.

## Install

Install from a checkout containing this feature, with Python 3.11 or newer and
the usual C++17 compiler. The CUDA 12 extra adds NumPy and CuPy:

```sh
python -m pip install ".[cuda12]"
```

A compatible NVIDIA driver and CUDA runtime/NVRTC are also needed. If a CUDA 12
toolkit is not installed, CuPy supports installing NVIDIA's component wheels:

```sh
python -m pip install "cupy-cuda12x[ctk]>=14,<15"
```

See [CuPy's installation guide](https://docs.cupy.dev/en/stable/install.html)
for platform and driver requirements. Do not install conflicting CuPy packages
in one environment. This extra targets CUDA 12; CPU-only users need neither
NumPy nor CuPy to import and run the default installed MAGNA package.

CUDA sources are packaged under `core/magna/native/cuda/` and compiled at runtime
by CuPy/NVRTC. Building the Python extension does not invoke `nvcc` or require a
GPU. First use includes context creation, uploads, compilation and allocation;
reuse a sampler when measuring sustained throughput.

## Python API

```python
from magna import compile_circuit
from magna.cuda_sampler import CudaSampler

compiled = compile_circuit(
    "examples/bh2-T/physical.magna",
    task="examples/bh2-T/task.json",
    sampling={"method": "physical"},
)
sampler = CudaSampler(compiled)
result = sampler.run(1_048_576, seed=7, batch_size=65_536)
print(result["estimates"], result["standard_errors"])
```

`CudaSampler.run(shots, *, seed=0, offset=0, batch_size=65536, backend="gpu")`
attempts `shots` draws, split into chunks of at most `batch_size`. A shorter tail
chunk processes the remainder. The batch limit is 1,000,000; the total draw count
is limited to `2**53` for exact float64 count representation. `seed`, `offset`
and the final global draw index must fit unsigned 64-bit integers.

The default batch is a generic starting point, not an optimal size for every
task. To tune throughput, keep total draws fixed, vary only the batch size,
warm each candidate and compare synchronized complete calls. A batch chosen as
the smallest attaining 95% of peak throughput is a memory/throughput compromise,
not necessarily the fastest measured batch. Large batches can also be slower.

`backend="cpu"` runs the new backend's Philox/query-plan control, which needs
NumPy but not CuPy. It is useful for deterministic differential checks. It is
**not** the original CPU sampler and must not be substituted for `magna.run` in
an original-CPU speedup claim.

Do not reconfigure or modify a compiled task's options after constructing a
sampler. Recreate both objects when changing the circuit, target or sampling
configuration. Use one sampler from one host thread at a time.

### Results and numerical contract

The returned dictionary is specific to this experimental API; it is not the
full `magna.run` report schema. Its principal fields are:

| Field | Meaning |
|---|---|
| `status`, `requested`, `completed` | Completion and attempted draw counts |
| `seed`, `offset`, `batch_size`, `rng` | Reproducibility and chunking information |
| `estimates.acceptance` | Mean accepted mass |
| `estimates.error` | Mean accepted-error mass, or `None` without a target |
| `estimates.conditional_error` | Aggregate error / acceptance, or `None` when undefined |
| `standard_errors` | Corresponding floating-point IID standard-error estimates |
| `moments`, `centered_statistics` | Raw diagnostic sums and stable centered summary |
| `timings.sampling_seconds` | Sampling and reduction time, before final Python result formatting |

Per-draw conditional probabilities use bounded exact integer representations of
cyclotomic scalars, including genuinely non-Clifford values. Target normalization
must be a positive power of two for the aggregate sampler. Likelihood conversion,
means and standard errors use guarded float64 arithmetic, not arbitrary-precision
rational statistics. Invalid probabilities and unsupported overflow/underflow
ranges fail explicitly.

Device reduction maintains centered covariance factors; Python merges chunk
summaries and formats estimates. It does not subtract two large raw moments to
estimate a small variance. A batch returns eleven scalar summary/status values
to the host, rather than copying every draw's frame and probability.

The lower-level `CudaExperiment` and `GpuQueries` interfaces expose unweighted
conditional masses. For importance sampling, use `CudaSampler` or explicitly
apply the returned likelihoods before averaging.

## Supported tasks

- Fixed Pauli-noise schedules, including native categorical/correlated Pauli
  actions and measurement-report flips, with affine frame propagation.
- Acceptance and ideal-target error queries. Observable-moment requests,
  decoder callbacks, sampled conditional histories and guarded noise are not
  supported by this backend.
- Physical sampling, or a homogeneous importance proposal represented by one
  complete native count-weight table covering all nonunit likelihood factors.
  Counted proposals currently require 2--512 sites. Heterogeneous multi-group
  and custom stratified proposals are rejected.
- Certified reduced queries: dynamic quadratic/Gauss queries with dimension at
  most 64; statically certified Gauss geometry up to 1,024; bounded residual
  phase sums up to dimension 12. These are reduced query dimensions, **not**
  the physical circuit's qubit count. Constructor and execution checks enforce
  the representation and geometry bounds.

Importance sampling preserves the compiled proposal and native likelihood
table. Each draw's accepted/error contribution is multiplied by its likelihood;
means divide by attempted draws, never by the sum of weights. Contributions may
exceed one. The full likelihood table must fit the positive float64 range and
the sampler's `2**480` weight bound.

For example, this explicitly changes BH 14-to-2 to a homogeneous importance
proposal. It is a different sampling configuration from the physical example:

```python
compiled = compile_circuit(
    "examples/bh2-T/physical.magna",
    task="examples/bh2-T/task.json",
    sampling={"method": "importance", "proposal_scale": "1000/7"},
)
result = CudaSampler(compiled).run(65_536, seed=7, batch_size=16_384)
```

## Random streams

CUDA uses Philox4x32-10 addressed by the global draw ID. Changing chunk size
preserves that backend's noise choices for a fixed seed and offset. Chunk merges
can differ by floating-point rounding. The original CPU sampler uses its existing
MT stream, so equal CPU/GPU seeds do **not** imply the same concrete histories.
Use the same-Philox CPU control or explicit noise histories for exact comparisons.
Existing CPU seeded results remain unchanged.

## Portable tasks and benchmarks

The CUDA task script uses the same `run.json` loader as `scripts/run_task.py`:

```sh
python scripts/run_cuda_task.py --task examples/bh2-T --shots 65536 --batch-size 16384 --output run.json
python scripts/benchmark_cuda_sampling.py --task examples/bh2-T --shots 262144 --batch-size 65536 --repeat 3 --output benchmark.json
```

Both preserve the task's options and sampling configuration. To override it,
provide `--sampling physical`, or explicitly provide importance and its scale:

```sh
python scripts/benchmark_cuda_sampling.py --task examples/bh2-T --sampling importance --proposal-scale 1000/7 --shots 262144 --batch-size 65536 --repeat 3 --output benchmark-importance.json
```

The benchmark compares original `magna.run` with the GPU on equal total attempted
draws, using complete synchronized API wall times and repeated medians. Preparation
and explicit warmups are separate. Each timed GPU call has an immediately preceding
same-size warmup, so long CPU calls do not give only some GPU trials an idle start.
The ordinary task runner warms one chunk; use the benchmark for throughput claims.
Records include task/configuration, input and
installed-module hashes, software, hardware and individual timings. Measure the
actual installed version and identify it with those records. These measurements do not imply cold-start speedups or a
comparison with a tuned multicore CPU implementation.

## Validation

The regular source suite runs CPU bridge, exact-plan, importance and numerical
statistics checks with NumPy. GPU tests skip when CuPy or a CUDA device is absent;
kernel compilation or execution failures are not silently skipped. Run all checks
with the regular project command, or only this backend with:

```sh
python -m unittest discover -s tests/core -p 'test_cuda*.py' -v
```

The CUDA checks include exact small-circuit oracles, the public RP2/BH tasks,
Philox controls, full-frame basis checks, importance weights, chunk invariance,
stable standard errors and explicit rejection paths. Standard CPU CI does not
establish GPU runtime correctness; report an actual CUDA run and its environment
when changing these kernels.
