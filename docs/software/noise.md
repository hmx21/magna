# Noise

Noise is declared explicitly in the physical circuit, at the place where it acts.
The simulator never adds noise of its own and never moves a channel.

```text
R 0 1
H 0
DEPOLARIZE1(0.001) 0
T 0
Z_ERROR(0.002) 0
CX 0 1
DEPOLARIZE2(0.001) 0 1
M(0.003) 1
CZ rec[-1] 0
```

## Channels

| Channel | Action |
|---|---|
| `X_ERROR(p)`, `Y_ERROR(p)`, `Z_ERROR(p)` | The named Pauli with probability p |
| `DEPOLARIZE1(p)`, `DEPOLARIZE2(p)` | A uniformly random non-identity Pauli with total probability p |
| `PAULI_CHANNEL_1(px, py, pz)`, `PAULI_CHANNEL_2(...)` | Explicit Pauli probabilities |
| `CORRELATED_ERROR(p) P...` followed by `ELSE_CORRELATED_ERROR(q) P...` | A chain of mutually exclusive Pauli products, as in Stim |
| `M(p)`, `MX(p)`, ... | Flip the *reported* result with probability p; the true outcome is unchanged |

Names, argument order and probability conventions follow the
[Stim noise reference](https://github.com/quantumlib/Stim/blob/main/doc/gates.md#noise-channels).
Probabilities may be decimals, scientific notation such as `1e-4`, or fractions such as `1/30`;
all are parsed as exact rationals.

## True and reported outcomes

A measurement with a report-flip probability has two outcomes: the true outcome, which the
quantum projector applies, and the reported outcome, which every classical reader sees.
Detectors, `rec[-k]`, decoders, `POSTSELECT` and logical observables all use reported values.
A decoder therefore inherits report errors through its inputs, and its decision may be wrong
without any fault having occurred on a data qubit.

## Conditional noise

A noise instruction after a conditional gate is unconditional by default.
To model a fault that occurs only when the gate is physically executed,
put the gate's control bit in front of the channel's targets:

```text
CX ctrl[0] 2
X_ERROR(0.01) ctrl[0] 2
```

The channel then fires only when `ctrl[0]` is one.
Correlated alternatives must be contiguous and share the same guard:

```text
CORRELATED_ERROR(0.1) ctrl[0] X0 Y1
ELSE_CORRELATED_ERROR(2/9) ctrl[0] Z1
```

Given an active guard, the first alternative has probability 0.1 and the second `0.9 * 2/9 = 0.2`.
The sampler resolves the guard from reported information in causal order and samples the channel
when the guard is one.
Any guarded channel makes the task use history sampling, even if the guard is affine or constant.
Guarded channels keep their physical probabilities under importance sampling.
A conditional gate adds no noise of its own. When a fault acts whichever way a correction goes,
for example after an injection that yields either `T` or `T_DAG`, write its channel unconditionally.

## How faults are drawn

Each channel is a category of mutually exclusive alternatives.
The sampler first decides whether an error occurs, then which alternative.
Occurrence probabilities are rounded down to a multiple of 2^-64 (about 5.4e-20);
zero and one stay exact.
At physical rates of 1e-3 to 1e-4 this rounding is below one part in 1e15,
and it does not limit how small an estimated logical error rate can be.
Exact queries, likelihood ratios and enumeration use the exact rational values.

Equal occurrence probabilities are grouped and up to 64 independent flags are generated at once
from packed random bits. The same seed therefore gives the same faults whether draws are taken
one at a time or in batches, within one package version.

Importance sampling scales every eligible unguarded error probability by a common factor and
keeps the relative rates and alternative proportions; see [sampling methods](sampling.md).
Enumeration sums the same categories exactly; see [exact enumeration](enumeration.md).
