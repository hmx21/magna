# Exact values

Results that the simulator reports as floating-point numbers are computed from exact
objects. The classes on this page are the Python views of those objects. They are
immutable, their arithmetic runs in the C++ kernel, and their conversion methods
produce ordinary Python numbers for inspection or serialisation.
Every class and function on this page is imported from `magna.operations`.

## ExactScalar

Every amplitude and probability in a Clifford+T circuit is a number of the form
`(a + b w + c w^2 + d w^3) * 2^power` with `w = exp(i pi / 4)` and integers `a, b, c, d, power`.
`ExactScalar(coefficients=(a, b, c, d), power=0)` stores exactly that; the coefficients may be
arbitrarily large and a common power of two is moved into `power` on construction.

```python
from magna.operations import ExactScalar

one = ExactScalar((1, 0, 0, 0))
half = ExactScalar((1, 0, 0, 0), -1)
assert half + half == one
assert half.ratio(one) == 0.5
assert half.to_dict() == {"coefficients": [1, 0, 0, 0], "power": -1}
```

| Operation | Meaning |
|---|---|
| `+`, `-`, `*`, `scaled(n)`, `conjugate()` | Exact arithmetic; results are new scalars |
| `ratio(denominator)` | A real scalar divided by a positive real scalar, as a float that does not overflow or underflow prematurely |
| `log2()` | Base-2 logarithm of a nonnegative real scalar |
| `real_decimal()`, `complex()` | Presentation conversions |
| `to_dict()` | The coefficients and the scale |

Invalid coefficient shapes raise `ValueError`.

## StabilizerKet

A stabilizer state can be written as a sum over the bit strings of an affine subspace,
each with a phase that is a quadratic function of the string.
`StabilizerKet(width, base, columns, linear, upper, power=0, phase=0)` stores it that way:

| Argument | Meaning |
|---|---|
| `width` | Number of qubits |
| `base`, `columns` | The subspace: `base` plus any combination of the independent `columns`, as packed little-endian integers |
| `linear` | Linear phase coefficients, taken modulo 4 |
| `upper` | The strictly upper triangular quadratic phase form, modulo 2 |
| `power` | Overall scale `2^(power/2)` |
| `phase` | Overall phase in eighth roots of unity |

The constructor rejects dependent columns and malformed phase data with `ValueError`.
`positive(width, basis=(), base=0)` builds the normalised uniform superposition over a basis,
for example `StabilizerKet.positive(1, (1,))` is the plus state.

```python
from magna.operations import ExactScalar, StabilizerKet

plus = StabilizerKet.positive(1, (1,))
zero_branch = plus.project((1,), (0,))
assert zero_branch.norm() == ExactScalar((1, 0, 0, 0), -1)
assert plus.project((0,), (1,)) is None
assert plus.pauli(z=1).norm() == plus.norm()
```

| Method | Meaning |
|---|---|
| `norm()` | The squared norm as an `ExactScalar` |
| `project(rows, outcomes)` | Project the listed parities onto the listed bits; `None` if the result is the zero vector |
| `pauli(x=0, z=0)` | Apply a Pauli product given by its X and Z masks, keeping its phase |
| `contract(...)` | Sum out internal coordinates; `None` if the result vanishes |
| `sample_rows(rows, rng)` | Sample the listed parities with the caller's `rng.integers(2)` stream |

## StabilizerExpansion

`StabilizerExpansion(terms)` is a linear combination of stabilizer kets: an ordered tuple of
`(ExactScalar, StabilizerKet)` pairs. This is the form in which a non-Clifford state is
represented after decomposition. `norm()` includes all cross terms between the kets;
`pauli_expectation(x=0, z=0)` evaluates a Hermitian Pauli product; `transformed_pauli`
returns the expansion with a Pauli applied. `contract_target(ket, target, output_count)` and
`target_weight` project the output wires onto a target ket and keep the rest.

## Record marginals

`NativeRecordMarginals(rows, event_rows=(), outcomes=(), max_plans=12)` computes the exact
joint probability distribution of a set of record parities for a given expansion,
optionally restricted to an event. It caches up to `max_plans` prepared factorisations.
`evaluate(expansion, x=0, z=0)` returns a `ProbabilityMarginal`:

| Method | Meaning |
|---|---|
| `total(rows=(), outcomes=())` | The probability of an event |
| `value(word)` | The exact probability of one bit string |
| `sample(rng)` | A sampled bit string and the name of the method used, from the caller's `getrandbits` stream |

A requested output Pauli must commute with the record rows; otherwise `ValueError`.
A distribution with zero total mass cannot be sampled.

```python
from random import Random
from magna.operations import ExactScalar, NativeRecordMarginals, StabilizerExpansion, StabilizerKet

state = StabilizerExpansion(((ExactScalar((1, 0, 0, 0)),
                             StabilizerKet.positive(1, (1,))),))
marginal = NativeRecordMarginals((1,)).evaluate(state)
assert marginal.total() == ExactScalar((1, 0, 0, 0))
word, method = marginal.sample(Random(7))
assert word in (0, 1)
assert method == "affine-mixture"
```

## NoiseModel

`NoiseModel` holds the noise sites of an `InstrumentIR` circuit.
`draw(rng, physical_rate=None, proposal_scale=None)` draws one fault assignment and returns
it with its exact physical-over-proposal likelihood ratio.
`proposal_scale` multiplies every stochastic site's error probability, exactly as the file
interface's importance sampling does; `physical_rate` replaces every site's physical rate
before scaling. Sites with probability 0 or 1 stay deterministic, and a scaled probability
of one or more raises an error. The draw uses the supplied object's `randrange` method.
Each decision compares a uniformly drawn integer with the exact rational probability.
The file interface instead seeds its own native random engine once per draw and decides
occurrence with 64-bit thresholds: each probability is rounded down to a multiple of 2^-64,
while zero and one stay exact (see [noise](../noise.md#how-faults-are-drawn)).
Both report exact rational likelihood ratios.
