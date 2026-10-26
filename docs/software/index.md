# magna: Magic-state-factory Assessment via Global Network Algebra

magna evaluates noisy quantum protocols that contain non-Clifford gates,
measurements, Pauli noise and classical feedback, and reports exact or statistically
estimated acceptance probabilities, logical error rates and target-state fidelities.
It is built for magic-state distillation, cultivation and injection protocols,
fault-tolerant memories with decoders, and any other fixed-input circuit whose
output is a few qubits or a few bits.

## What it computes

You describe a protocol in a Stim-style text file: preparations, Clifford gates,
`T`, `CS` and `CCZ` gates, measurements, explicit Pauli noise, corrections that
depend on measurement results or on an external decoder, and a `POSTSELECT` line
that says which detector pattern counts as acceptance.
You then state a *task*: which qubits are the output and what counts as an error.
The simulator returns:

| Quantity | Meaning |
|---|---|
| Acceptance | Probability that the protocol's postselection event occurs |
| Error | Joint probability of acceptance and a logical error |
| Conditional error | Error divided by acceptance: the logical error rate of accepted runs |
| Pauli moments | Expectation values of Pauli operators on the output qubits |

An error can be a wrong measurement parity, a wrong decoder output, or infidelity
against an ideal target state you supply as a small noiseless circuit,
optionally after a noiseless terminal recovery step.

Three estimators share the same circuit and task:

- **Physical sampling** draws faults from the declared noise and averages exact per-draw contributions.
- **Importance sampling** draws faults from a scaled proposal and reweights each draw exactly.
- **Low-weight enumeration** sums every fault pattern up to a chosen weight exactly and bounds the rest.

These describe the general CPU interfaces. The optional experimental
[CUDA backend](cuda.md) supports a bounded subset of fixed Pauli-noise tasks,
with exact conditional probabilities and floating-point likelihood/statistics.

## What makes it different

Stabilizer samplers such as Stim handle Clifford circuits only.
State-vector simulators handle any gate but their cost grows with the number of qubits.
magna takes a third route.
It treats the whole circuit, for its fixed input, as one amplitude.
A global compilation step eliminates every Clifford constraint and every measurement
record that nothing reads, before any non-Clifford resource is expanded.
What remains is a small stabilizer decomposition whose size is set by the requested
output and the non-Clifford structure that survives elimination, not by the circuit's
qubit count alone.

Consequences for you as a user:

- **Exact per-draw values.** For each noise draw and each decoder decision, the accepted
  mass and error mass are computed exactly, with all interference retained.
  Statistical error comes only from the noise and decoder-input draws.
- **Measurements you never read cost nothing to sample.** Only the records that
  a decoder or your task explicitly needs are sampled; all others are integrated analytically.
- **No state is evolved in time.** Causal feedback is honoured by ordering classical
  decisions, not by propagating a quantum state through the circuit.
- **Explicit failure.** If a task exceeds its resource budget, the run reports a failure.
  There is no silent approximation.

The simulator works with a fixed input state (all zeros, or whatever your preparation
gates produce). It does not simulate open-input processes or arbitrary unitaries on many qubits.

## A first run

```text
R 0
H 0
T 0
H 0
M 0
DETECTOR rec[-1]
POSTSELECT det[0]
```

`POSTSELECT det[0]` accepts the runs whose measurement returns zero, so this asks for the
probability of that outcome after `H T H`:

```sh
magna run example.magna --shots 8
```

The summary reports an acceptance of 0.853553, which is `(2 + sqrt(2)) / 4`,
with standard error zero because nothing in this circuit is random: every draw evaluates the same exact amplitude.
The [tutorial](getting-started.md) continues from here to noise, feedback, ideal targets,
enumeration and importance sampling.

## How the documentation is organised

| Section | Read it when you want to |
|---|---|
| [Installation](install.md) | Build the package from a checkout |
| [Tutorial](getting-started.md) | Run your first circuits end to end |
| [Concepts](concepts.md) | Understand what is exact, what is sampled, and how results are defined |
| User guide | Look up one topic: [circuit files](circuit-format.md), [noise](noise.md), [feedback and decoders](controllers.md), [tasks and targets](tasks.md), [results](results.md), [sampling methods](sampling.md), [exact enumeration](enumeration.md) |
| Reference | Check the exact [command line](cli.md), [Python API](api.md), [resource options](options.md) and [schemas](schemas.md) |
| [Examples](examples.md) | Follow complete protocols: RP² cultivation, 15-to-1 distillation and a decoded TQD memory |
| [How it works](architecture/index.md) | See the pipeline from parsing to sampling and the layout of the code |
