# Concepts

This page defines the vocabulary used throughout the manual and in result files.
Read it once; the user guide pages assume it.

## One circuit, one amplitude

The simulator treats the whole protocol, for its fixed input, as a single quantum instrument:
the preparation, every gate, every Pauli fault, every measurement, every classical decision
and every qubit that is eventually discarded.
A query asks for one number about that instrument, for example the probability of a
measurement pattern, or the overlap of the output with a target state.
The query is answered by compiling the instrument globally and contracting it, not by
evolving a state through the circuit step by step.

Two things follow. First, the cost of a query depends on its *resource*: the size of the
stabilizer decomposition that remains after global elimination for *that query*.
It is often much smaller than the number of T gates in the circuit would suggest. Second, every measurement that no decision and no task
depends on is integrated analytically; it never has to be sampled.

Preparation compiles one *reference network* for each kind of query a task needs:
acceptance, fidelity and, when bits must be sampled, a sampling network
([compilation](architecture/compiler.md) describes them).
Each network is first written as a sum over binary *path variables*.
Global elimination leaves a smaller number of *phase coordinates*, and the remaining
non-Clifford phase is expanded into *stabilizer terms*; the number of terms is the resource.
The [resource options](options.md) bound each of these quantities.

## The three files

A task is specified by up to three circuit files in the same syntax, each with a distinct role.

| File | Role | Noise | Classical functions |
|---|---|---|---|
| Physical circuit | The protocol as implemented: preparation, gates, faults, measurements, feedback | As declared | Allowed |
| Terminal recovery (optional) | A noiseless correction step applied to the selected output before comparing with the target | None | Allowed |
| Ideal state (optional) | A noiseless preparation of the target state from zero | None | Not applicable |

The physical file is always required.
The ideal state turns the error metric into target-state infidelity.
Recovery requires an ideal state and models an ideal final decoding round;
it changes the task, so use it deliberately and say so when reporting results.

## Records: true and reported

Every measurement produces a *true* outcome, which the quantum projector uses, and a
*reported* outcome, which is the true outcome flipped with the measurement's declared
report-error probability.
Everything classical reads reported values: `rec[-k]`, detectors, control registers,
decoders, `POSTSELECT` and logical observables.
A report flip therefore changes what a decoder sees without changing the quantum state.

| Reference | Meaning |
|---|---|
| `rec[-k]` | The k-th most recent reported measurement result |
| `det[n]` | The parity of the records listed in detector n |
| `ctrl[k]` | A classical register written by a `CONTROL` instruction |

## Feedback: affine and external

Feedback is a classical function of reported bits that chooses which gates to apply.
Two kinds are distinguished:

- **Affine** feedback is a parity of reported bits, possibly complemented: direct `rec[-k]`
  controls and `CONTROL[XOR]` / `CONTROL[NOT]`. The compiler absorbs it into the global
  amplitude, and the bits it reads need not be sampled.
- **External** feedback is any other function, such as a majority vote or a matching decoder,
  supplied as a registered Python function. Before the function runs, the selected
  [history sampler](#what-is-exact-and-what-is-sampled) draws the reported bits it reads;
  the simulator then calls the function and continues with its decision.

Results label a task's *route* as `linear` when all feedback is affine and `nonlinear` otherwise.
The simulator classifies the composed function symbolically: if an external decoder's
contribution cancels exactly and only a parity remains, the task is linear.

Decoders see reported observations and their own earlier decisions only.
They never see which faults occurred. A decoder that returns a wrong correction simply
produces a wrong output, which the error metric records; the run does not abort or discard that draw.

## Acceptance, error and conditional error

The circuit declares an *acceptance event* with `POSTSELECT`: detector values or decoder bits that must hold.
The task declares an *error*: a logical observable, which may include a decoder's prediction,
or infidelity against the ideal state.

For one complete draw (one set of faults and decoder decisions), the simulator computes exactly:

| Quantity | Definition |
|---|---|
| `acceptance` | The probability mass of the acceptance event |
| `error` | The joint mass of acceptance *and* error. For an ideal target it is accepted mass minus accepted success mass, where success mass is the overlap with the target after recovery |

Over many draws the runner reports the means of these two quantities and their ratio:

| Estimate | Meaning |
|---|---|
| `estimates.acceptance` | Probability that a run is accepted |
| `estimates.error` | Probability that a run is accepted and wrong |
| `estimates.conditional_error` | Error rate among accepted runs: `error / acceptance` |

With an ideal target, `conditional_error` is the infidelity of the accepted output.
Rejected draws stay in the averages with zero accepted mass; they are never redrawn.
If the aggregate acceptance is zero, the conditional error is undefined and reported as `null`.

Acceptance and error are joint means, so they are unbiased.
Their ratio is a ratio estimator; its standard error uses the delta method and it is not
exactly unbiased at small sample counts.

## What is exact and what is sampled

| Part of a run | Treatment |
|---|---|
| Pauli faults and report flips | Sampled from the declared noise (or an importance proposal) |
| Reported bits read by external decoders | Drawn by the history sampler, given the faults and earlier decisions |
| Records requested through `sample_records` | Drawn by the history sampler |
| All other measurements | Integrated analytically |
| Acceptance and error for one draw | Exact, including all interference terms |

Standard errors therefore describe sampling variation over faults and decoder inputs only.
A circuit without noise, without external decoders and without requested records gives
zero standard error after any number of shots.

Two *history samplers* are available for these measurement bits:
`affine_support` (default) proposes histories from a linear family that is known to contain every
possible history, then reweights each proposal by its exact probability;
`exact_marginal` draws each bit from its exact conditional probability.
Both are unbiased. [Sampling methods](sampling.md) explains when to prefer which.

## Three estimators

| Estimator | Command | Use when |
|---|---|---|
| Physical sampling | `magna run` | Errors are common enough to be observed in the available shots |
| Importance sampling | `magna run --sampling` | Errors are rare combinations of faults; scale up noise and reweight |
| Exact enumeration | `magna enumerate` | Low-weight fault sectors dominate and you want exact numbers with bounds |

All three accept the same physical circuit, task, ideal state, recovery and decoder bindings.

## Budgets and failure

The compile step reports how much resource a task needs.
Options such as `max_terms` and `max_seconds` bound the work.
If a budget is exceeded, the result has `status: "failed"` with the reason,
and any completed draws are reported separately as partial statistics.
The simulator never substitutes an approximation. See [resource options](options.md).

## Reproducibility

A fixed package version and seed reproduce a run exactly.
Different versions may use the random stream differently, so compare runs across versions
by their physical estimates and uncertainties, not draw by draw.
Results carry fingerprints of the circuit, task, decoder bindings and options.
