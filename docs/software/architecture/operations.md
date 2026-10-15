# Operation-level API

The file interface described in the [Python API](../api.md) covers ordinary use.
The classes on this page sit one level below it: you describe a circuit as Python
dictionaries, specify one complete measurement outcome together with its faults,
and obtain its exact probability. Tests and custom samplers use this interface.
Every class and function on this page is imported from `magna.operations`.

## Describe a circuit

```python
from magna.operations import ExactScalar, InstrumentIR, SampleAssignment, Simulator, StabilizerKet

ir = InstrumentIR((0,), (0,), (
    {"id": "phase", "kind": "T", "qubits": [0]},
    {"id": "read", "kind": "MX", "qubits": [0], "record": "m"},
), StabilizerKet.positive(1, (1,)))
sim = Simulator(ir)
a = SampleAssignment({"m": 0}, faults=({"after": "phase", "qubit": 0, "pauli": "Z"},))
p = sim.probability(a)
certificate = sim.compile(a)
assert certificate.probability == p
assert sum(sim.probabilities(SampleAssignment({"m": bit}) for bit in (0, 1)), ExactScalar()).real_decimal() == 1
```

`InstrumentIR(inputs, outputs, operations, initial, controls={}, name="instrument")`
is the circuit. Wires are integers; `inputs` and `outputs` fix their order.
`initial` is a normalised `StabilizerKet` on exactly the input wires.
Each operation is a dictionary with a unique `id`, a `kind` (the gate name) and `qubits`;
a measurement that should be part of the outcome also names its `record`.
Feedback is declared with `ControlNode` entries in `controls`: each names the records it reads,
a version string, and either a `ControlRule` (an explicit list of gates conditioned on record
parities) or a Python callback. A callback may add a `PhaseFeedback` declaration listing the
gates it can emit, which lets the compiler prepare them once instead of per decision.

`SampleAssignment(true, faults=(), report_flips=(), proposal_probability=1, noise_likelihood_ratio=1, sample_id="")`
is one complete outcome. `true` gives every declared record exactly once.
Each fault is `{"after": operation id, "qubit": wire, "pauli": "X" | "Y" | "Z"}`.
Report flips change what feedback sees without changing the measurement projector;
two flips on the same record cancel. The two probabilities are exact rationals used for weighting.

To add exact probabilities, start the sum from `ExactScalar()`; the integer zero that
Python's `sum` uses by default is not part of the scalar arithmetic.

## Evaluate outcomes

`Simulator(ir, options=None)` prepares the circuit on first use and again whenever the
circuit, its controls or the options change. `Options` holds the limits:
`max_quotient_rank` (default 8), `max_path_bits` (2048), `max_terms` (256),
cache sizes and batch settings. `prepare()` runs preparation without calling any callback.

| Method | Returns |
|---|---|
| `probability(assignment)` | The exact probability of that complete outcome, as an `ExactScalar` |
| `probabilities(assignments)` | An iterator of exact probabilities |
| `compile(assignment)` | A `CompiledSample`: the compiled form, its stabilizer expansion, a resource report and the exact probability |
| `compile_instantiated(branch)` | The same for a branch whose feedback gates are already written out |
| `evaluate_observable(compiled, x=0, z=0, target=None)` | The unnormalised Pauli moment, or the overlap with a target ket, on the output wires |
| `sample_one(assignment, capture_failures=False)` | A `SampleResult` with the compiled sample, its status and diagnostics |
| `compact_noise_assignments(noise, rows, reported_words)` | Immutable assignments tied to this circuit, for batching |

If a callback declared a `PhaseFeedback` family and then emits a gate outside it, the
circuit with those actual gates is compiled once more and kept.
A callback without a declaration is compiled from its actual gates from the start.
An unsupported operation or an exceeded limit raises; there is no fallback algorithm.

`SampleResult.weight_decimal()` and `log2_weight` combine the exact probability with the
noise likelihood ratio and the proposal probability. `estimate(results, score=...)` turns a
list of results into joint means and a paired ratio with standard errors; one failed result
makes the aggregate incomplete. `IndependentProposal` proposes record values uniformly;
`AffineProposal` proposes from an affine subspace and requires the caller to supply the
coverage guarantee for it. Neither changes the physical circuit.

## Draw only what a task needs

`BoundaryTask(postselection=(), outcomes=None, observables=())` lists the record parities
required for acceptance, their required values (zero by default) and the output Pauli
observables to evaluate. `BoundarySampler(ir, task, max_plans=4, **limits)` prepares the circuit
for that task and, on `sample(rng, faults=(), report_flips=())`, samples only the reported bits
that callbacks need and returns a `BoundaryDraw` with:

| Field | Meaning |
|---|---|
| `mass` | The probability of the sampled feedback history |
| `acceptance`, `error` | The accepted mass and accepted-error mass of that history, as exact scalars |
| `moments` | The requested Pauli moments |
| `reports`, `true` | The sampled reported and true record values |
| `acceptance_probability`, `error_contribution` | `acceptance / mass` and `error / mass`, the per-draw statistical contributions |

Callbacks receive only the reported bits they declared. A draw with zero acceptance is a
valid completed draw and stays in the statistics. On this interface every callback must
declare a `PhaseFeedback` family; a callback that emits a gate outside it fails preparation
or the check of the reached gate.

`NativeBoundaryTrace` is the lowest layer: it holds a prepared circuit and answers
exact event queries for packed fault assignments (`rows`, `pack`, `query`, `totals`,
`prepare_queries`). Invalid record names or Pauli effects that do not commute with the
requested rows raise `ValueError`; an exceeded limit raises `ResourceLimit`.
For automatic noise draws, importance weights and prepared task reports, use the
[file interface](../api.md#file-interface) instead.
