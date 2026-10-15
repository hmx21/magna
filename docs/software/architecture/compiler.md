# Compilation and sampling

This page explains what `compile_circuit` prepares, what one draw does with it,
what the `inspect` report means, and how the compiler handles rotation axes
that do not commute. It assumes the [overview](index.md).

## What preparation builds

```python
from random import Random
from magna import compile_circuit, parse_circuit

circuit = parse_circuit("H 0\nT 0\nH 0\nM(1/10) 0\nCX rec[-1] 1\n")
compiled = compile_circuit(circuit, task={"outputs": [1]}, ideal_state=parse_circuit(""))
report = compiled.inspect()["compilation"]
assert report["status"] == "complete"
assert compiled.queries == 0
rng = Random(7)
draws = [compiled.sample(rng) for _ in range(10)]
```

`compile_circuit` writes the whole circuit, for its fixed input, as one amplitude,
eliminates everything Clifford and every measurement that nothing reads, and expands
the remaining non-Clifford phase into a stabilizer decomposition.
It does this once for each *reference network* the task needs:

| Reference network | When it is prepared | What it computes |
|---|---|---|
| Acceptance | always | The mass of the acceptance event, with the output traced out |
| Fidelity | with an ideal target | The overlap of the accepted output with the target, after recovery |
| Sampling | with external decoders, guarded noise or `sample_records` | The distribution of the measurement bits that must be sampled, with selection and output traced |

The physical circuit and the terminal recovery are compiled together as one network.
Preparation draws no randomness and calls no decoder.

## What a draw does

Each `sample(rng)` call:

1. draws faults and report flips from the declared noise, or from the importance proposal;
2. walks the circuit in causal order and, at each external decoder call, samples the reported
   bits that decoder reads from the sampling network, given the faults and the decisions so far;
3. calls the decoder and records its decision;
4. applies all faults and decisions to the acceptance and fidelity networks as Pauli updates;
5. contracts those networks to the draw's exact acceptance and error masses.

Faults and decisions change the Pauli assignment of a prepared network, never its structure,
so the decompositions built during preparation are reused by every draw.
Exact enumeration runs the same steps for every low-weight fault pattern and every decoder
history it can reach, and sums the masses instead of averaging them.
Both keep draws with zero acceptance, and both use the same target overlap.

## Reading the inspect report

| Field | Meaning |
|---|---|
| `query_families` | The prepared reference networks, grouped by stage: `protocol` for the physical circuit (named `recovery` when the task has a terminal recovery, which is compiled together with it) and `target` for the fidelity network. Each entry names its `purpose` (`acceptance`, `fidelity`, `sampling`, or `sampling_capped` for the `affine_support` view of the sampling network), the number of phase coordinates left after elimination, its nonzero term count, the bound on its decomposition size, its preparation time, its `reference_outcomes` and which resource guarantees were established within the configured limits |
| `resource_summary` | The largest and the summed decomposition bounds over all prepared networks |
| `compiler_diagnostics` | Per stage: rotation layer sizes, which layer was kept on shared coordinates, phase-copy ancillas, Clifford gate count and path variables (see below) |
| `sampled_history`, `conditional_noise` | Whether measurement bits are sampled, and whether guarded channels are present |
| `reference_scope`, `guarded_noise_coverage` | Fixed descriptions of how the prepared networks cover later queries and guarded noise |
| `preparation_seconds` | Preparation time |

An entry's `reference_outcomes` are the measurement outcomes at which that network was prepared;
other outcomes are reached by Pauli updates.

Two kinds of resource guarantee appear in the report. The *direct* guarantee says that a
network with the same reduced phase will reuse this decomposition; the *inherited* guarantee
says that any Pauli reassignment of the same compiled circuit will. Together they are why the
report does not change from draw to draw.

## Failures

| Situation | Behaviour |
|---|---|
| A budget is exceeded during preparation | `compile_circuit` raises `ResourceLimit`, naming the option, its value and the amount needed; `run` returns `status: "failed"`, a `preparation` failure entry and zero completed draws |
| A budget or the time limit is exceeded while sampling | The run stops; completed draws are kept as `partial_statistics` |
| Unsupported operation | An exception; there is no alternative algorithm |

`magna validate` checks the file and task without building decompositions;
`inspect` completes preparation and returns the report above.
Physical draws use the declared noise exactly; importance sampling changes only the proposal
and keeps the exact likelihood ratio. See [sampling methods](../sampling.md),
[exact enumeration](../enumeration.md) and [resource options](../options.md).

## Noncommuting rotation axes

After Clifford transport, every `T`, `CS` and `CCZ` rotation acts along some Pauli axis.
When all these axes commute, they share one set of phase coordinates.
When they do not, the compiler splits them into consecutive groups of mutually commuting axes,
keeps the largest group (the earliest wins a tie) on the shared coordinates, and gives each
rotation in the other groups a phase-copy ancilla. A `CCZ` counts as seven rotation slots.
When an ideal target is attached to an already compiled circuit, the target keeps the group
the circuit chose, so that the prepared decomposition is reused.

Before contraction, dependent phase observations are replaced by independent coordinates;
the metadata that ties each rotation to its original fault location is kept separately.
Pauli sign changes therefore reuse the same compiled support and only add Clifford phases.

The report lists, per stage, `layer_slots` (the group sizes), `reference_layer`
(the zero-based index of the kept group, or `-1` without rotations), `reference_layer_slots`,
`phase_ancillas`, `shared_phase_coordinates`, `clifford_gate_count` and `path_variables`.
The `linear` and `nonlinear` route labels describe classical feedback, not rotation commutation.

For example, the two `T` gates below have noncommuting transported axes, and the selected
measurement probability is still evaluated exactly:

```python
from random import Random
from magna import compile_circuit, parse_circuit

circuit = parse_circuit("H 0\nT 0\nH 0\nT 0\nH 0\nM 0\nDETECTOR rec[-1]\nPOSTSELECT det[0]\n")
compiled = compile_circuit(circuit, task={"outputs": []})
assert compiled.sample(Random(7))["acceptance"] == 0.75
```

There is no dedicated "noncommuting axes" error. A circuit that needs more phase-copy
ancillas than the budget allows fails at a resource limit, as described above;
`options={"max_terms": 0}` on this example demonstrates that failure.
This construction does not claim that the residual resource equals the output size for every circuit.
