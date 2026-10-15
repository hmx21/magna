# Tasks and targets

The circuit says what the protocol does, including when a run is accepted:
its `POSTSELECT` lines name the detectors and decoder bits that must hold.
The task says what you want to know about it: which qubits are the output and what counts as an error.
A task is a JSON object, passed as a file to the CLI or as a dictionary in Python.

```json
{
  "schema_version": 1,
  "outputs": [15],
  "ideal_state": "ideal.magna"
}
```

This task keeps qubit 15 as the output and measures infidelity against the state prepared by
`ideal.magna`; the circuit ends with `POSTSELECT det[0] det[1] det[2] det[3]`.
The complete field list is in the [schema reference](schemas.md#task-schema).

## Choose the output

`outputs` lists the physical qubit indices kept as quantum output, in order.
Omitting it keeps every qubit; `[]` keeps none and asks a purely classical question.
Qubits not listed are traced out. For encoded-state fidelity, keep the whole code block.

## Acceptance

Acceptance is part of the circuit, not the task: see [acceptance](circuit-format.md#acceptance).
A circuit without `POSTSELECT` accepts every run. To study a different acceptance rule,
edit the circuit's `POSTSELECT` lines, for instance in a copy of the file.

## Choose the error definition

One of these two.

**Ideal target.** `ideal_state` names a noiseless circuit that prepares the target from zero.
The error is the infidelity of the accepted output. Every protocol example uses this definition.

**Logical observables.** `error_observables` lists `OBSERVABLE_INCLUDE` indices;
a run is wrong when any listed observable parity is one.
This asks a purely classical question, so `outputs` and `observables` must be omitted or empty,
and it cannot be combined with `ideal_state` or `recovery`.

```json
{"schema_version": 1, "error_observables": [0]}
```

For example, the circuit `X_ERROR(1/5) 0`, `M(1/7) 0`, `OBSERVABLE_INCLUDE(0) rec[-1]`
with this task has error `2/7`: the reported bit is one when exactly
one of the data fault and the report flip occurs. Repeated records cancel in pairs,
and all `OBSERVABLE_INCLUDE` lines with the same index form one parity.
To score a decoder, add its predicted flip to the observable with a register target,
`OBSERVABLE_INCLUDE(0) ctrl[0]`; see [logical observables](circuit-format.md#logical-observables).

`validate`, `inspect` and the result describe the circuit as you wrote it: its qubits, instructions and
fingerprint. `evaluation.output_mapping` is empty, `evaluation.ideal_state` is `null`, and
`evaluation.error_metric` names the listed observables, for example
`error_observables [0]: wrong when any listed OBSERVABLE_INCLUDE parity is one`.

In either case a wrong output does not reject the draw; it contributes
accepted error mass, and `estimates.conditional_error` is the probability of error given acceptance.
Measurements that no condition reads are integrated, so a single draw's error can be fractional.

## Specify an ideal target

The simplest form is a file path, relative to the task JSON:

```json
{"schema_version": 1, "outputs": [15], "ideal_state": "ideal.magna"}
```

The file uses the ordinary circuit syntax on *local* wires `0 ... len(outputs)-1`,
which map in order to the physical `outputs`. For 15-to-1 distillation the file is
`H 0` followed by `T 0`: local wire 0 is physical wire 15.
Keep it unitary and noiseless; an empty file means the all-zero state.

A target that needs auxiliary qubits, postselection or a traced qubit uses the object form:

```json
{"circuit": "target.magna", "outputs": [0, 1], "postselection": {"2": 0}}
```

Here the target circuit acts on three local wires. Wires 0 and 1 are the target outputs,
wire 2 is projected onto zero, and any other auxiliary wire is traced.
The simulator computes the postselection norm exactly and normalises the overlap;
a zero-norm target is rejected.
A pure target gives fidelity; a target left mixed by a trace gives the normalised
density-operator overlap.
Keep auxiliary preparations as small as practical, because they add to the resource of the fidelity query.

Instead of the task JSON, `ideal_state` may be passed as an argument to `run`
or as `--ideal-state` on the command line; then paths are relative to the working directory.
Give each file in one place only.

## Add a terminal recovery

`recovery` names an optional noiseless circuit applied to the selected output
before it is compared with the target. It models an ideal final decoding round.

```json
{"schema_version": 1, "outputs": [0, 1, 2], "ideal_state": "ideal.magna", "recovery": "recovery.magna"}
```

Inside the recovery file, local wires `0 ... len(outputs)-1` hold the physical output in order;
higher-numbered wires start in zero and are discarded at the end.
It may contain measurements, resets, affine controls and registered decoders,
with its own file-local record and register numbering.
Write it with zero report-flip probability.
Its measurement outcomes are averaged, and its decoders receive their inputs from the
[history sampler](sampling.md#history-samplers), like decoders in the physical circuit.
An empty or omitted recovery is the identity. Recovery requires an ideal target.

`tests/fixtures/terminal-recovery` is a three-file example: a deterministic X fault on an
encoded T state, a recovery that measures the repetition-code syndrome and calls a decoder,
and the ideal encoded state. With recovery the infidelity is zero; without it, one.
`examples/dk` applies the same pattern to a seven-qubit code with a syndrome decoder.

Recovery changes the physical task being evaluated. When you report results, say whether one was used.

## Request raw measurement bits

`sample_records` lists absolute record indices whose sampled values are returned with each draw,
in the result's `sampled_records` field (see [results](results.md)).
Requested records are drawn by the [history sampler](sampling.md#history-samplers) before acceptance
is applied, so they are not conditioned on the run being accepted; under `affine_support` each
draw's weight corrects for the proposal.
Use this for diagnostics; it forces history sampling of those bits.
Remove it for enumeration, which sums histories instead of sampling them.

## Request output Pauli moments

`observables` lists Pauli operators on the selected outputs as `[x, z]` bit masks in output order.
The result's `moments` are the accepted joint expectations, before division by acceptance.
With a recovery, the moments refer to the state after recovery and before the target projection.

## How the pieces fit

| Question | Where it is declared |
|---|---|
| Which qubits matter | `outputs` |
| When is a run accepted | `POSTSELECT` in the circuit |
| What is an error | `ideal_state`, or `error_observables` with `OBSERVABLE_INCLUDE` in the circuit |
| What ideal decoding is assumed | `recovery` |
| Extra diagnostics | `sample_records`, `observables` |

The [results page](results.md) explains how these choices appear in the output.
