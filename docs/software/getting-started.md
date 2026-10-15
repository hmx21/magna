# Tutorial

This tutorial takes one small circuit from an exact probability to a noisy,
feedback-controlled, target-compared estimate, shows the two other estimators on the
same files, and ends by running one of the packaged protocols.
Every command runs from the repository root after [installation](install.md).

## 1. An exact probability

Save this as `example.magna`:

```text
R 0
H 0
T 0
H 0
M 0
DETECTOR rec[-1]
POSTSELECT det[0]
```

Qubits start in zero; `R` makes that explicit. `H T H` rotates the qubit, `M` measures it
in the Z basis, and `DETECTOR rec[-1]` gives the latest measurement result a name, detector 0.
`POSTSELECT det[0]` accepts a run only when detector 0 is zero, that is, when the measurement returned zero,
so the acceptance is the probability of measuring zero.
Now validate, inspect and run:

```sh
magna validate example.magna
magna inspect example.magna
magna run example.magna --shots 8 --seed 7 --output result.json
```

- `validate` checks the syntax.
- `inspect` compiles the circuit and reports what will be computed and how much resource it needs.
- `run` evaluates it, prints a short summary and, with `--output`, writes the full JSON result.

No task file is needed yet: by default every qubit is output and no error is defined.
The steps below add one.

The summary starts with `run: complete, 8 of 8 draws, seed 7` and an acceptance of 0.853553.
In `result.json`, `estimates.acceptance` is `0.8535533905932737`, which is `(2 + sqrt(2)) / 4`.
`standard_errors.acceptance` is zero: the circuit has no noise and no decoder,
so every draw evaluates the same exact amplitude.
`requested` and `completed` still report eight draws.
To receive the individual measurement bits instead of integrating over them,
pass a task `{"schema_version": 1, "sample_records": [0]}`.

## 2. The same run from Python

```python
from magna import load_circuit, compile_circuit, run

circuit = load_circuit("example.magna")
compiled = compile_circuit(circuit)
result = run(compiled, shots=8, seed=7)
print(result["status"], result["estimates"]["acceptance"])
```

`compile_circuit` prepares the circuit and its task once without drawing anything; `compiled.inspect()`
returns the same report as the `inspect` command. Pass the compiled circuit to `run` as often as you like,
for instance with different seeds. `run` also accepts a path or a loaded circuit, with an optional
`task=...`, which compiles and runs in one call.

## 3. Add noise

Insert a bit-flip channel before the measurement:

```text
R 0
H 0
T 0
H 0
X_ERROR(0.01) 0
M 0
DETECTOR rec[-1]
POSTSELECT det[0]
```

Run it with a few thousand shots. The acceptance estimate moves to about 0.8465,
and `standard_errors.acceptance` is now nonzero because the fault is drawn at random.
Each draw either applies the X fault or not; the amplitude for that draw is still exact.
Noise channels are written where they physically act, with Stim's names and probability
conventions; [noise](noise.md) lists the supported channels.

## 4. Add classical feedback

This circuit measures three qubits, computes their parity, and applies a correction
to a fourth qubit when the parity is one:

```text
R 0 1 2 3
H 0 1 2
M 0 1 2
CONTROL[XOR] rec[-3] rec[-2] rec[-1] -> ctrl[0]
CX ctrl[0] 3
X_ERROR(0.01) 3
M 3
DETECTOR rec[-4] rec[-3] rec[-2] rec[-1]
POSTSELECT det[0]
```

`rec[-k]` is the k-th most recent measurement result. `CONTROL[XOR]` computes the parity
of its inputs into the control register `ctrl[0]`, and `CX ctrl[0] 3` applies X to qubit 3
when that bit is one. The detector is the parity of all four results, which is zero unless
the fault fires, so the acceptance is 0.99.

This file is `tests/fixtures/basic/parity_feedback.magna`:

```sh
magna run tests/fixtures/basic/parity_feedback.magna --shots 1000 --seed 7
```

Feedback through `XOR` is affine, so the simulator folds it into the global amplitude
and never needs to sample the three measurement bits.
A decoder that is not affine, such as a majority vote or a matching decoder, is a Python
function you register by name. Like a Stim decoder it reads detectors:
`CONTROL[majority] det[0] det[1] det[2] -> ctrl[0]`; see [classical feedback](controllers.md).

## 5. Compare the output with an ideal state

The files in `tests/fixtures/importance-sampling` define the smallest possible target task.
The physical circuit is a single bit-flip channel:

```text
X_ERROR(1/1000) 0
```

The ideal state file `ideal.magna` prepares the target from zero; here it is just `I 0`,
the zero state itself. The task names that file:

```json
{"schema_version": 1, "ideal_state": "ideal.magna"}
```

With an `ideal_state`, the error metric becomes the infidelity between the physical output
and the ideal state. Exactly: acceptance is 1, the error is 0.001.

```sh
magna run tests/fixtures/importance-sampling/physical.magna --task tests/fixtures/importance-sampling/task.json --shots 2000 --seed 7
```

The estimate of `estimates.error` fluctuates around 0.001, and with two thousand draws
only about two of them contain the fault, so the standard error is large.
The next two steps address exactly this problem.

## 6. Enumerate low-weight faults exactly

```sh
magna enumerate tests/fixtures/importance-sampling/physical.magna --task tests/fixtures/importance-sampling/task.json --max-weight 1
```

Enumeration sums every fault pattern with at most one fault, with its exact probability.
Here that covers the whole distribution: `totals.acceptance` is 1, `totals.error` is 0.001
and `totals.covered_probability` is 1. For larger circuits the omitted mass is reported and
turned into bounds; see [exact enumeration](enumeration.md).

## 7. Sample from a scaled proposal

Importance sampling multiplies every eligible error probability by a common factor
and corrects each draw by its exact likelihood ratio. `sampling.json` in the same directory
contains `{"method": "importance", "proposal_scale": "100"}`:

```sh
magna run tests/fixtures/importance-sampling/physical.magna --task tests/fixtures/importance-sampling/task.json --sampling tests/fixtures/importance-sampling/sampling.json --shots 2000 --seed 7
```

Now about a tenth of the draws contain the fault, and the standard error of the error estimate
drops accordingly while the estimate stays unbiased. [Sampling methods](sampling.md)
explains the weights and how to choose the scale.

## 8. Run a packaged protocol

Each directory under `examples/` is a complete, self-contained protocol task:
circuit, target, task, resource options, seed and any decoder code.
The universal runner executes one of them through the same public interface:

```sh
python scripts/run_task.py examples/bh2-T --shots 256 --seed 7 --output bh2.json
```

This runs 14-to-2 Bravyi–Haah magic-state distillation with 0.001 resource noise and reports the
joint infidelity of the two accepted outputs against two ideal T states.
`run.json` in the directory records the shot count, seed and resource budgets; the command-line
flags override shots, seed and the history sampler.
Copy a task directory together with `scripts/run_task.py` next to an installed simulator to run it anywhere.
The [examples](examples.md) describe the packaged protocols.

## 9. Beyond files: an exact outcome probability

The file interface is the normal way to work. For a single fully specified outcome,
the operation-level API evaluates its exact probability directly:

```python
from magna.operations import InstrumentIR, SampleAssignment, Simulator, StabilizerKet

circuit = InstrumentIR(
    inputs=(0,), outputs=(0,), initial=StabilizerKet.positive(1),
    operations=(
        {"id": "h0", "kind": "H", "qubits": [0]},
        {"id": "t0", "kind": "T", "qubits": [0]},
        {"id": "h1", "kind": "H", "qubits": [0]},
        {"id": "m0", "kind": "MZ", "qubits": [0], "record": "m"},
    ),
)
print(Simulator(circuit).probability(SampleAssignment(true={"m": 0})).real_decimal())
```

The printed value is the same `(2 + sqrt(2)) / 4` as in step 1.
See [the operation-level API](architecture/operations.md) when you need this level of control.

## Where to go next

- [Concepts](concepts.md) defines acceptance, error and conditional error precisely, and explains
  which parts of a run are exact and which are sampled.
- The [examples](examples.md) walk through RP² cultivation, magic-state distillation
  and a TQD memory with an external matching decoder.
