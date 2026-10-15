# RP² cultivation

Magic-state cultivation grows a logical T state on a small code and keeps it only when
every check passes. `examples/rp2-d3` and `examples/rp2-d5` are the pinned RP² cultivation
circuits at distance 3 (25 qubits, a 238-line file) and distance 5 (61 qubits),
with physical error probability 0.001 throughout, compared against the ideal T state.

```sh
python scripts/run_task.py examples/rp2-d3 --shots 1024 --seed 7 --output rp2.json
```

| File | Contents |
|---|---|
| `physical.magna` | The pinned noisy circuit, a noiseless Clifford decoder onto one output wire, and the `POSTSELECT` acceptance condition |
| `task.json` | The output wire and the target |
| `target.magna` | `H 0` then `T 0` |
| `run.json` | Run configuration: shots, seed and resource options |

## The physical circuit

The file begins with explicit preparation and noise:

```text
RX 0 3 6 1
Z_ERROR(0.001) 0 3 6 1
RX 4
Z_ERROR(0.001) 4
R 8 5 2 7
X_ERROR(0.001) 8 5 2 7
TICK
CX 0 8 3 5 6 2 1 7
DEPOLARIZE2(0.001) 0 8 3 5 6 2 1 7
T 4
DEPOLARIZE1(0.001) 4 0 3 6 1 8 5 2 7 4
```

`RX` prepares plus states and `R` zero states, each followed by the noise channel of the
original circuit. One `CX` line applies four CNOTs. Later, corrections are controlled
directly by reported measurement bits:

```text
CX rec[-8] 1 rec[-7] 7
```

`rec[-8]` is the eighth most recent reported result at that point.
Direct record controls are affine feedback: the simulator folds them into the global amplitude
and samples no measurement bit. Inverted targets such as `M !11` follow Stim's convention.

The pinned noisy circuit is kept verbatim as the prefix of the file.
Appended to it is a noiseless Clifford decoder that maps the code's logical X and Y
to X and Y on one wire, so that the task compares a single qubit with the ideal state.
The decoder adds no noise, no measurement and no acceptance condition.

## The task

```json
{
  "outputs": [4],
  "ideal_state": "target.magna"
}
```

- Acceptance is in the circuit: it ends with `POSTSELECT` lines naming all 44 detectors.
  Four of them are written `!det[n]`, which requires the detector to be one:
  a detector's value is the raw parity of its records, and these four have noiseless parity one.
- `outputs` keeps the decoded logical wire, qubit 4; the other qubits are traced.
- The target is the T state on that wire.

Because the decoder acts on the logical operators, the success effect is the ideal T-state
projector on every accepted syndrome sector; no separate code-space check is needed.

## Run with the command line or Python

The same files work without the runner:

```sh
magna inspect examples/rp2-d3/physical.magna --task examples/rp2-d3/task.json
magna run examples/rp2-d3/physical.magna --task examples/rp2-d3/task.json --shots 1024 --seed 7 --output rp2.json
```

```python
from magna import run

result = run("examples/rp2-d3/physical.magna", task="examples/rp2-d3/task.json", shots=1024, seed=7)
assert result["status"] == "complete", result["failures"]
print(result["estimates"], result["standard_errors"])
```

`inspect` reports the route as `linear`.

## Read the result

- `estimates.acceptance` is the probability that all postselection parities match.
- `estimates.conditional_error` is the infidelity of the accepted logical T state.

At p = 0.001 failures are rare. Increase the shot count until the standard error of the
conditional error is small compared with its value, or set an importance-sampling law in
`run.json` (see [sampling methods](sampling.md#importance-sampling)).
Keep the noise model, task and precision together when comparing against published numbers.

`examples/rp2-d3-S` and `rp2-d5-S` replace T by S; they are Clifford circuits
for comparison with stabilizer samplers, not separate protocols.
