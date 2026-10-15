# TQD memory

The TQD tasks simulate a non-Abelian topological memory: two toric codes are entangled
through a memory lattice, the memory is decoded round by round by the authors' just-in-time
matching decoder, and the encoded output is compared with a four-logical-qubit magic state.
This is the example to read for external decoders with long histories and explicit state.

`examples/tqd3`, `examples/tqd4` and `examples/tqd5` are the same protocol at lattice sizes L = 3 to 5.
Each directory contains the physical circuit, the decoder adapter with the provenance of the
pinned author sources, the controller bindings, the encoded target and the task for the positive entrance branch.

| Task | L | Memory rounds | Data and report error probability |
|---|---|---|---|
| `tqd3` | 3 | 5 | 0.01 |
| `tqd4` | 4 | 6 | 0.01 |
| `tqd5` | 5 | 8 | 0.01 |

Data faults act in every round and report flips in all but the last, matching the authors' driver;
`layout.json` records this as the `author` noise window.

```sh
python -m pip install -r examples/tqd3/requirements.txt
python scripts/fetch_author_decoder.py examples/tqd3
python scripts/run_task.py examples/tqd3 --shots 32 --seed 7 --output tqd.json
```

The decoder needs `scipy` and `pymatching`, and the authors' matching code, which is not
distributed with the tasks. `scripts/fetch_author_decoder.py` downloads its five modules
at the pinned commit from the authors' GitHub repository and checks each against the SHA-256 hash
recorded in `decoder/author_source/provenance.json`; the decoder checks the hashes again before it runs.

## Physical task

Two square toric codes start in their two-logical-qubit plus states, with green data qubits in zero.
The entrance measures the green A operators, which projects onto one of two logical branches.
The memory then runs for several rounds; in the noisy rounds, independent X and Z faults
act on data qubits and the B and A syndrome reports are flipped with the declared probabilities.
After each colour's B measurements the JIT decoder returns X corrections that the circuit applies
before that colour's A measurements. At the end a charge decoder returns Z corrections, and a
fixed exit rule turns the reported green readout into CZ corrections on the data.
The output is the two toric-code blocks, still encoded.

`layout.json` names every measurement record (`entrance.A.g.0.0`, `memory.0.B.b.1.0`, …),
records the lattice size, depth, noise probabilities and the noise window, and lists the output mapping.

## Decoder calls in the circuit

Syndrome differences are declared as detectors, and each decoder call reads detector bits
plus its own state registers. The first blue JIT call in `tqd3` (L = 3, five rounds) reads:

```text
CONTROL[jit_b_0] det[1] det[2] ... det[9] ctrl[0] ctrl[1] ... ctrl[134] -> ctrl[405] ctrl[406] ... ctrl[557]
CX ctrl[405] 3
CX ctrl[406] 6
```

It receives nine detector bits and 135 state bits (five rounds times 27 spacetime slots),
and returns 18 X-correction bits followed by the 135 updated state bits.
The correction bits control X gates on the data qubits; the state registers were initialised
earlier with `CONTROL[XOR] -> ctrl[k]`, and the next call for the same colour reads them back.
Pending decisions therefore live in explicit registers, and every draw starts from the
circuit's initialisation. Larger sizes and depths change the register counts; the files are consistent by construction.

After the last round, one charge decoder per colour reads the A-detector history and returns
Z corrections, applied with the two-target form of a conditional gate:

```text
CONTROL[charge_b] det[10] det[11] ... det[18] det[64] ... -> ctrl[2700] ctrl[2701] ...
CZ ctrl[2700] 0
CZ ctrl[2701] 1
```

Finally the `exit` function turns the reported green readout into the filling corrections on the data.

## Bindings

`controllers.json` binds every JIT step (`jit_b_0`, `jit_g_0`, `jit_r_0`, `jit_b_1`, …),
the three charge decoders and the exit function to functions in the local `decoder` package,
with the lattice size, depth, step and pinned author revision in each configuration:

```json
"jit_b_0": {
  "import": "decoder.decoder:decode",
  "version": "physical-2-d4f0918c841bb4dde632dd5010425a28f4c0e43f",
  "config": {"size": 3, "depth": 5, "step": 1, "revision": "d4f0918c841bb4dde632dd5010425a28f4c0e43f"}
}
```

The decoder sees reported observations and its own earlier decisions only.
If a reported syndrome cannot be matched, it returns no new correction; the physical
output then carries the residual error, which the target comparison measures.

## The task and the target

`branch-0.task.json` keeps the 4L² data qubits of the two blocks as outputs (36 at L = 3,
100 at L = 5) and names the target in object form. The circuit's final `POSTSELECT det[0]`
accepts when the entrance detector is zero (the positive branch, with ideal probability 5/8). At L = 3:

```json
"ideal_state": {
  "circuit": "target.magna",
  "outputs": [0, 1, ..., 35],
  "postselection": {"36": 0}
}
```

`target.magna` prepares the logical branch state with one auxiliary qubit, applies the
noiseless encoders to both blocks, and the task projects the auxiliary wire (local index 4L²)
onto zero to select the branch. The simulator computes the branch norm exactly and includes it
in the fidelity, so `estimates.conditional_error` is the normalised infidelity of the
accepted encoded output against the complete encoded target.
For the negative branch, change the circuit's final line to `POSTSELECT !det[0]` and project the auxiliary onto one.

## Run with the command line

The runner reads the resource limits from `run.json` and makes the local `decoder` package importable.
To use the plain CLI instead, write the limits to a file and run from inside the task directory
with that directory on `PYTHONPATH`, so that `decoder.decoder` in `controllers.json` can be imported:

```sh
cd examples/tqd3
printf '%s\n' '{"max_quotient_rank": 12, "max_terms": 4096}' > /tmp/tqd-options.json
PYTHONPATH=. magna inspect physical.magna --controllers controllers.json --task branch-0.task.json --options /tmp/tqd-options.json
PYTHONPATH=. magna run physical.magna --controllers controllers.json --task branch-0.task.json --options /tmp/tqd-options.json --shots 4 --seed 7
```

The route is `nonlinear`; read `status` before the estimates.
Four draws only check that the decoder bindings work; estimates need many more shots.

## What to expect

Table I of the accompanying paper lists the circuit sizes, representation ranks
and sampling times of the three tasks.
Each draw calls the decoder once per colour and round; `timings.decoder_seconds` reports
the time spent in Python. The draw loop is otherwise native and the prepared resources are
reused across draws.
