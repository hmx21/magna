# Examples

Every directory under `examples/` is a complete protocol, packaged so that
it runs with one command and can be copied anywhere. They are the protocols of Table I of the
accompanying paper, together with Clifford counterparts used for comparison.
Three walkthroughs explain representative tasks in detail;
the table at the end lists every packaged task.

| Walkthrough | Feedback | What it shows |
|---|---|---|
| [RP² cultivation](example-rp2.md) | Affine (`rec` controls) | A pinned literature circuit with physical noise, detector postselection and a logical T target |
| [Magic-state distillation](example-distillation15.md) | Affine (`CONTROL[XOR]`) | 15-to-1 and Bravyi–Haah injection, Clifford corrections, detector acceptance, ideal targets, exact enumeration |
| [TQD memory](example-tqd.md) | External matching decoder | Long detector histories, decoder state in registers, a decoder-controlled exit and a multi-qubit encoded target |

## Running a packaged task

```sh
python scripts/run_task.py examples/bh2-T --shots 1024 --seed 7 --output bh2.json
```

`scripts/run_task.py` reads the directory's `run.json`, registers any local decoder listed in
`controllers.json`, and calls the public `run` function. It contains no protocol-specific code.
Override the shot count, seed and history sampler on the command line; `--progress-every N`
prints progress with kappa. The exit status is 0 when the result is complete.

To run a task outside the repository, copy its directory and `scripts/run_task.py`
next to an installed simulator, install the task's `requirements.txt`, and run:

```sh
python run_task.py /path/to/task --shots 1024 --seed 7 --output result.json
```

Run each task in its own Python process, because decoder module names are local to the task directory.

### Files in a task directory

| File | Role |
|---|---|
| `physical.magna` | The protocol as implemented: preparation, gates, noise, measurements, corrections and the `POSTSELECT` acceptance condition |
| `task.json` (or a named `*.task.json`) | Output qubits, ideal target and optional recovery |
| `ideal.magna`, `target.magna`, `recovery.magna` | Circuits referenced by the task |
| `controllers.json`, `decoder/` | External decoder bindings and their local code |
| `run.json` | Circuit and task file names, shots, seed, resource options and sampling method |
| `requirements.txt` | Python dependencies of the task and its decoder |

Change `shots`, `seed`, `options` or `sampling` in `run.json` to reconfigure a task;
the physical noise lives in the circuit file.

### Reading a result

1. Check `status == "complete"` and compare `completed` with `requested`.
2. `estimates.acceptance` is the probability of the declared acceptance event.
3. `estimates.conditional_error` is the error rate of accepted runs: the infidelity against the declared target.
4. Use `standard_errors` to decide whether the sample count is enough. A short run checks that a task executes; it does not have the precision of a published curve.

The [results page](results.md) explains every field.

## All packaged tasks

| Directory | Protocol | Feedback | Dependencies |
|---|---|---|---|
| `bh2-T` … `bh100-T` | Bravyi–Haah 3K+8-to-K distillation for K = 2, 4, 6, 10, 20, 40, 60, 80, 100, with K-qubit product-T targets | Affine | none |
| `bh2-S` … `bh100-S` | The same circuits with S in place of T: Clifford counterparts for comparison | Affine | none |
| `logical-distillation` | Encoded six-patch serial 15-to-1 distillation on surface-code patches with a local decoder | External | `stim`, `pymatching` |
| `rp2-d3`, `rp2-d5` | RP² cultivation at distance 3 and 5, logical T target | Affine | none |
| `rp2-d3-S`, `rp2-d5-S` | Clifford counterparts of the RP² tasks | Affine | none |
| `color3`, `color5` | Colour-code cultivation at distance 3 and 5 with readout target | Affine | none |
| `wg-T`, `wg-S` | QRM-to-surface transfer with T and S targets | Affine | none |
| `dk` | Seven-qubit colour-code preparation with a syndrome decoder in the terminal recovery | External | none |
| `tqd3`, `tqd4`, `tqd5` | TQD memory at lattice size 3 to 5 with the authors' JIT matching decoder | External | `scipy`, `pymatching`, author sources |

The `-S` variants replace every T by S so that the same circuit is Clifford; they exist
for comparisons with stabilizer samplers and are not separate protocols.
The TQD decoder is the authors' published code, which is not distributed with the tasks.
Download it once before running a TQD task:

```sh
python scripts/fetch_author_decoder.py examples/tqd3 examples/tqd4 examples/tqd5
```

The script fetches the pinned commit recorded in `decoder/author_source/provenance.json`
and checks every file against its SHA-256 hash; the decoder refuses to run if a file is missing or changed.

Small tutorial files that are not protocol tasks live in `tests/fixtures/`:
`basic` (one-qubit circuits, parity feedback, an installable majority decoder),
`importance-sampling`, `terminal-recovery`, `nonlinear-injection` (a two-qubit majority-feedback
check against an exact formula) and `distillation15` (the 15-to-1 circuit of the
[distillation walkthrough](example-distillation15.md) and a generator for other noise rates).
