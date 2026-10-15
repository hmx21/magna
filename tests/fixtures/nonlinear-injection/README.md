# Two-qubit nonlinear-feedback injection

`physical.magna` prepares a data plus state and a T-state ancilla,
couples them with CNOT, repeats the ancilla Z measurement three times,
and applies S to the data when a majority of the reported bits are one.
Each report flips independently with probability 0.01 in the supplied file.
`task.json` retains the data qubit and compares it with `ideal.magna`.

Run the ordinary Monte Carlo curve from the repository root:

```sh
./scripts/python.sh tests/fixtures/nonlinear-injection/run_curve.py --output build/nonlinear-injection
```

The script changes only the report-flip probability in copies of the circuit file
and calls the public `run` interface at nine predeclared probabilities.
It uses 131072 independent physical trials per point, with seeds 20260927 through 20260935.
The majority function is reused from the existing external-decoder example.
The theory is used only after simulation to compare the returned estimates.
Output files include the literal circuits, full API results, a summary, and PNG/PDF plots.
Error bars show 1.96 times the simulator's reported standard error.

For a single circuit through the CLI:

```sh
PYTHONPATH=tests/fixtures/basic/decoder_plugin ./scripts/python.sh -m magna.file_api run \
  tests/fixtures/nonlinear-injection/physical.magna \
  --task tests/fixtures/nonlinear-injection/task.json \
  --controllers tests/fixtures/basic/controllers.json \
  --shots 131072 --seed 20260927
```

No run is retained in the repository; write each run to a new output directory.
