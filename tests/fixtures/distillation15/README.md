# RM15 file-interface example

See the [user walkthrough](../../../docs/software/example-distillation15.md) for wire roles,
literal injection, affine feedback, acceptance and target preparation.
The supplied files run with the installed simulator and need no external decoder.

`code.json` retains the RM15 rows, Clifford correction and encoder gate sequence
from the revision and source paths named in that file.
`generate.py` writes the corresponding physical gate sequence as a circuit file.
The tutorial declares independent Z noise on each prepared magic resource, with all other operations ideal.

Regenerate the supplied p=0.001 instance from the repository root:

```sh
./scripts/python.sh -m tests.fixtures.distillation15.generate --output tests/fixtures/distillation15
```

Use a different output directory and `--resource-error 0` for the noiseless circuit.
The noiseless acceptance is 1 and normalized T-target infidelity is 0.
Run the example checks with:

```sh
./scripts/python.sh -m unittest tests.fixtures.distillation15.test_example
```
