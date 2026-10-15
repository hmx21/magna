# Installed file-interface examples

Install the repository package with `python -m pip install .`.
Then run from the repository root:

```sh
magna run tests/fixtures/basic/t_interference.magna --task tests/fixtures/basic/t_interference.task.json --shots 8 --seed 7
magna run tests/fixtures/basic/parity_feedback.magna --task tests/fixtures/basic/parity_feedback.task.json --shots 16 --seed 7
```

The T interference accepted mass is `(2 + sqrt(2)) / 4`.
The parity example applies one correction and one unconditional Pauli channel;
its expected acceptance over noise is 0.99.

The external-function example requires its separately installed classical package:

```sh
python -m pip install ./tests/fixtures/basic/decoder_plugin
magna run tests/fixtures/basic/majority_feedback.magna --controllers tests/fixtures/basic/controllers.json --task tests/fixtures/basic/majority_feedback.task.json --shots 16 --seed 7
```

The majority function returns a bit. Its controlled quantum gate is in the circuit.
The reported error event is the final measurement equaling one, with expectation 0.5.

The same controller can define nonlinear acceptance without extra quantum wires:

```sh
magna enumerate tests/fixtures/basic/nonlinear_postselection.magna --controllers tests/fixtures/basic/controllers.json --task tests/fixtures/basic/nonlinear_postselection.task.json --max-weight 0
magna run tests/fixtures/basic/nonlinear_postselection.magna --controllers tests/fixtures/basic/controllers.json --task tests/fixtures/basic/nonlinear_postselection.task.json --shots 1000 --seed 7
```

Here the majority inputs are two fair reported bits a,b and a fixed zero, giving `a AND b`.
`POSTSELECT ctrl[0]` accepts three of four histories, while the output is the computational-basis state labelled by a.
Exact enumeration gives acceptance `3/4`, accepted-error mass `1/4`, conditional error `1/3`
and accepted joint Z moment `1/4`. Sampling estimates these quantities using every attempted draw,
including the rejected history a=b=1.
