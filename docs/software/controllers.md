# Classical feedback and decoders

Classical feedback reads reported measurement bits, computes a decision, and applies gates.
Parities are built in. Anything else, from a majority vote to a matching decoder,
is a Python function that you register under the name used in the circuit.
As in Stim, a decoder reads detectors: declare a `DETECTOR` for each bit it needs.

```text
CONTROL[XOR] rec[-3] rec[-1] -> ctrl[0]
DETECTOR rec[-2]
CONTROL[recovery] ctrl[0] det[-1] -> ctrl[1] ctrl[2]
CX ctrl[1] 0
CZ ctrl[2] 0
```

The function `recovery` receives the two listed bits in order, the register and then the detector,
and returns two bits, which land in `ctrl[1]` and `ctrl[2]`.
The [circuit format](circuit-format.md#computing-control-bits) page lists the syntax.

## Write a decoder

A decoder has the signature `decode(bits, config)` and returns a sequence of zeros and ones,
one per declared output register.

```python
def majority(bits, config):
    return (int(sum(bits) >= 2),)
```

Rules the function must follow:

- **Deterministic.** The same inputs and configuration must give the same outputs.
  Put tie-breaking rules, random seeds and algorithm parameters in `config`.
- **Stateless across draws.** Each draw starts from the circuit's initialisation.
  Carry per-draw state through explicit `ctrl` inputs and outputs, as the example below does.
  Immutable data such as geometry or a matching graph may be cached per configuration.
- **Observations only.** The inputs are reported bits and earlier decisions.
  Which faults occurred is never an input.
- **Exact output shape.** A wrong number of outputs or a non-bit value is an error that fails the run.

## Register a function in Python

```python
from magna import ControllerRegistry, parse_circuit, run

def majority(bits, config):
    return (int(sum(bits) >= 2),)

registry = ControllerRegistry()
registry.register("majority", majority, version="1")
circuit = parse_circuit(
    "H 0 1 2 3\n"
    "M 0 1 2\n"
    "DETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    "CONTROL[majority] det[-3] det[-2] det[-1] -> ctrl[0]\n"
    "CCX ctrl[0] 3 4\n",
    controllers=registry,
)
result = run(circuit, task={"outputs": [3, 4]}, shots=16, seed=7)
print(result["status"])
```

The `version` string is recorded in the result's fingerprints so that runs identify the decoder used.

## Register functions from a JSON file

The command line takes `--controllers controllers.json`:

```json
{
  "schema_version": 1,
  "bindings": {
    "recovery": {
      "import": "my_decoder.recovery:decode",
      "version": "1.0",
      "config": {"tie_break": 0}
    }
  }
}
```

`import` is `package.module:function`; the module must be importable from the Python environment
that runs the simulator, either installed or on `PYTHONPATH`.
`load_controllers(path)` builds the same registry in Python and imports every module when it loads the file.
Every non-built-in name in the circuit needs a binding; a missing one is reported when the circuit is loaded.

The repository ships a tiny installable decoder package for the examples:

```sh
python -m pip install ./tests/fixtures/basic/decoder_plugin
magna run tests/fixtures/basic/majority_feedback.magna --controllers tests/fixtures/basic/controllers.json --task tests/fixtures/basic/majority_feedback.task.json --shots 16 --seed 7
```

## Decoder state and detector histories

Pass detector bits with `det[n]` and carry state in registers:

```text
H 0 1
M 0 1
DETECTOR rec[-2]
DETECTOR rec[-1]
CONTROL[XOR](0) -> ctrl[0]
CONTROL[majority] det[0] det[1] ctrl[0] -> ctrl[0]
CX ctrl[0] 2
M 2
DETECTOR rec[-1]
CONTROL[majority] det[0] det[2] ctrl[0] -> ctrl[1]
CX ctrl[1] 3
POSTSELECT det[0] det[1]
```

The first call reads two detectors and an initialised state bit and writes the state bit back in place.
The second call reads the updated state.
The [TQD example](example-tqd.md) applies this pattern to a matching decoder with long syndrome histories.

## Accept or reject on a decoder bit

`POSTSELECT ctrl[k]` requires the bit to be zero and `POSTSELECT !ctrl[k]` requires it to be one:

```text
POSTSELECT !ctrl[7]
```

The decoder still runs for rejected histories, and the rejected draws remain in the statistics
with zero accepted mass. Acceptance and error are independent: `POSTSELECT` defines what counts
as accepted, and the [task](tasks.md#choose-the-error-definition) defines what counts as wrong.
To score a decoder's prediction, add its register to a logical observable,
`OBSERVABLE_INCLUDE(0) rec[-1] ctrl[0]`, and list that observable in `error_observables`.

A final readout decoder may consume the whole detector history, the final reported readout
and its earlier decisions. Declare the final readout bits as detectors when detector parities alone
do not determine the logical outcome.

## How decoders are scheduled

Before an external function runs, the selected [history sampler](sampling.md#history-samplers)
draws exactly the reported bits it reads, given the faults and all earlier decisions:
`exact_marginal` from their exact conditional distribution, and the default `affine_support`
from a proposal whose draws are then reweighted exactly.
Affine feedback (`XOR`, `NOT`, direct `rec` controls) needs no sampling and is folded into the amplitude.
If a decoder's contribution cancels symbolically and only a parity remains, the task is classified as linear.

A decision only switches the gates it controls; the prepared task is reused for every decision.
A decoder that fails to restore the ideal state does not abort the draw; the error metric records the outcome.
Final postselection and the ideal-target comparison never condition the bits a decoder saw.

`magna inspect` lists every binding with the records it depends on, and reports the
route as `linear` or `nonlinear`.
