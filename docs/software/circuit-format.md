# Circuit files

Circuits are plain text files, conventionally named `.magna`.
The grammar follows [Stim's circuit format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_stim_circuit.md)
for gates, targets, measurement lookbacks and Pauli products, and adds classical control
registers, external functions, conditional noise and in-circuit postselection.

```text
# Comments start with a hash.
R 0 1            # one instruction per line: name, then targets
H 0
CX 0 1           # several target groups on one line: CX 0 1 2 3 applies two CNOTs
T 0
REPEAT 2 {       # repeated blocks
    M 0
}
```

Qubits are nonnegative integers and start in the zero state.
Indices up to 65535 are accepted by default; raise `max_qubits` in `parse_circuit` or `load_circuit`
for larger ones. Every index below the largest one used counts toward the task's default outputs,
so prefer compact numbering or list the `outputs` explicitly in the task.
Write preparation gates at the beginning of the file to start from any other stabilizer state.

## Instructions

| Family | Instructions |
|---|---|
| One-qubit gates | `I X Y Z H S S_DAG T T_DAG SQRT_X SQRT_X_DAG SQRT_Y SQRT_Y_DAG H_XY H_YZ C_XYZ C_ZYX` |
| Two-qubit gates | `CX CY CZ SWAP ISWAP ISWAP_DAG CS CS_DAG CH` |
| Three-qubit gates | `CCZ CCX` |
| Resets | `R RX RY` |
| Measurements | `M MX MY MR MRX MRY MXX MYY MZZ MPP` |
| Noise | `X_ERROR Y_ERROR Z_ERROR DEPOLARIZE1 DEPOLARIZE2 PAULI_CHANNEL_1 PAULI_CHANNEL_2 CORRELATED_ERROR ELSE_CORRELATED_ERROR`, and `M(p)` for report flips |
| Annotations | `DETECTOR OBSERVABLE_INCLUDE QUBIT_COORDS SHIFT_COORDS TICK` |
| Classical | `CONTROL`, `POSTSELECT` |
| Structure | `REPEAT n { ... }` |

Accepted aliases are `CNOT`, `ZCX`, `ZCY`, `ZCZ`, `MZ`, `MRZ`, `RZ`, `E` and `H_XZ`.
`S_DAG`, `T_DAG` and `CS_DAG` are the inverses of `S`, `T` and `CS`.
Targets within one gate must be distinct.
`MPP X0*X1 Z2` records the `X0X1` and `Z2` measurements as two separate results.
Noise instructions are described on the [noise](noise.md) page.

## Measurement records, detectors and registers

| Reference | Meaning |
|---|---|
| `rec[-k]` | The k-th most recent reported measurement result at that point in the file |
| `det[n]` | The parity of the records listed in detector n; `det[-1]` is the latest detector |
| `ctrl[k]` | Classical register k, holding its most recent assignment |
| `!x` | The complement of any of the above |

Detector indices are absolute and zero-based, counted in the expanded order of `REPEAT` blocks.
A detector's coordinates are annotations only; its value is the parity of its records.
Unlike Stim, which reports a detector relative to its noiseless value, the value here is the raw parity,
so a detector whose noiseless parity is one reads one in a fault-free run.
Detectors are the data that decoders read and the bits that `POSTSELECT` tests.
Every reference must point to an earlier instruction.
Writing a control register does not change the measurement numbering.

## Conditional gates

Put a classical bit in the control position of a controlled gate to apply its target gate
when the bit is one:

| Instruction | Applies |
|---|---|
| `CX c q` | X on q |
| `CY c q` | Y on q |
| `CZ c q` | Z on q |
| `CS c q` | S on q |
| `CS_DAG c q` | S-dagger on q |
| `CH c q` | H on q |
| `CCZ c a b` | CZ on a and b |
| `CCX c a b` | CNOT from a to b |

Here `c` is `rec[-k]`, `det[n]` assigned to a register, or `ctrl[k]`, optionally prefixed with `!`.
An integer in that position is an ordinary quantum control instead.
For `CZ` the classical target may also be written second, as `CZ q rec[-1]`.
To apply a compound correction, write its decomposition with the same control bit on each gate.

```text
H 0 1 2
M 0
CONTROL[XOR] rec[-1] -> ctrl[0]
CCZ ctrl[0] 1 2
CCX !ctrl[0] 1 2
```

## Computing control bits

`CONTROL[name] inputs... -> outputs...` computes output registers from reported bits.
Two functions are built in:

| Function | Meaning |
|---|---|
| `CONTROL[XOR] a b c -> ctrl[k]` | Parity of the inputs. `CONTROL[XOR](1) ...` adds a constant one. With no inputs, `CONTROL[XOR] -> ctrl[k]` or `CONTROL[XOR](0) -> ctrl[k]` initialises a register to zero and `CONTROL[XOR](1) -> ctrl[k]` to one |
| `CONTROL[NOT] a -> ctrl[k]` | Complement of exactly one input; equivalent to `CONTROL[XOR](1) a` |

Any other name refers to an external Python function, a decoder, that you register.
It reads detectors and earlier decisions in registers, never raw records: its inputs are
`det[n]` and `ctrl[k]` references, each optionally complemented.
It receives the listed bits in order and returns one bit per output register.
Declare a detector for each measurement result the decoder needs:

```text
M 5 6
DETECTOR rec[-2]
DETECTOR rec[-1]
CONTROL[recovery] det[-2] det[-1] ctrl[0] -> ctrl[3] ctrl[4]
CX ctrl[3] 0
CCZ ctrl[4] 1 2
```

Repeated inputs to `XOR` cancel modulo two. `XOR` and `NOT` expressions simplify
algebraically across registers: `f XOR f XOR m` is `m`, and `NOT NOT f` is `f`.
An instruction reads all its inputs before writing its outputs, so `CONTROL[f] ctrl[0] -> ctrl[0]`
updates a register in place. Registers may be overwritten; later references see the new value.
Initialise a register before its first use.
External functions receive repeated and complemented inputs exactly as listed.
See [classical feedback](controllers.md) for registering functions.

## Acceptance

`POSTSELECT` declares when a run is accepted. Each target is a separate requirement:
`det[n]` and `ctrl[k]` must be zero, `!det[n]` and `!ctrl[k]` must be one.
A circuit without `POSTSELECT` accepts every run.
Write it at the end of the file, naming the detectors that the acceptance depends on:

```text
POSTSELECT det[0] det[1] det[2] !det[3]
```

A decoder's verdict is selected through its register:

```text
CONTROL[readout_decoder] det[0] det[1] det[2] -> ctrl[7]
POSTSELECT !ctrl[7]
```

A selection reads the register value assigned before the `POSTSELECT` line, also inside `REPEAT`;
overwriting the register later does not change the selection.
Several `POSTSELECT` lines all apply; conflicting requirements give zero acceptance.
Selection never conditions earlier decisions: decoders see the physical distribution of their inputs,
and rejected draws count as attempts.
Use `POSTSELECT` only in the physical file. The ideal state is unitary, and recovery averages its outcomes.

## Logical observables

`OBSERVABLE_INCLUDE(k)` adds measurement records and registers to logical observable k;
all lines with the same index form one parity.
A task that lists k in `error_observables` counts a run as wrong when that parity is one;
see [tasks](tasks.md#choose-the-error-definition).
A register target adds a decoder's prediction, as in a memory experiment scored against its decoder:

```text
CONTROL[decoder] det[0] det[1] det[2] det[3] -> ctrl[0]
OBSERVABLE_INCLUDE(0) rec[-1] ctrl[0]
```

A register target reads its value on that line; overwriting the register later does not change the observable.

## Format identifier

`inspect` reports the format identifier `magna-0.1` for every circuit file.
The format is versioned 0.x until the first stable release and may change between versions.

`XOR` and `NOT` are reserved function names.
Syntax errors report the source location of the offending instruction.
