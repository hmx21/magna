# Magic-state distillation

The distillation circuits share one pattern: noisy T resources are
injected into an encoded state, measurement results drive Clifford corrections,
parity checks decide acceptance, and the output is compared with an ideal product of T states.

| Files | Protocol | Resources in, T states out |
|---|---|---|
| `tests/fixtures/distillation15` | 15-to-1 Reed–Muller distillation (tutorial circuit) | 15 → 1 |
| `examples/bh2-T` … `examples/bh100-T` | Bravyi–Haah 3K+8-to-K for K = 2, 4, 6, 10, 20, 40, 60, 80, 100 | up to 308 → 100 |
| `examples/logical-distillation` | 15-to-1 on six surface-code patches with a local decoder | 15 → 1, encoded |

```sh
magna run tests/fixtures/distillation15/physical.magna --task tests/fixtures/distillation15/task.json --shots 1024 --seed 7
python scripts/run_task.py examples/bh100-T --shots 1024 --output bh100.json
```

The rest of this page walks through the 15-to-1 circuit in `tests/fixtures/distillation15`,
which is small enough to read.

## Wires and noise

| Wires | Role |
|---|---|
| 0–14 | The fifteen encoded data wires, each measured during its injection |
| 15 | The retained output wire |
| 16–30 | The fifteen T resources, measured in X after injection |

Every wire starts in zero. The circuit first prepares the encoded Bell pair with
an explicit Clifford gate sequence. The only noise is an independent `Z_ERROR(1/1000)`
after each resource's `H; T` preparation; the encoder, injections, readout and corrections are noiseless.
To study gate or measurement noise, add the channels at their physical locations.

## Injection with affine feedback

Each injection reads:

```text
H 16
T 16
Z_ERROR(1/1000) 16
CX 16 0
M 0
CX rec[-1] 16
CONTROL[XOR](1) rec[-1] -> ctrl[0]
CS_DAG ctrl[0] 16
S 16
```

The data wire is measured; the reported bit controls X on the resource, its complement
(`CONTROL[XOR](1)` adds a constant one) controls S-dagger, and an unconditional S fixes the
output convention. The logical data now lives on wire 16.
If report errors were added with `M(p)`, the reported bits would drive these corrections.

After the resources are measured in X, four code parities become detectors and
the logical parity controls the final correction on the output:

```text
MX 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
DETECTOR rec[-15] rec[-10] rec[-9] rec[-6] rec[-5] rec[-4] rec[-3] rec[-1]
DETECTOR rec[-14] rec[-11] rec[-9] rec[-7] rec[-5] rec[-4] rec[-2] rec[-1]
DETECTOR rec[-13] rec[-12] rec[-9] rec[-7] rec[-6] rec[-3] rec[-2] rec[-1]
DETECTOR rec[-13] rec[-11] rec[-10] rec[-8] rec[-5] rec[-3] rec[-2] rec[-1]
CONTROL[XOR] rec[-13] rec[-11] rec[-10] rec[-7] rec[-6] rec[-4] rec[-1] -> ctrl[1]
CZ ctrl[1] 15
```

All feedback is affine, so the task runs on the linear route and no measurement is sampled.

## The task

```json
{
  "schema_version": 1,
  "outputs": [15],
  "ideal_state": "ideal.magna"
}
```

The circuit ends with `POSTSELECT det[0] det[1] det[2] det[3]`: acceptance requires all four detectors to be zero. The output is wire 15;
in `ideal.magna` it is local wire 0, prepared as the T state by `H 0` then `T 0`.
`estimates.conditional_error` is therefore the infidelity of the accepted output T state.

## Enumerate instead of sampling

The output error of this protocol comes from rare combinations of resource faults,
so exact enumeration is a natural tool:

```sh
magna enumerate tests/fixtures/distillation15/physical.magna --task tests/fixtures/distillation15/task.json --max-weight 3
```

This sums every pattern of up to three resource faults exactly, in under a second.
Up to weight 2 the enumerated error is exactly zero, because the code detects any two
resource faults. The weight-3 sector contributes an accepted error of about 3.5e-8,
which is the familiar leading term 35 p³ of this protocol, while the omitted mass of
higher-weight patterns is about 1.4e-9. `totals.full_distribution_bounds` turns that omitted
mass into intervals for the full acceptance and error.
Raise `--max-weight` to tighten them. See [exact enumeration](enumeration.md).

## Bravyi–Haah and encoded tasks

The `bh*-T` tasks use the same injection pattern on the 3K+8 code with K output wires,
and an `ideal.magna` that prepares K T states. Their `-S` counterparts replace T by S
to give Clifford circuits for comparison. Resource noise is again Z with probability 0.001.

`examples/logical-distillation` runs the protocol on surface-code patches:
each resource is encoded, depolarised with a per-qubit probability, and injected
with reported logical readout and a conditional S-dagger on the working patch.
Its decoder is a local Python module listed in `controllers.json`, so the task needs
`stim` and `pymatching` from its `requirements.txt`. The output is the complete
encoded patch, compared with its complete encoded ideal state.

## Other noise rates

`tests/fixtures/distillation15` also contains the generator of the 15-to-1 circuit:

```sh
python -m tests.fixtures.distillation15.generate --resource-error 1/100 --output /tmp/rm15-p01
```

With `--resource-error 0` the acceptance is exactly 1 and the infidelity exactly 0.
