# Ideal target and optional terminal recovery

The physical circuit prepares an encoded T state and applies one deterministic X fault.
The separate recovery measures ideal repetition-code syndromes and calls a classical decoder.
The ideal file defines only the target preparation. All three use the same circuit syntax.

Install the example decoder package first:

```sh
python -m pip install ./tests/fixtures/basic/decoder_plugin
magna run tests/fixtures/terminal-recovery/physical.magna --controllers tests/fixtures/terminal-recovery/controllers.json --task tests/fixtures/terminal-recovery/task.json --shots 4
magna run tests/fixtures/terminal-recovery/physical.magna --ideal-state tests/fixtures/terminal-recovery/ideal.magna
```

Acceptance is one in both runs. Target infidelity is zero with recovery and one without it.
Target/recovery paths inside a task JSON are relative to that JSON file.
The decoder is provided by the example package installed above.
