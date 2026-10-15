"""Independent textual oracle for compiler record-success lowering.

These builders append ordinary noiseless gates. Evaluation uses ideal_state.
"""
from pathlib import Path

from magna import parse_circuit

ZERO_TARGET = str(Path(__file__).resolve().parents[2] / 'tests' / 'fixtures' / 'basic' / 'zero.magna')


def quantum_readout(text, rows, *, controllers=None):
    """Retain each reported parity on a fresh zero-initialized quantum output."""
    circuit = parse_circuit(text, controllers=controllers)
    rows = tuple(tuple(row) for row in rows)
    outputs = list(range(circuit.num_qubits, circuit.num_qubits + len(rows)))
    suffix = ['# Noiseless quantum parity outputs for the declared ideal target.']
    for q, row in zip(outputs, rows):
        suffix.append(f'I {q}')
        for record in row:
            if type(record) is not int or not 0 <= record < circuit.num_records:
                raise ValueError('readout references an unknown record')
            suffix.append(f'CX rec[{record-circuit.num_records}] {q}')
    return text.rstrip()+'\n'+'\n'.join(suffix)+'\n', outputs


def annotated_readout(text, indices, *, controllers=None):
    circuit = parse_circuit(text, controllers=controllers)
    rows = []
    for index in indices:
        instructions = [i for i in circuit.instructions
                        if i.name == 'OBSERVABLE_INCLUDE' and i.args[0] == index]
        if not instructions:
            raise ValueError('unknown observable annotation')
        rows.append([t.index for i in instructions for t in i.targets])
    return quantum_readout(text, rows, controllers=controllers)
