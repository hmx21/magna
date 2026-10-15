"""Compact ownership boundary for the optional whole-instrument C++ compiler.

This module serializes original IR metadata and declared control templates. It
does not construct path variables, perform elimination, normalize a tensor, or
prepare a Python CliffordSkeleton. Numerical preparation occurs in compiler.hpp.
"""


def serialize_compiler(prepared, max_path_bits):
    """Return owned native input and Python-only ID/control metadata.

    Operation row convention is (kind, qubits, exponent, record, sign,
    original_location, feedback_slot, pauli, paulis). Synthetic declared phase
    rows have no original location; the following I marker retains the original
    control's fault site after every declared phase. Actual callbacks still run
    on each assignment in the execution facade.
    """
    from . import _native
    ir = prepared.ir
    if type(max_path_bits) is not int or max_path_bits < 0:
        raise ValueError('nonnegative integer path budget required')
    k = ir.initial
    handle, metadata = _native.serialize_instrument(
        (k.width,k.base,k.columns,k.linear,k.upper,k.power,k.phase), ir.inputs, ir.outputs, ir.operations,
        tuple((oid, None if node.feedback is None else
               (node.feedback.qubits,node.feedback.odd_axes,node.feedback.even_axes))
              for oid,node in ir.controls.items()), max_path_bits)
    metadata['feedback_specs'] = {oid:node.feedback for oid,node in ir.controls.items()}
    metadata['fingerprint'] = prepared.fingerprint
    return handle, metadata


def compiler_snapshot_to_python(snapshot, metadata):
    """Decorate an explicitly requested native debug snapshot with original IDs.

    This is an export-only operation. A production prepare/evaluate path need
    not materialize the boundary matrices or fault suffixes in Python.
    """
    result = dict(snapshot)
    result['ir_fingerprint'] = metadata['fingerprint']
    result['record_names'] = metadata['record_names']
    result['operation_ids'] = metadata['operation_ids']
    if 'phase_controls' in result:
        result['phase_controls'] = tuple(None if j < 0 else metadata['feedback_slots'][j]
                                         for j in result['phase_controls'])
    if 'environment' in result:
        result['environment'] = tuple(metadata['operation_ids'][j] for j in result['environment'])
    return result
