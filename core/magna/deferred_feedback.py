"""Literal coherent dilation of declared affine record controls.

This frontend knows no code, target state, detector pattern or injection
identity. Its record outputs must be traced (or conditioned), not coherently
summed. Opaque callbacks and nonlinear controls are explicitly unsupported.
"""
from dataclasses import dataclass
from .feedback import ControlRule
from . import _native
from .ir import InstrumentIR


@dataclass(frozen=True)
class DeferredInstrument:
    ir: InstrumentIR
    record_wires: tuple
    original_output_count: int
    report_anchors: tuple = ()


def defer_affine_feedback(ir, *, report_flips=(), expose_report_faults=False,
                          preserve_records=()):
    """Preserve fixed preparation, operation anchors and original fault sites.

    Supports Z/X measurements (including destructive), Pauli products, and authoritative
    ControlRule Pauli gates or constant even parity phases with affine guards.
    Each original CONTROL remains an identity anchor after its ordered gates.
    Report flips affect controls, never the true measurement record copies.
    No branch table, protocol-specific identity or stabilizer decomposition.
    Selected records may remain literal projectors to condition an earlier
    global query. Controls reading those records must first be resolved.
    """
    return _defer_affine_feedback(ir, InstrumentIR, report_flips=report_flips,
        expose_report_faults=expose_report_faults, preserve_records=preserve_records)


def _defer_affine_feedback_owned(ir, **options):
    """Consume a private IR, transferring retained mutable leaves to the result.

    The caller must relinquish the input. Public deferral still snapshots its
    entire output graph, preserving arbitrary annotation aliases and cycles.
    """
    return _defer_affine_feedback(ir, InstrumentIR._from_owned, **options)


def _defer_affine_feedback(ir, make_ir, *, report_flips=(), expose_report_faults=False,
                           preserve_records=()):
    ops, records, anchors = _native.defer_operations(ir.inputs, ir.outputs, ir.operations,
        tuple((oid, node.callback._native_gates() if type(node.callback) is ControlRule else None)
              for oid, node in ir.controls.items()), tuple(report_flips), expose_report_faults, tuple(preserve_records))
    converted = tuple(dict(ir.operations[op]) if type(op) is int else op for op in ops)
    result = make_ir(ir.inputs, ir.outputs + tuple(wire for _, wire in records), converted,
                     ir.initial, name=ir.name+'-deferred-affine')
    return DeferredInstrument(result, records, len(ir.outputs), anchors)
