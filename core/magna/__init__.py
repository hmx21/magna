"""Simulate noisy circuits with non-Clifford gates, measurements, Pauli noise and classical feedback.

The top level is the file interface. The operation-level interface, which evaluates
fully specified outcomes of circuits written as Python values, is ``magna.operations``.
"""
try:
    from . import _native  # noqa: F401  (fail early with a build hint)
except ImportError as exc:
    from pathlib import Path as _Path
    _built = any(_Path(__path__[0]).glob('_native*'))
    raise ImportError(
        f"magna's compiled extension {'failed to load' if _built else 'is missing'} in {__path__[0]}. "
        "In a source checkout, build it with './scripts/python.sh scripts/build_native_core.py'; "
        "otherwise install the package with 'python -m pip install .' and run plain 'python'."
    ) from (exc if _built else None)
from ._native import ResourceLimit
from .circuit_file import CircuitFile, CircuitSyntaxError, ControllerRegistry, SourceLocation, load_circuit, parse_circuit
from .file_api import CompiledCircuit, compile_circuit, load_controllers, run
from .low_weight import enumerate_errors
from .sampler_decision import compare_samplers

__all__ = ['load_circuit', 'parse_circuit', 'load_controllers', 'compile_circuit', 'run', 'enumerate_errors',
           'compare_samplers', 'CompiledCircuit', 'CircuitFile', 'ControllerRegistry', 'ResourceLimit',
           'CircuitSyntaxError', 'SourceLocation']
