"""Immutable expression indices; fresh causal values for every physical draw."""
from types import MappingProxyType


class ControlMetadata:
    def __init__(self, circuit):
        from . import _native
        from .circuit_file import _circuit_handle, Target
        self._native_handle, terms, calls = _native.make_file_controls(_circuit_handle(circuit))
        self.index = MappingProxyType({Target(*term): j for j, term in enumerate(terms)})
        self.callbacks = tuple((circuit.instructions[index],
            circuit.controllers.resolve(circuit.instructions[index].control_name), inputs, outputs)
            for index, inputs, outputs in calls)
