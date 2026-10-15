"""Ideal fixed-zero-input target preparation, using the ordinary file grammar.

The inverse preparation is a compact effect boundary. The shared native engine
contracts it jointly with the actual output; no dense state or Pauli expansion
is constructed. Literal lowering is cached here, native query plans by the runner.
"""


class IdealState:
    def __init__(self, circuit, width, *, outputs=None, postselection=None):
        from . import _native
        from .circuit_file import _circuit_handle
        self.circuit = circuit
        self.width = width
        self.outputs = None if outputs is None else tuple(outputs)
        self.postselection = tuple((int(q), bit) for q, bit in (postselection or {}).items())
        self.preparation_width = width if outputs is None else circuit.num_qubits
        self._native_handle, self.inverse = _native.make_ideal_target(
            _circuit_handle(circuit), width, self.outputs, self.postselection)

    def to_dict(self):
        return dict(source=self.circuit.source, fingerprint=self.circuit.fingerprint,
                    width=self.width, input='fixed all-zero', noise=False,
                    preparation_width=self.preparation_width, outputs=self.outputs,
                    postselection=dict(self.postselection),
                    method='native joint contraction with inverse-preparation target boundary')
