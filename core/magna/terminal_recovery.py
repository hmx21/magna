"""Optional noiseless output recovery in the ordinary circuit-file grammar.

Local wires 0..width-1 are the actual selected output in its declared order.
Additional local wires are fresh zero ancillas, traced after the recovery.
Local records/controllers never alias those of the physical protocol.
"""


class TerminalRecovery:
    def __init__(self, circuit, width):
        from . import _native
        from .circuit_file import _circuit_handle
        self.circuit = circuit
        self.width = width
        _native.validate_terminal_recovery(_circuit_handle(circuit))

    def to_dict(self):
        return dict(source=self.circuit.source, fingerprint=self.circuit.fingerprint,
                    width=self.width, ancillas=max(0, self.circuit.num_qubits-self.width),
                    num_records=self.circuit.num_records, route=self.circuit.analysis.route,
                    controllers=self.circuit.controllers.to_dict(),
                    noise=False, acceptance='unchanged; all terminal outcomes are averaged')
