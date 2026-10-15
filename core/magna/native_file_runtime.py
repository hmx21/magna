"""Language handles and explicitly installed callbacks for the native file runtime."""
from functools import partial
import json
from pathlib import Path

from . import _native
from .circuit_file import _circuit_handle


def noise_data(noise):
    if noise is None:
        return None
    actions = getattr(noise, '_actions', None)
    if actions is None:
        actions = tuple((oid, tuple((gate['kind'], gate['qubits'][0]) for gate in gates))
                        for oid, gates in noise.gates.items())
    return actions, noise.report_flips, noise.likelihood_ratio


class NativeFileExecution:
    def __init__(self, facade):
        def callbacks(metadata):
            return () if metadata is None else tuple(partial(binding.invoke, source=inst.source)
                for inst, binding, _, _ in metadata.callbacks)
        bindings = dict(protocol=callbacks(facade._control_metadata),
                        recovery=callbacks(facade._recovery_control_metadata))
        self.handle = _native.make_file_execution(_circuit_handle(facade.circuit),
            None if facade.recovery is None else _circuit_handle(facade.recovery.circuit),
            None if facade.ideal_state is None else facade.ideal_state._native_handle,
            facade.task,
            facade.noise_schedule._native_plan,
            facade.options, json.loads((Path(__file__).parent/'data/phase_decompositions.json').read_text()), bindings)

    def prepare(self):
        _native.prepare_file_execution(self.handle)

    def state(self, details=True):
        return _native.file_execution_state(self.handle, details)

    def configure(self, options):
        _native.configure_file_execution(self.handle, options)

    def begin_run(self):
        _native.begin_file_execution_run(self.handle)

    def sample(self, rng, noise=None):
        return _native.sample_file_execution(self.handle, rng, noise_data(noise))
