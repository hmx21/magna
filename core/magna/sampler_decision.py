"""Explicit pilot comparison; it never configures or starts a production run."""
from . import _native
from .file_api import run


def compare_samplers(circuit, *, pilot_shots, seed=0, task=None, options=None,
                     controllers=None, ideal_state=None, recovery=None, sampling=None):
    """Run pilot_shots per channel and report warm-time Gamma and preparation costs."""
    from .file_api import CompiledCircuit
    if isinstance(circuit, CompiledCircuit):
        raise TypeError('compare_samplers compiles the task once per history sampler; pass the circuit or its path, not a compiled circuit')
    if type(pilot_shots) is not int or pilot_shots < 2:
        raise ValueError('pilot_shots must be an integer of at least two draws per channel')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    pilots = {}
    for method, pilot_seed in (('affine_support', seed), ('exact_marginal', seed+1)):
        configuration = dict(options or {}, history_sampler=method)
        pilots[method] = run(circuit, shots=pilot_shots, seed=pilot_seed,
            task=task, options=configuration, controllers=controllers,
            ideal_state=ideal_state, recovery=recovery, sampling=sampling)
    capped, born = pilots['affine_support'], pilots['exact_marginal']
    result = dict(schema_version=1, status='inconclusive', pilot_shots_per_channel=pilot_shots,
                  pilots=pilots, gamma=None, recommended_sampler=None,
                  automatic_switching=False,
                  interpretation='Pilot point estimate for matched fidelity precision; excludes preparation. '
                  'Inspect preparation and resource costs separately. No production run is started.')
    if any(p['status'] != 'complete' for p in pilots.values()):
        result.update(status='failed', reason='pilot_incomplete')
    elif capped['efficiency']['kappa'] is None:
        result['reason'] = capped['efficiency']['status']
    else:
        try:
            decision = _native.decide_file_sampler(capped['efficiency']['kappa'],
                capped['timings']['sampling_seconds'], born['timings']['sampling_seconds'], pilot_shots)
        except ValueError as exc:
            result['reason'] = str(exc)
        else:
            result.update(decision, status='complete', kappa=capped['efficiency']['kappa'])
    return result
