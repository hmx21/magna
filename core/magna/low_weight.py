"""Bounded, exact low-weight noise enumeration through the public circuit runner.

Only original categorical noise choices are enumerated. Quantum histories are
integrated by completed global queries, including explicit classical feedback.
"""
from time import monotonic
from .file_api import SCHEMA_VERSION, compile_circuit, _ordinary
from .ir import fingerprint


def _summary(raw):
    def exact(v):
        return None if v is None else dict(coefficients_hex=[hex(x) for x in v[0]], exponent2=v[1],
                                          denominator_hex=hex(raw['denominator']))
    def number(v):
        value = _ordinary(v)
        return None if v is not None and v[0] and value == 0 else value
    values = raw['scientific']
    result = dict(
        covered_probability=number(values[0]), acceptance=number(values[1]),
        error=number(values[2]), moments=[number(v) for v in values[3:]],
        conditional_error_within_enumerated_sector=number(raw['conditional_scientific']),
        scientific=dict(covered_probability=values[0], acceptance=values[1], error=values[2],
                        moments=values[3:], conditional_error_within_enumerated_sector=raw['conditional_scientific']),
        exact=dict(covered_probability=exact(raw['numerators'][0]), acceptance=exact(raw['numerators'][1]),
                   error=exact(raw['numerators'][2]), moments=[exact(v) for v in raw['numerators'][3:]],
                   omitted_probability=dict(numerator_hex=hex(raw['omitted_numerator']),denominator_hex=hex(raw['denominator']))),
        full_distribution_bounds={name:None if value is None else [number(v) for v in value]
                                  for name,value in raw['bounds_scientific'].items()},
        full_distribution_bounds_scientific=raw['bounds_scientific'])
    if raw.get('error_normalization', ((1, 0, 0, 0), 0)) != ((1, 0, 0, 0), 0):
        value = raw['error_normalization']
        result['exact']['target_normalization'] = dict(coefficients_hex=[hex(x) for x in value[0]], exponent2=value[1])
    return result


def enumerate_errors(circuit, *, max_weight, task=None, ideal_state=None, recovery=None,
                     controllers=None, options=None, max_patterns=100000, max_branches=4096,
                     max_record_bits=64, include_patterns=False):
    """Sum original noise assignments of weight <= max_weight without sampling.

    Weight counts nonidentity selections at original mutually exclusive noise
    categories, including report flips; a correlated chain counts as one event.
    Guarded categories count latent selections (even on an inactive branch),
    exactly as in the noise model used by run(). No claim of minimum distance
    follows from a truncated sum. Omitted physical mass gives explicit bounds.
    Budgets fail explicitly; failed/partial totals are never complete estimates.
    """
    for name,value,minimum in (('max_weight',max_weight,0),('max_patterns',max_patterns,1),
            ('max_branches',max_branches,1),('max_record_bits',max_record_bits,1)):
        if type(value) is not int or value < minimum: raise ValueError(name+' must be an integer >= '+str(minimum))
    if max_patterns>1000000 or max_branches>1000000 or max_record_bits>1000000:
        raise ValueError('enumeration budget exceeds supported maximum')
    if type(include_patterns) is not bool: raise ValueError('include_patterns must be bool')
    start=monotonic()
    compiled=compile_circuit(circuit,task=task,ideal_state=ideal_state,recovery=recovery,
                             controllers=controllers,options=options)
    if compiled.task['sample_records']:
        raise ValueError('sample_records is a sampling output; remove it for exact enumeration')
    from . import _native
    result=dict(schema_version=SCHEMA_VERSION,method='exact_low_weight_enumeration',status='complete',
                max_weight=max_weight, weight_definition='nonidentity original noise-category selections, including inactive guarded selections',
                evaluation=compiled.evaluation(), noise_categories=compiled.noise_schedule.to_dict(),
                fingerprints=dict(circuit=fingerprint(compiled._source_circuit.to_text()), task=fingerprint(compiled.task['raw']),
                    controllers=fingerprint({name:b.fingerprint for name,b in compiled.circuit.controllers.bindings.items()}),
                    options=fingerprint(compiled.options)),
                limits=dict(max_patterns=max_patterns,max_branches=max_branches,max_record_bits=max_record_bits),
                completed_patterns=0, feedback_branches=0, failures=[], totals=None, by_weight=[],
                patterns=[] if include_patterns else None)
    result['route'] = compiled.route
    action_table = compiled.noise_schedule.actions
    if include_patterns:
        result['category_definitions'] = [dict(index=i, channels=[
            dict(index=j, probability=dict(numerator_hex=hex(p[0]), denominator_hex=hex(p[1])),
                 operation_id=oid, report=value if oid == 'report' else None,
                 paulis=None if oid == 'report' else value)
            for j, ((oid,value),p) in enumerate(zip(actions,probabilities))])
            for i,(actions,probabilities) in enumerate(zip(action_table,compiled.noise_schedule.probabilities))]
    cursor = None
    try:
        cursor = _native.make_file_enumeration(compiled._execution.handle, max_weight,
            max_patterns, max_branches, max_record_bits)
        state = _native.file_enumeration_state(cursor)
        result['required_patterns'] = state['required_patterns']
        while (pattern := _native.next_file_enumeration(cursor)) is not None:
            if include_patterns:
                result['patterns'].append(dict(weight=pattern['weight'],
                    choices=[dict(category=c, channel=v) for c, v in pattern['selected']],
                    probability=dict(numerator_hex=hex(pattern['probability']), denominator_hex=hex(state['denominator'])),
                    response=[None if v is None else dict(coefficients_hex=[hex(x) for x in v[0]], exponent2=v[1])
                              for v in pattern['response']]))
        totals, sectors = _native.file_enumeration_totals(cursor)
        result['totals'] = _summary(totals)
        result['by_weight'] = [dict(weight=w, patterns=count, **_summary(values)) for w, count, values in sectors]
    except (RuntimeError, ValueError, ArithmeticError, MemoryError) as exc:
        result['status'] = 'failed'; result['failures'].append(dict(type=type(exc).__name__, message=str(exc)))
        completed = cursor is not None and _native.file_enumeration_state(cursor)['completed_patterns']
        result['partial_totals'] = _summary(_native.file_enumeration_totals(cursor)[0]) if completed else None
    finally:
        if cursor is not None:
            state = _native.file_enumeration_state(cursor)
            for key in ('completed_patterns', 'feedback_branches'):
                result[key] = state[key]
        compiled._sync_execution()
    result['timings']=dict(total_seconds=monotonic()-start,query_seconds=compiled.kernel_seconds,
                           compile_seconds=compiled.compile_seconds,decoder_seconds=compiled.decoder_seconds)
    result['resources']=dict(queries=compiled.queries,compilations=compiled.compilations)
    return result
