# Exact enumeration

`enumerate` sums every noise assignment with at most `max_weight` faults, each with its
exact physical probability, and reports the exact accepted and error mass within that sector
together with bounds on what was left out. There is no Monte Carlo error.

```sh
magna enumerate tests/fixtures/importance-sampling/physical.magna --task tests/fixtures/importance-sampling/task.json --max-weight 1
```

```python
from magna import enumerate_errors

result = enumerate_errors(
    "tests/fixtures/importance-sampling/physical.magna",
    task="tests/fixtures/importance-sampling/task.json",
    max_weight=1,
    max_patterns=100000,
    max_branches=4096,
)
assert result["status"] == "complete", result["failures"]
print(result["totals"])
```

The circuit, task, ideal state, recovery and decoder bindings are the same as for `run`.
Remove `sample_records` from the task; enumeration sums histories rather than sampling them.
Enumeration always uses the physical noise; a `--sampling` proposal is rejected.

## What counts as weight

Weight is the number of noise categories in which a non-identity alternative is selected.

- A measurement report flip counts as one.
- A correlated-error chain counts as one, whatever its Pauli support.
- A guarded channel counts when its alternative is selected, even in branches where the guard is off.

For each noise assignment, every reachable decoder history is evaluated exactly and summed.
With many external decisions the number of histories can grow exponentially;
`max_branches` bounds it.

## Reading the totals

| Field | Meaning |
|---|---|
| `totals.acceptance` | Exact accepted mass of the enumerated sector |
| `totals.error` | Exact accepted-and-wrong mass of the enumerated sector |
| `totals.conditional_error_within_enumerated_sector` | Their ratio |
| `totals.covered_probability` | Total physical probability of the enumerated sector |
| `by_weight` | The same quantities per weight |
| `totals.full_distribution_bounds` | Intervals for the full acceptance, error and conditional error, using the omitted mass `1 - covered_probability` |

Use the bounds, not the sector values alone, when quoting a full logical error rate.
Displayed numbers are rounded; the `exact` section holds algebraic numerators and rational
denominators for exact reconstruction. `include_patterns=True` (or `--include-patterns`)
adds every assignment with its integrated response.

## Limits

`max_patterns` bounds the number of assignments and `max_branches` the number of decoder histories.
Hitting a limit returns `status: "failed"` with a diagnostic and, where available,
explicitly labelled `partial_totals`. Invalid configuration raises an exception.
