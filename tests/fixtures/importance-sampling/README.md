# Optional importance sampling

```sh
magna run tests/fixtures/importance-sampling/physical.magna --task tests/fixtures/importance-sampling/task.json --sampling tests/fixtures/importance-sampling/sampling.json --shots 2000 --seed 7
```

The physical X-error probability is 0.001. The configured `proposal_scale` of `100`
multiplies it to a proposal error probability of 0.1.
Exact likelihood weights restore the original model. The target is the zero state.
Expected acceptance is one and expected joint/conditional target error is 0.001.
Finite estimates fluctuate; weighted acceptance is not forcibly normalized to one.
Omit `--sampling` for ordinary physical sampling. See the user documentation's Sampling page.
