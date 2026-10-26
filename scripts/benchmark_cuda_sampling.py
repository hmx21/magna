"""Compare warm original magna.run and CUDA sampling on one portable task.

Both backends attempt the same total shots with the same declared noise/proposal.
GPU batch size controls chunking only. CPU MT and GPU Philox seeds do not generate
identical histories; timings are not a same-random-stream correctness test.
"""
import argparse
import statistics
from pathlib import Path
from time import perf_counter

from run_cuda_task import (add_task_arguments, base_report, emit,
                           load_task, measure, record_gpu, sha256, verify_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_task_arguments(parser)
    parser.add_argument('--repeat', type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('repeat must be positive')
    report = dict(schema_version=1, status='failed', task=str(args.task))
    try:
        from magna import compile_circuit, run
        from magna.cuda_sampler import CudaSampler
        circuit, settings, config, shots, seed = load_task(args)
        if seed+args.repeat-1 >= 2**64:
            raise ValueError('the last repeat seed must fit an unsigned 64-bit integer')
        start = perf_counter()
        compiled = compile_circuit(circuit, **settings)
        compile_seconds = perf_counter()-start
        start = perf_counter()
        sampler = CudaSampler(compiled)
        plan_seconds = perf_counter()-start
        report = base_report(args, config, settings, compiled, circuit)
        report['runner_sha256'][Path(__file__).name] = sha256(__file__)
        report.update(shots_per_call=shots, batch_size=args.batch_size, repeats=args.repeat, first_seed=seed,
                      preparation=dict(cpu_compile_seconds=compile_seconds, backend_plan_seconds=plan_seconds),
                      warmup={}, trials=[],
                      comparison=dict(cpu='Original magna.run with its unchanged default execution settings',
                                      gpu='CudaSampler.run with explicit GPU chunk size',
                                      random_streams='CPU mt19937_64 via the public seed interface; GPU Philox4x32-10. Equal numeric seeds do not imply equal noise histories.',
                                      scope='Warm complete API wall time, including GPU device-wide completion. Preparation/JIT/warmup are separate. This is not a cold-start or best-multicore-CPU comparison.'))
        _, cpu_warm = measure(lambda: run(compiled, shots=shots, seed=seed), shots)
        report['warmup']['cpu_complete_call_seconds'] = cpu_warm
        start = perf_counter()
        measure(lambda: sampler.run(shots, seed=seed, batch_size=args.batch_size), shots)
        cp = sampler.noise.cp
        sync = cp.cuda.runtime.deviceSynchronize
        sync()
        report['warmup']['gpu_complete_call_seconds'] = perf_counter()-start
        record_gpu(report, cp)
        for repeat in range(args.repeat):
            trial_seed = seed+repeat
            order = ('cpu', 'gpu') if repeat % 2 == 0 else ('gpu', 'cpu')
            trial = dict(repeat=repeat, seed=trial_seed, order=order, runs={})
            report['trials'].append(trial)
            for backend in order:
                if backend == 'cpu':
                    result, wall = measure(lambda: run(compiled, shots=shots, seed=trial_seed), shots)
                else:
                    # A long CPU call can leave the GPU idle. Rewarm this exact
                    # workload before each timed GPU call for consistent warm
                    # throughput; retain its cost separately from the ratio.
                    _, warm_wall = measure(lambda: sampler.run(shots, seed=trial_seed,
                                          batch_size=args.batch_size), shots, sync)
                    trial['gpu_same_N_warmup_seconds'] = warm_wall
                    result, wall = measure(lambda: sampler.run(shots, seed=trial_seed,
                                          batch_size=args.batch_size), shots, sync)
                trial['runs'][backend] = dict(completed_call_wall_seconds=wall,
                                               internal_sampling_seconds=result['timings']['sampling_seconds'],
                                               attempted_draws_per_second=shots/wall, result=result)
        summaries = {}
        for backend in ('cpu', 'gpu'):
            times = [t['runs'][backend]['completed_call_wall_seconds'] for t in report['trials']]
            median = statistics.median(times)
            summaries[backend] = dict(median_wall_seconds=median, min_wall_seconds=min(times),
                                      max_wall_seconds=max(times), attempted_draws_per_second=shots/median,
                                      seconds_per_attempted_draw=median/shots)
        verify_inputs(report, args, circuit, settings)
        report.update(status='complete', summary=summaries,
                      cpu_over_gpu_wall_ratio=summaries['cpu']['median_wall_seconds']/summaries['gpu']['median_wall_seconds'])
    except Exception as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
    emit(report, args.output)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
