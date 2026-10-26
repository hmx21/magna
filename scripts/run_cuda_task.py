"""Run a portable run.json task with the optional CUDA sampling backend.

Uses the installed magna package and the same task loader as run_task.py.
The cpu backend is the CUDA backend's Philox CPU control, not magna.run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from run_task import load


def add_task_arguments(parser):
    parser.add_argument('--task', type=Path, required=True, help='directory containing run.json')
    parser.add_argument('--shots', type=int, help='total attempted draws; default: run.json shots')
    parser.add_argument('--seed', type=int, help='random seed; default: run.json seed')
    parser.add_argument('--batch-size', type=int, default=65536,
                        help='draws per GPU/Philox batch, independent of total --shots')
    parser.add_argument('--sampling', choices=('physical', 'importance'),
                        help='explicitly override run.json sampling method')
    parser.add_argument('--proposal-scale', help='exact rational importance scale, e.g. 10 or 1000/7')
    parser.add_argument('--output', type=Path, help='JSON destination; default: stdout')


def load_task(args):
    circuit, settings, config = load(args.task)
    sampling = dict(settings['sampling'])
    if args.sampling == 'physical':
        sampling = {'method': 'physical'}
    elif args.sampling == 'importance':
        sampling['method'] = 'importance'
    if args.proposal_scale is not None:
        if sampling.get('method') != 'importance':
            raise ValueError('--proposal-scale requires importance in run.json or --sampling importance')
        sampling['proposal_scale'] = args.proposal_scale
    if sampling.get('method') == 'importance' and 'proposal_scale' not in sampling:
        raise ValueError('importance requires an explicit --proposal-scale or a scale in run.json')
    settings['sampling'] = sampling
    shots = args.shots if args.shots is not None else config.get('shots', 1024)
    seed = args.seed if args.seed is not None else config.get('seed', 0)
    if type(shots) is not int or not 1 <= shots <= 2**53:
        raise ValueError('shots must be a positive integer at most 2**53')
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    return circuit, settings, config, shots, seed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_files(directory, circuit, settings):
    """Hash declared files, including target/recovery references, without scanning folders."""
    directory = directory.resolve()
    task_path = Path(settings['task'])
    task = json.loads(task_path.read_text(encoding='utf-8'))
    paths = [directory/'run.json', Path(circuit), task_path]
    config = json.loads((directory/'run.json').read_text(encoding='utf-8'))
    if config.get('controllers'):
        paths.append(directory/config['controllers'])
    for field in ('ideal_state', 'recovery'):
        value = task.get(field)
        if isinstance(value, dict):
            value = value.get('circuit')
        if value is not None:
            paths.append(task_path.parent/str(value))
    return {os.path.relpath(p.resolve(), directory).replace(os.sep, '/'):
            {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in paths}


def package_record():
    import magna
    from magna import _native
    package = Path(magna.__file__).resolve().parent
    try:
        version = metadata.version('magna-quantum')
    except metadata.PackageNotFoundError:
        version = None
    modules = ['__init__.py', 'file_api.py', '_cuda_source.py', 'cuda_experiment.py', 'cuda_noise.py',
               'cuda_queries.py', 'cuda_sampler.py', Path(_native.__file__).name,
               'native/cuda/frame_transport.cu', 'native/cuda/noise_sampling.cu',
               'native/cuda/probability_queries.cu', 'native/cuda/statistics.cu']
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                            'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return dict(distribution='magna-quantum', version=version, script_checkout_revision=revision,
                installed_module_sha256={name: sha256(package/name) for name in modules},
                note='The script checkout revision need not identify the installed package; module hashes identify the measured installation.')


def base_report(args, config, settings, compiled, circuit):
    import numpy as np
    cpu_model = platform.processor()
    if sys.platform.startswith('linux'):
        cpuinfo = Path('/proc/cpuinfo')
        if cpuinfo.is_file():
            cpu_model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                              if line.startswith('model name')), cpu_model)
    return dict(schema_version=1, status='running', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                task=str(args.task), run_config=config,
                overrides=dict(sampling=args.sampling, proposal_scale=args.proposal_scale),
                requested_sampling=settings['sampling'], inspection=compiled.inspect(),
                input_files=input_files(args.task, circuit, settings), package=package_record(),
                runner_sha256={p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('run_task.py'))},
                hardware=dict(cpu=cpu_model, logical_cpus=os.cpu_count()),
                software=dict(python=sys.version, platform=platform.platform(), numpy=np.__version__))


def record_gpu(report, cp):
    device = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props['name']
    report['hardware']['gpu'] = dict(index=device, name=name.decode() if isinstance(name, bytes) else name,
                                     memory_bytes=props['totalGlobalMem'])
    report['software'].update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                              cuda_driver=cp.cuda.runtime.driverGetVersion())


def verify_inputs(report, args, circuit, settings):
    if input_files(args.task, circuit, settings) != report['input_files']:
        raise RuntimeError('task inputs changed during sampling')
    if package_record()['installed_module_sha256'] != report['package']['installed_module_sha256']:
        raise RuntimeError('installed simulator files changed during sampling')
    for name, expected in report['runner_sha256'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f'runner changed during sampling: {name}')


def measure(function, shots, synchronize=None):
    if synchronize is not None:
        synchronize()
    start = perf_counter()
    result = function()
    if synchronize is not None:
        synchronize()
    wall = perf_counter()-start
    if result.get('status') != 'complete' or result.get('completed') != shots:
        raise RuntimeError(f'incomplete sampling: {result.get("failures", result.get("status"))}')
    return result, wall


def emit(report, output):
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if output is None:
        print(text, end='')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_task_arguments(parser)
    parser.add_argument('--offset', type=int, default=0, help='first global Philox shot ID')
    parser.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
    args = parser.parse_args()
    report = dict(schema_version=1, status='failed', task=str(args.task))
    try:
        from magna import compile_circuit
        from magna.cuda_sampler import CudaSampler
        circuit, settings, config, shots, seed = load_task(args)
        if not 0 <= args.offset < 2**64 or args.offset+shots > 2**64:
            raise ValueError('offset and final global shot ID must fit unsigned 64-bit integers')
        start = perf_counter()
        compiled = compile_circuit(circuit, **settings)
        compile_seconds = perf_counter()-start
        start = perf_counter()
        sampler = CudaSampler(compiled)
        plan_seconds = perf_counter()-start
        report = base_report(args, config, settings, compiled, circuit)
        report.update(shots=shots, seed=seed, offset=args.offset, batch_size=args.batch_size, backend=args.backend,
                      preparation=dict(cpu_compile_seconds=compile_seconds, backend_plan_seconds=plan_seconds),
                      rng='philox4x32-10-shot-counter',
                      timing_scope='Warm complete API return; GPU includes device-wide completion. Preparation and the explicitly reported warmup are excluded.')
        start = perf_counter()
        warm_shots = min(shots, args.batch_size)
        measure(lambda: sampler.run(warm_shots, seed=seed, offset=args.offset,
                                   batch_size=args.batch_size, backend=args.backend), warm_shots)
        synchronize = None
        if args.backend == 'gpu':
            cp = sampler.noise.cp
            synchronize = cp.cuda.runtime.deviceSynchronize
            synchronize()
        report['warmup'] = dict(shots=warm_shots, seconds=perf_counter()-start)
        if args.backend == 'gpu':
            record_gpu(report, cp)
        result, wall = measure(lambda: sampler.run(shots, seed=seed, offset=args.offset,
                              batch_size=args.batch_size, backend=args.backend), shots, synchronize)
        verify_inputs(report, args, circuit, settings)
        report.update(status='complete', result=result, completed_call_wall_seconds=wall,
                      attempted_draws_per_second=shots/wall)
    except Exception as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
    emit(report, args.output)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
