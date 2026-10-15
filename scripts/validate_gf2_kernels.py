"""Build and run independent byte-arithmetic GF(2) checks, optionally audited.

CLI runs always use the existing fail-closed 2 GiB process-group RSS watchdog.
The unittest adapter calls validate() inside its already supervised test suite.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/native/gf2_kernel.cpp'
HEADERS = tuple(ROOT / f'core/magna/native/foundation/{name}.hpp' for name in ('bits', 'scalar', 'stabilizer'))
SEED = 2026100409


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('clang++') or shutil.which('g++')
    if compiler is None:
        raise RuntimeError('C++17 compiler required')
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in (SOURCE, *HEADERS)}
    binary = output / 'gf2-kernel'
    command = [compiler, '-std=c++17', '-O1', '-fsanitize=undefined',
               '-fno-sanitize-recover=all', '-Wall', '-Wextra',
               '-I' + str(ROOT / 'core/magna/native'), str(SOURCE), '-o', str(binary)]
    started = time.monotonic()
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (output / 'build.stdout').write_text(build.stdout)
    (output / 'build.stderr').write_text(build.stderr)
    if build.returncode:
        raise RuntimeError('GF(2) validation build failed: ' + build.stderr)
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=90)
    (output / 'check.stdout').write_text(run.stdout)
    (output / 'check.stderr').write_text(run.stderr)
    if run.returncode:
        raise RuntimeError('GF(2) independent oracle failed: ' + run.stdout + run.stderr)
    result = json.loads(run.stdout)
    assert result['status'] == 'passed' and result['seed'] == SEED
    assert inputs == {str(p.relative_to(ROOT)): digest(p) for p in (SOURCE, *HEADERS)}, 'Inputs changed during check'
    result.update(command=command, compiler=subprocess.check_output([compiler, '--version'], text=True),
                  elapsed_seconds=time.monotonic() - started, binary_sha256=digest(binary),
                  source_sha256=inputs, arithmetic='Unpacked byte GF(2), exact Z/4 quadratic phases and Z/8 scalar phases',
                  oracle='Left-to-right scalar elimination, exhaustive small affine solution sets, direct quadratic evaluation',
                  non_claims=['Bounded deterministic correctness checks; no universal proof or performance claim.'])
    save(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skill-dir', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    folder = args.output.resolve()
    if args.worker:
        print(json.dumps(validate(folder)))
        return
    if folder.exists():
        raise FileExistsError('Refusing to overwrite validation evidence: ' + str(folder))
    guard = [sys.executable, str(ROOT / 'scripts/memory_guard.py'),
             '--seconds', '210', '--output', str(folder / 'guard.json'), '--',
             sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(folder)]
    if args.skill_dir:
        relative = lambda p: str(p.resolve().relative_to(ROOT))
        artifacts = [relative(p) for p in (SOURCE, *HEADERS, Path(__file__), ROOT / 'scripts/memory_guard.py')]
        contract = dict(claim_id='GF2_SHARED_PACKED_KERNEL_VALIDATION_20261004',
            mathematics=dict(assertion_tested='Packed parity, selected-row XOR, matrix transpose, rectangular GF(2) solutions and affine quadratic restriction agree with independent scalar oracles.',
                coefficient_domain='Exact GF(2), Z/4 and Z/8 integer arithmetic.',
                conventions='Least significant bit is column zero; highest-bit production pivots; strict upper-triangular quadratic cross terms.',
                inputs=['Seeded dense/sparse matrices, zero and duplicate rows, independent full-rank systems and word-boundary dimensions.'],
                bounds=dict(max_bit_width=264, max_matrix_width=257, max_factor_rows=257, max_packed_rows=137, max_quadratic_input_width=137,
                            max_quadratic_output_width=129, exhaustive_solution_width=5,
                            exhaustive_quadratic_output_width=8, low_limit_includes="SIZE_MAX", rss_bytes=2 * 1024**3, wall_seconds=210),
                non_claims=['Finite checks do not prove all widths or establish a performance gain.']),
            execution_artifacts=artifacts, software=[dict(name='Python', version=platform.python_version())],
            randomness=dict(used=True, generator='C++ std::mt19937_64; modulo selection fixed in versioned test source', seed=SEED))
        contract_path = folder.with_name(folder.name + '-contract.json')
        save(contract_path, contract)
        command = [sys.executable, str(args.skill_dir / 'scripts/run_experiment.py'), '--root', str(ROOT),
                   '--contract', str(contract_path), '--output', relative(folder), '--timeout', '240',
                   '--max-output-bytes', '1048576', '--max-threads', '1']
        for path in artifacts:
            command += ['--input', path]
        for name in ('result.json', 'guard.json'):
            command += ['--result', relative(folder / name)]
        subprocess.run([*command, '--', *guard], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(args.skill_dir / 'scripts/validate_manifest.py'),
                        str(folder / 'manifest.json'), '--root', str(ROOT)], check=True)
    else:
        subprocess.run(guard, cwd=ROOT, start_new_session=True, check=True)


if __name__ == '__main__':
    main()
