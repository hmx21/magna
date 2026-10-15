#!/usr/bin/env python3
"""Run the generic simulator regression suite."""
from pathlib import Path
import argparse
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('core',)


def load_suite(groups=GROUPS, pattern='test_*.py'):
    """Shared discovery for the runner, benchmarks and sanitizer checks."""
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / 'core'),
                   *(str(ROOT / 'tests' / group) for group in GROUPS)]
    suite = unittest.TestSuite()
    for group in groups:
        for path in sorted((ROOT / 'tests' / group).glob(pattern)):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(path.stem))
    if suite.countTestCases() == 0:
        raise ValueError(f'No tests selected: groups={groups}, pattern={pattern}')
    return suite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=('current', 'all', 'core'), default='current')
    p.add_argument('--list', action='store_true', help='list source files without importing tests')
    args = p.parse_args()
    groups = GROUPS
    files = sorted(f for group in groups for f in (ROOT / 'tests' / group).glob('test_*.py'))
    if args.list:
        for f in files:
            print(f.relative_to(ROOT))
        return 0
    suite = load_suite(groups)
    print(f'{args.suite}: {len(files)} files, {suite.countTestCases()} tests', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
