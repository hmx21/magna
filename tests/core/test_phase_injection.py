"""Exact injection pullback, scalar and transport checks in the native owner."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

class PhaseInjectionTests(unittest.TestCase):
    def test_exact_native_injection(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/'phase-injection'
            built = subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=undefined',
                '-fno-sanitize-recover=all', '-I'+str(ROOT/'core/magna/native'),
                str(ROOT/'tests/native/phase_injection.cpp'), '-o', str(binary)],
                capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stdout+ran.stderr)
            self.assertIn('injection checks passed', ran.stdout)
