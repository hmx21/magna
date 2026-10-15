"""Check online causal reduction with independent finite assignment/span oracles."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeCausalTest(unittest.TestCase):
    def test_online_basis_without_cpython(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler, 'C++17 compiler required for native causal checks')
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'causal'
            build = subprocess.run([compiler, '-std=c++17', '-O1', '-fsanitize=undefined',
                '-fno-sanitize-recover=all', '-I'+str(ROOT/'core/magna/native'),
                str(ROOT/'tests/native/causal.cpp'), '-o', str(executable)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('native causal checks passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
