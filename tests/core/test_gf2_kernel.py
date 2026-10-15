"""Independent native oracles for shared packed GF(2) operations and phases."""
from pathlib import Path
import tempfile
import unittest

from scripts.validate_gf2_kernels import validate


class SharedGF2KernelTest(unittest.TestCase):
    def test_byte_oracles_with_undefined_behavior_sanitizer(self):
        with tempfile.TemporaryDirectory(prefix='magic-gf2-check-') as folder:
            result = validate(Path(folder))
        self.assertEqual(result['status'], 'passed')
        self.assertGreater(result['factor_cases'], 100)
        self.assertGreater(result['quadratic_cases'], 20)
