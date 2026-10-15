"""Native source navigation and profiling hooks must survive directory changes."""
import importlib
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile
import unittest

from scripts.native_sources import native_source

ROOT = Path(__file__).resolve().parents[2]


class NativeSourceToolsTests(unittest.TestCase):
    def test_source_resolution_rejects_absent_and_ambiguous_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError): native_source(root, 'probe.hpp')
            (root/'probe.hpp').write_text('')
            self.assertEqual(native_source(root, 'probe.hpp'), root/'probe.hpp')
            (root/'nested').mkdir()
            (root/'nested/probe.hpp').write_text('')
            with self.assertRaises(ValueError): native_source(root, 'probe.hpp')
