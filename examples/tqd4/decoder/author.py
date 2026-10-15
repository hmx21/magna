"""Load the pinned, unmodified author package for this repository example."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType

SOURCE = Path(__file__).resolve().parent / 'author_source'
REVISION = 'd4f0918c841bb4dde632dd5010425a28f4c0e43f'
MODULES = ('geometry', 'lattice', 'decoder', 'twisted', 'runner')


def verify(snapshot='current'):
    """Check every imported module before executing any author code."""
    if snapshot not in ('current', 'paper-era'):
        raise ValueError('unknown author snapshot')
    directory = SOURCE if snapshot == 'current' else SOURCE / snapshot
    provenance = json.loads((SOURCE / 'provenance.json').read_text())
    hashes = {}
    for module in MODULES:
        filename = module + '.py'
        key = filename if snapshot == 'current' else snapshot + '/' + filename
        # The older provenance omitted this imported module. This hash is
        # verified against the provenance-pinned paper-era source archive.
        expected = (provenance['files'].get(key) or
                    {'paper-era/lattice.py': 'a490a55914b83a5512193fe5d5527c1e7eac707f917ec655a272a422e4124a72'}[key])
        if not (directory / filename).is_file():
            raise FileNotFoundError(
                f'Missing author source: {directory / filename}. '
                'Download it with: python scripts/fetch_author_decoder.py <task directory>')
        actual = sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'author source hash changed: {key}')
        hashes[key] = actual
    return hashes


@lru_cache(None)
def load(snapshot='current'):
    verify(snapshot)
    directory = SOURCE if snapshot == 'current' else SOURCE / snapshot
    name = '_tqd2604_author_' + snapshot.replace('-', '_')
    # The paper-era __init__ imports functions absent from its runner.
    # Supply only the package namespace; execute the original submodules.
    module = ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module
    for child in MODULES:
        setattr(module, child, __import__(name + '.' + child, fromlist=[child]))
    return module
