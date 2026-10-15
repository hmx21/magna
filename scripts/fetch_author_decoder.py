"""Download the pinned author decoder sources used by a task directory.

The TQD tasks call the just-in-time decoder published by its authors on GitHub.
Its source is not distributed with these tasks. For each task directory, this script reads
decoder/author_source/provenance.json, downloads every recorded module at the pinned commit,
checks its SHA-256 hash and writes it next to provenance.json. Nothing is imported or executed.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def pinned(provenance):
    """Return (repository, commit, {file: hash}) of the snapshot rooted at the repository top level."""
    snapshot = next(s for s in provenance['snapshots'] if s['directory'] == '.')
    files = {name: digest for name, digest in provenance['files'].items() if '/' not in name}
    return provenance['repository'].removeprefix('https://github.com/'), snapshot['commit'], files


def fetch(source, repository, commit, name):
    if source is not None:
        return (source / name).read_bytes()
    with urlopen(f'https://raw.githubusercontent.com/{repository}/{commit}/{name}', timeout=60) as response:
        return response.read()


def install(task, source=None):
    target = Path(task) / 'decoder' / 'author_source'
    repository, commit, files = pinned(json.loads((target / 'provenance.json').read_text()))
    for name, digest in sorted(files.items()):
        path = target / name
        if path.is_file() and sha256(path.read_bytes()).hexdigest() == digest:
            continue
        data = fetch(source, repository, commit, name)
        if sha256(data).hexdigest() != digest:
            raise ValueError(f'{repository}@{commit[:12]}/{name} does not match the pinned hash')
        path.write_bytes(data)
    print(f'{task}: {len(files)} files of {repository}@{commit[:12]} verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='*', type=Path,
                        help='task directories (default: every examples/ task with author sources)')
    parser.add_argument('--from', dest='source', type=Path,
                        help='copy from a local checkout of the pinned commit instead of downloading')
    args = parser.parse_args()
    tasks = args.tasks or sorted(p.parents[2] for p in ROOT.glob('examples/*/decoder/author_source/provenance.json'))
    if not tasks:
        sys.exit('no task directory with decoder/author_source/provenance.json found')
    for task in tasks:
        install(task, args.source)


if __name__ == '__main__':
    main()
