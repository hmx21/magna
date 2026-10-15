"""Locate a unique native instrumentation source across current and archived layouts."""
from pathlib import Path


def native_source(directory, filename):
    matches = list(Path(directory).rglob(filename))
    if len(matches) != 1:
        raise ValueError(f'Expected one native source {filename!r}, found {matches}')
    return matches[0]
