#!/usr/bin/env python3
"""Prepare an isolated documentation environment and open the local manual."""

import argparse
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, help='use this port instead of selecting a free one')
    parser.add_argument('--no-browser', action='store_true', help='do not open the browser')
    args = parser.parse_args()
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')

    # Check the port before installing anything; bind only to the local machine.
    port = None
    for candidate in ([args.port] if args.port is not None else range(8000, 8100)):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', candidate))
            except OSError:
                continue
            port = candidate
            break
    if port is None:
        parser.error('no available port; choose a free port with --port')

    environment = ROOT / 'build' / 'docs-venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Creating the documentation environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = ROOT / 'docs' / 'requirements-docs.txt'
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / '.requirements-sha256'
    if not stamp.exists() or stamp.read_text() != fingerprint:
        print('Installing documentation dependencies (first run or requirements changed)...', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
        stamp.write_text(fingerprint)

    print(f'Opening http://127.0.0.1:{port}/ — press Ctrl-C to stop.', flush=True)
    os.chdir(ROOT)
    command = [str(python), '-m', 'mkdocs', 'serve', '--dev-addr', f'127.0.0.1:{port}']
    if not args.no_browser:
        command.append('--open')
    os.execv(str(python), command)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print('Documentation setup failed. Check the installation output above and rerun.', file=sys.stderr)
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
