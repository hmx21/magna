#!/bin/sh
# Use a project venv or a system Python with NumPy.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$project_dir/core:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${MAGNA_PYTHON:-}" ]; then
    exec "$MAGNA_PYTHON" "$@"
fi
if [ -x "$project_dir/.venv/bin/python" ]; then
    exec "$project_dir/.venv/bin/python" "$@"
fi
if python3 -c 'import numpy' >/dev/null 2>&1; then
    exec python3 "$@"
fi
printf '%s\n' 'NumPy is required: create .venv and install requirements.txt.' >&2
exit 1
