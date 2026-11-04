#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROLE=${0##*_simulate_}
PYTHON_BIN=${NAVI_SCHED_PYTHON_BIN:-}

is_python310() {
    "$1" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' \
        >/dev/null 2>&1
}

if [ -n "${PYTHON_BIN}" ]; then
    if ! is_python310 "${PYTHON_BIN}"; then
        echo "NAVI_SCHED_PYTHON_BIN must reference Python 3.10" >&2
        exit 1
    fi
else
    PYTHON_BIN=
    for candidate in python3 /opt/conda310/bin/python3 python; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            if is_python310 "${candidate}"; then
                PYTHON_BIN=${candidate}
                break
            fi
        fi
    done
fi

if [ -z "${PYTHON_BIN}" ]; then
    echo "Python 3.10 interpreter not found; set NAVI_SCHED_PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/../python${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/../python/navi_sched_simulate_entry.py" \
    "${ROLE}" "$@"
