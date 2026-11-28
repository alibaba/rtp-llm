#!/usr/bin/env bash
set -euo pipefail
# Match model smoke: do not generate multi-GB core artifacts in CI.
ulimit -c 0
: "${TEST_TMPDIR:?}" "${TEST_UNDECLARED_OUTPUTS_DIR:?}"
exec /opt/conda310/bin/python3 "$(dirname "$0")/run_cache_storage_smoke.py" --binary "$1" --config "$2"
