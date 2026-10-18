#!/usr/bin/env bash
set -euo pipefail
: "${UPSTREAM_BENCH_SCRIPT:?Set an absolute path to the official benchmark invocation}"
[[ "$UPSTREAM_BENCH_SCRIPT" = /* && -f "$UPSTREAM_BENCH_SCRIPT" ]]
exec /bin/bash "$UPSTREAM_BENCH_SCRIPT" "$@"
