#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: cc_test_wrapper <binary> [args...]" >&2
  exit 1
fi

exec "$@"
