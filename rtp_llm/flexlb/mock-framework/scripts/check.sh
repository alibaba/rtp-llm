#!/usr/bin/env bash
set -euo pipefail
framework_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$framework_dir/scripts/build.sh"
uv run --locked --directory "$framework_dir" pytest "$framework_dir/tests" "$@"
