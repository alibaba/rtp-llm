#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPDATE_SCRIPT="${SCRIPT_DIR}/update_scheduler_config.sh"

[[ -x "$UPDATE_SCRIPT" ]] || {
    echo "update script is missing or not executable: $UPDATE_SCRIPT" >&2
    exit 1
}

exec "$UPDATE_SCRIPT" --set latency_current_sample_weight=4.0
