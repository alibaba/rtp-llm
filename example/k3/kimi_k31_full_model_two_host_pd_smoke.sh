#!/usr/bin/env bash
# Start both PD roles over SSH; request argv belong to the existing case runner.
set -Eeuo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
exec python3 -u "${repo_root}/rtp_llm/test/smoke/kimi_k3/two_host_launcher.py" "$@"
