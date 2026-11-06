#!/usr/bin/env bash
# Run both smoke stages. Explicit role lifecycle commands use the lower-level launcher.
set -Eeuo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
smoke_dir="${repo_root}/rtp_llm/test/smoke/kimi_k3"
case "${1:-}" in
    launch|stop|prepare|serve|status|stop-role|requests|jit-keys)
        exec python3 -u "${smoke_dir}/two_host_launcher.py" "$@"
        ;;
    *)
        exec python3 -u "${smoke_dir}/k31_two_host_smoke.py" "$@"
        ;;
esac
