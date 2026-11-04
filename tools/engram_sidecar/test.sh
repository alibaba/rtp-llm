#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
engram_python="${PYTHON:-python3}"
engram_prefix=rtp_llm.models_py.modules.dsv41.test
engram_tests=()
for name in exchange publish descriptors capacity manifest shard shard_config shard_storage storage model_attach; do
    engram_tests+=("${engram_prefix}.test_engram_${name}")
done
case "${1:---cpu}" in
    --cpu) ;;
    --torch)
        for name in checkpoint bootstrap external_checkpoint external_warmup; do
            engram_tests+=("${engram_prefix}.test_engram_${name}")
        done
        engram_tests+=(tools.engram_sidecar.framework.test_startup_store)
        ;;
    *) echo 'Usage: test.sh [--cpu|--torch]' >&2; exit 2 ;;
esac
exec "$engram_python" -m unittest "${engram_tests[@]}"
