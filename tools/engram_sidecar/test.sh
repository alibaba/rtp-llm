#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
engram_python="${PYTHON:-python3}"
engram_prefix=rtp_llm.models_py.modules.dsv41.test
engram_tests=()
engram_contract=0
for name in exchange publish descriptors capacity manifest shard shard_config shard_storage storage model_attach; do
    engram_tests+=("${engram_prefix}.test_engram_${name}")
done
case "${1:---cpu}" in
    --cpu) ;;
    --torch)
        for name in checkpoint bootstrap external_checkpoint external_warmup external_scale_source; do
            engram_tests+=("${engram_prefix}.test_engram_${name}")
        done
        engram_tests+=(tools.engram_sidecar.framework.test_startup_store)
        engram_contract=1
        ;;
    --cuda-lookup)
        if [[ "${ENGRAM_RUN_CUDA_TESTS:-0}" != "1" ]]; then
            echo 'Set ENGRAM_RUN_CUDA_TESTS=1 to run the real Fabric lookup test.' >&2
            exit 2
        fi
        engram_tests=(tools.engram_sidecar.native.test_fabric_lookup_cuda)
        ;;
    *) echo 'Usage: test.sh [--cpu|--torch|--cuda-lookup]' >&2; exit 2 ;;
esac
"$engram_python" -m unittest "${engram_tests[@]}"
if [[ "$engram_contract" == "1" ]]; then
    # This runner initializes its CPU-only Torch fixture in main(); importing it
    # as another unittest module would silently miss that setup contract.
    "$engram_python" tools/engram_sidecar/native/test_embedding_contract.py --repo "$PWD"
fi
