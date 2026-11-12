#!/usr/bin/env bash
set -euo pipefail
mode=${1:?verification mode required}
shift
: "${RTP_V41_TEST_SOURCE_ROOT:?checked native runtime required}"
export PYTHONPATH="$RTP_V41_TEST_SOURCE_ROOT:${PYTHONPATH:-}"
if [[ "$mode" == components ]]; then
    if (( $# )); then
        exec python -B -m unittest -v "$@"
    fi
    exec python -B -m unittest -v \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_indexer \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_verify_ops \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_verify_bf16_topk \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_verify_moe_overlap
fi
if [[ "$mode" == native_moe ]]; then
    exec python -B -m rtp_llm.models_py.modules.dsv41.ppu.test.verify_moe_overlap_role "$@"
fi
[[ "$mode" == role ]]
exec python -B "$RTP_V41_TEST_SOURCE_ROOT/rtp_llm/models_py/modules/dsv41/ppu/verify_opt/lease_role.py" "$@"
