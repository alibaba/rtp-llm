#!/usr/bin/env bash
# Run inherited PPU model and device checks against a published snapshot.
set -euo pipefail

v41_role=${1:?functional test role required}
: "${RTP_V41_TEST_SOURCE_ROOT:?published snapshot required}"
v41_python=${RTP_V41_TEST_PYTHON:-/opt/conda312/bin/python}
v41_output=${TEST_UNDECLARED_OUTPUTS_DIR:?Bazel test output directory required}
v41_source=$(cd "$RTP_V41_TEST_SOURCE_ROOT" && pwd)
cd "$v41_source"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONPATH="$v41_source:${PYTHONPATH:-}"

if [[ "$v41_role" == s5-kernel-surface ]]; then
    # The normal runtime_identity package import queries device availability.
    # This direct script validates the same snapshot with standard-library IO.
    exec "$v41_python" -B rtp_llm/models_py/modules/dsv41/ppu/test/test_mhc_mega_probe.py --mode surface
fi
"$v41_python" -B -c 'from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import runtime_identity; print(runtime_identity())'
if [[ "$v41_role" == ptpc-qnorm-contracts || "$v41_role" == ptpc-qnorm-device ]]; then
    export RTP_V41_PPU_PTPC_FUSION=1
    if [[ "$v41_role" == ptpc-qnorm-device ]]; then export RTP_V41_TEST_PTPC_DEVICE=1; fi
    exec "$v41_python" -B -m unittest -v \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_quant \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_norm \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prepared \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_input_reuse \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_qnorm_reuse
fi
if [[ "$v41_role" == ptpc-prepared-contracts || "$v41_role" == ptpc-prepared-device ]]; then
    export RTP_V41_PPU_PTPC_FUSION=1
    if [[ "$v41_role" == ptpc-prepared-device ]]; then export RTP_V41_TEST_PTPC_DEVICE=1; fi
    exec "$v41_python" -B -m unittest -v \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_quant \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_norm \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prepared \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_input_reuse
fi
if [[ "$v41_role" == ptpc-fusion-contracts || "$v41_role" == ptpc-fusion-device ]]; then
    export RTP_V41_PPU_PTPC_FUSION=1
    if [[ "$v41_role" == ptpc-fusion-device ]]; then export RTP_V41_TEST_PTPC_DEVICE=1; fi
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_quant \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_norm \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prefill_out \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_prefill_reuse \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_woa_direct
fi
if [[ "$v41_role" == ptpc-fusion-micro ]]; then
    export RTP_V41_PPU_PTPC_FUSION=1
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.ptpc_microbench
fi
if [[ "$v41_role" == ptpc-contracts || "$v41_role" == ptpc-device ]]; then
    if [[ "$v41_role" == ptpc-device ]]; then export RTP_V41_TEST_PTPC_DEVICE=1; fi
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc \
        rtp_llm.tools.v41.test_integration.IntegrationTest.test_checkpoint_metadata_does_not_claim_weight_content_signature
fi
if [[ "$v41_role" == ptpc-alignment ]]; then
    exec "$v41_python" -B -m rtp_llm.tools.v41.validate_ptpc \
        --source "${RTP_V41_TEST_CHECKPOINT:?source checkpoint required}" \
        --reference "${RTP_V41_TEST_REFERENCE:?reference checkpoint required}" \
        --output "$v41_output"
fi
if [[ "$v41_role" == ptpc-audit ]]; then
    exec "$v41_python" -B -m rtp_llm.tools.v41.validate_ptpc \
        --audit-dir "${RTP_V41_TEST_AUDIT_DIR:?actual loader audit required}" \
        --expected-report "${RTP_V41_TEST_EXPECTED_REPORT:?full checkpoint comparison required}" \
        --output "$v41_output"
fi
if [[ "$v41_role" == ptpc-layout ]]; then
    exec "$v41_python" -B -m rtp_llm.tools.v41.validate_ptpc \
        --audit-dir "${RTP_V41_TEST_AUDIT_DIR:?source loader receipts required}" \
        --compare-layout-dir "${RTP_V41_TEST_COMPARE_LAYOUT_DIR:?reference loader receipts required}" \
        --output "$v41_output"
fi
if [[ "$v41_role" == moe-device ]]; then
    export RTP_V41_TEST_W4A8_DEVICE=1
    exec "$v41_python" -B rtp_llm/models_py/modules/dsv41/ppu/test/test_moe_w4a8.py
fi
if [[ "$v41_role" == graph-contracts ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.tools.v41.test_integration \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_cache \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_engram \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_decode_graph
fi
if [[ "$v41_role" == graph-cache-device ]]; then
    export RTP_V41_TEST_GRAPH_DEVICE=1
    exec "$v41_python" -B -m unittest rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_cache
fi
if [[ "$v41_role" == s5-mhc-mega ]]; then
    exec "$v41_python" -B rtp_llm/models_py/modules/dsv41/ppu/test/test_mhc_mega_probe.py --mode mhc-mega "${@:2}"
fi
if [[ "$v41_role" == mhc-integration ]]; then
    exec "$v41_python" -B -m unittest rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_integration
fi
if [[ "$v41_role" == shared-prefill-adapter ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_prefill_runtime.SharedPrefillDevice
fi
if [[ "$v41_role" == shared-sharded-boundary ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_cp_sharding.SharedShardedBoundary
fi
if [[ "$v41_role" == shared-dspark-modes-boundary ]]; then
    exec "$v41_python" -B rtp_llm/models_py/modules/dsv41/ppu/test/test_dspark_modes.py --device
fi
if [[ "$v41_role" == shared-optimized-boundary ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_packed_attention.SharedPackedAttentionDevice \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_fused_ops.SharedFusedOpsDevice \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_fused_ops.GraphSwaWriteDevice \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_projection_streams.SharedStreamsDevice
fi
if [[ "$v41_role" == graph-packed-boundary ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_packed_attention.SharedPackedAttentionDevice.test_decode_graph_reads_live_packed_pool_and_lengths \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_graph_indexer
fi
: "${RTP_V41_TEST_CHECKPOINT:?checkpoint directory required}"
if [[ "$v41_role" == shared-grouped-graph-decode ]]; then
    [[ "${RTP_V41_PPU_MOE_BACKEND:?select grouped explicitly}" == grouped ]]
    [[ "${RTP_V41_PPU_MOE_PRECISION:?select w4a8 explicitly}" == w4a8 ]]
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.graph_modes_role
fi
if [[ "$v41_role" == shared-dspark-direct-graph ]]; then
    [[ "${RTP_V41_PPU_MOE_BACKEND:?select a backend explicitly}" == "${2:-grouped}" ]]
    [[ "${RTP_V41_PPU_MOE_PRECISION:?select a precision explicitly}" == "${3:-w4a8}" ]]
    if [[ -n "${4:-}" ]]; then
        export RTP_V41_N9_DIRECT_MODE="$4"
    fi
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.graph_modes_role --direct
fi
if [[ "$v41_role" == shared-eplb-prefill || "$v41_role" == shared-eplb-decode ]]; then
    export RTP_V41_SHARED_ROLE=${v41_role##shared-eplb-}
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.eplb_rdma_role
fi
if [[ "$v41_role" == shared-dspark-modes-prefill || "$v41_role" == shared-dspark-modes-decode ]]; then
    export RTP_V41_SHARED_ROLE=${v41_role##shared-dspark-modes-}
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.dspark_rdma_role --mode-from-env
fi
if [[ "$v41_role" == shared-sharded-prefill || "$v41_role" == shared-sharded-decode ]]; then
    export RTP_V41_SHARED_ROLE=${v41_role##shared-sharded-}
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role --sharded
fi
if [[ "$v41_role" == shared-optimized ]]; then
    [[ "${RTP_V41_PPU_MOE_BACKEND:?select megamoe explicitly}" == megamoe ]]
    [[ "${RTP_V41_PPU_MOE_PRECISION:?select w4a4 explicitly}" == w4a4 ]]
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.optimized_smoke_role
fi
if [[ "$v41_role" == shared-graph ]]; then
    [[ "${RTP_V41_PPU_MOE_BACKEND:?select megamoe explicitly}" == megamoe ]]
    [[ "${RTP_V41_PPU_MOE_PRECISION:?select w4a4 explicitly}" == w4a4 ]]
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.graph_fidelity_role
fi
if [[ "$v41_role" == shared-dspark-prefill || "$v41_role" == shared-dspark-decode ]]; then
    export RTP_V41_SHARED_ROLE=${v41_role##shared-dspark-}
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.dspark_rdma_role
fi
if [[ "$v41_role" == shared-yoco140-prefill || "$v41_role" == shared-yoco140-decode ]]; then
    export RTP_V41_SHARED_ROLE=${v41_role##shared-yoco140-}
    exec "$v41_python" -B -m rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role
fi
if [[ "$v41_role" == shared-model ]]; then
    exec "$v41_python" -B -m unittest \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_model.TestSharedModelBoundaries \
        rtp_llm.models_py.modules.dsv41.ppu.test.test_model.TestSharedModelWeightsDevice.test_real_checkpoint_shared_consumer
fi
printf 'Unknown V4.1 PPU functional role: %s\n' "$v41_role" >&2
exit 2
