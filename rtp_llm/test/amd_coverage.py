"""Map main's 36 ROCm Python targets and two GTests to native pytest cases.

Baseline: main-internal CI run 67542102 (AMD UT report, 38/38).
Shared tests are routed to AMD only for the py_ut_amd profile.
"""

from typing import NamedTuple


class AmdTarget(NamedTuple):
    name: str
    path: str
    gpu_count: int = 1
    case: str = ""

    def matches(self, nodeid: str) -> bool:
        path, _, case = nodeid.replace("\\", "/").partition("::")
        return path.endswith(self.path) and self.case in case


_BASE = "rtp_llm/models_py/modules/base/rocm/test/"
_ATTN = "rtp_llm/models_py/modules/factory/attention/rocm_impl/test/"
_LINEAR = "rtp_llm/models_py/modules/factory/linear/impl/rocm/test/"
_MOE = "rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/"
_FLA = "rtp_llm/models_py/triton_kernels/fla/test/"
_DESC = "rtp_llm/models_py/model_desc/test/"

AMD_TARGETS = (
    AmdTarget(
        "test_inline_fp8_quant", "rtp_llm/model_loader/test/test_inline_fp8_quant.py"
    ),
    AmdTarget(
        "ckpt_database_test_rocm",
        "rtp_llm/utils/test/ckpt_database_test.py",
        case="HandleRecyclingTest",
    ),
    AmdTarget(
        "rocm_general_layer_norm_test", _BASE + "rocm_general_layer_norm_test.py"
    ),
    AmdTarget("trt_allreduce_test", _BASE + "trt_allreduce_test.py", 2),
    AmdTarget("rocm_norm_test", _BASE + "rocm_norm_test.py"),
    AmdTarget("rocm_embedding_test", _BASE + "rocm_embedding_test.py"),
    AmdTarget(
        "rocm_fused_add_layernorm_test", _BASE + "rocm_fused_add_layernorm_test.py"
    ),
    AmdTarget("rocm_fusedqkrmsnorm_test", _BASE + "rocm_fusedqkrmsnorm_test.py"),
    AmdTarget(
        "test_trt_allreduce_graph_replay",
        _BASE + "test_trt_allreduce_graph_replay.py",
        2,
    ),
    AmdTarget("test_unified_allreduce_tp", _BASE + "test_unified_allreduce_tp.py", 2),
    AmdTarget("rocm_layer_norm_test", _BASE + "rocm_layer_norm_test.py"),
    AmdTarget("rocm_fmha_test", _BASE + "rocm_fmha_test.py"),
    AmdTarget("test_attn_utils", _ATTN + "test_attn_utils.py"),
    AmdTarget(
        "test_aiter_decode_triton_noasm", _ATTN + "test_aiter_decode_triton_noasm.py"
    ),
    AmdTarget("test_aiter_prefill_op", _ATTN + "test_aiter_prefill_op.py"),
    AmdTarget("test_fused_qkv_transpose_v3", _ATTN + "test_fused_qkv_transpose_v3.py"),
    AmdTarget(
        "fp8_ptpc_solution_cache_test", _LINEAR + "fp8_ptpc_solution_cache_test.py"
    ),
    AmdTarget("rocm_linear_test", _LINEAR + "rocm_linear_test.py"),
    AmdTarget("fp8_ptpc_linear_test", _LINEAR + "fp8_ptpc_linear_test.py"),
    AmdTarget("rocm_fused_moe_test", _MOE + "rocm_fused_moe_test.py"),
    AmdTarget(
        "moriep_intranode_router_test_2gpu",
        _MOE + "moriep_intranode_router_test.py",
        2,
        "test_world_size_2",
    ),
    AmdTarget(
        "moriep_intranode_router_test_4gpu",
        _MOE + "moriep_intranode_router_test.py",
        4,
        "test_world_size_4",
    ),
    AmdTarget("test_generic_moe_allreduce", _MOE + "test_generic_moe_allreduce.py", 2),
    AmdTarget("test_pure_tp_router", _MOE + "test_pure_tp_router.py"),
    AmdTarget("rocm_fp8_fused_moe_test", _MOE + "rocm_fp8_fused_moe_test.py"),
    AmdTarget(
        "fused_moe_allreduce_contract_test_rocm",
        "rtp_llm/models_py/modules/factory/fused_moe/defs/test/fused_moe_allreduce_contract_test.py",
    ),
    AmdTarget("test_gdn_decode_rocm", _FLA + "test_gdn_decode.py"),
    AmdTarget(
        "test_flydsl_chunk_gdn_shape_gate", _FLA + "test_flydsl_chunk_gdn_shape_gate.py"
    ),
    AmdTarget("test_l2norm_rocm", _FLA + "test_l2norm.py"),
    AmdTarget(
        "test_flydsl_chunk_gdn_cache_store",
        _FLA + "test_flydsl_chunk_gdn_cache_store.py",
    ),
    AmdTarget("test_chunk_prefill_amd", _FLA + "test_chunk_prefill.py"),
    AmdTarget("test_gdn_block_prefill_rocm", _FLA + "test_gdn_block_prefill.py"),
    AmdTarget(
        "test_remap_local_ids",
        "rtp_llm/models_py/triton_kernels/moe/test/test_remap_local_ids.py",
    ),
    AmdTarget("moriep_test", "rtp_llm/models_py/distributed/test/moriep_test.py", 8),
    AmdTarget(
        "qwen3_next_qkvz_ba_fusion_test_rocm",
        _DESC + "qwen3_next_qkvz_ba_fusion_test.py",
    ),
    AmdTarget(
        "generic_moe_allreduce_test_rocm", _DESC + "generic_moe_allreduce_test.py"
    ),
)

AMD_GTEST_CASES = (
    "RocmBeamSearchOpTest.simpleTest",
    "RocmBeamSearchOpTest.variableBeamWidthTest",
)
AMD_GTEST_ADAPTER = (
    "rtp_llm/models_py/bindings/rocm/ops/tests/test_rocm_beam_search_op.py"
)


def route_amd_items(items):
    import pytest

    for item in items:
        for target in AMD_TARGETS:
            if target.matches(item.nodeid):
                item.add_marker(
                    pytest.mark.gpu(type="MI308X", count=target.gpu_count),
                    append=False,
                )
                break


def validate_amd_coverage(nodeids):
    """Reject missing baseline targets even if unrelated cases keep totals high."""
    nodeids = set(nodeids)
    missing = [
        target.name
        for target in AMD_TARGETS
        if not any(target.matches(nodeid) for nodeid in nodeids)
    ]
    missing.extend(
        case
        for case in AMD_GTEST_CASES
        if not any(
            nodeid.startswith(AMD_GTEST_ADAPTER + "::") and f"[{case}]" in nodeid
            for nodeid in nodeids
        )
    )
    if missing:
        raise ValueError("AMD baseline coverage missing: " + ", ".join(missing))
