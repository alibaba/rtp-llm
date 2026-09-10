"""Recorded baseline configuration; set before model/weight construction."""

import os

BASELINE = {
    "MOE_STRATEGY": "mega_moe_fp8_se",
    "DSV4_MOE_CHUNK_TOKENS": "131072",
    "GLM53_PREFILL_SHARED_EXPERT_LOCAL": "1",
    "GLM53_KDA_LOCAL_LOW_RANK": "0",
    "GLM53_KDA_PREFILL_CONV_LAYOUT": "1",
    "GLM53_KDA_DECODE_FUSION": "1",
    "GLM53_KDA_RECURRENT_LOW_WARPS": "1",
    "GLM53_PREFILL_AG_GEMM": "1",
    "GLM53_PREFILL_GEMM_RS": "1",
    "GLM53_PREFILL_STABLE_RS": "1",
    "GLM53_MOE_TP_TOKEN_SHARD": "1",
    "GLM53_MOE_SHARED_OVERLAP": "0",
    "GLM53_INDEXER_REQUEST_CHUNKS": "1",
    "GLM53_SPARSE_MLA_NATIVE_Q_CHUNK": "131072",
    "GLM53_SPARSE_MLA_BF16_BACKEND": "auto",
    "GLM53_SPARSE_MLA_BF16_Q_CHUNK": "4096",
    "GLM53_PREFILL_INDEXER_TOPK_BACKEND": "topk_v3_tie_break",
    "GLM53_DECODE_INDEXER_TOPK_BACKEND": "topk_v3",
    "DSV4_INDEXER_TOPK_CANONICALIZE": "1",
    "GLM5_KDA_PREFILL_BACKEND": "cula",
    "ENABLE_LINEAR_ATTN_REQUEST_CACHE": "1",
    "DETERMINISTIC_GEMM": "1",
    "DG_GEMM_RS_NVLINK_BARRIER_TIMEOUT_SECS": "60",
    "DG_MEGA_MOE_FP8_NVLINK_BARRIER_TIMEOUT_SECS": "60",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def configure(phase):
    for key, value in BASELINE.items():
        os.environ.setdefault(key, value)
    prefill = phase == "prefill"
    for key in ("GLM53_PREFILL_SEQUENCE_PARALLEL", "GLM53_PREFILL_MLA_CP"):
        os.environ[key] = "1" if prefill else "0"
    os.environ.setdefault("GLM53_INDEXER_FUSED_Q_QUANT", "1" if prefill else "0")
    os.environ["PREFILL_CP_KV_CACHE_SHARDED"] = "true"
    os.environ["PREFILL_CP_SIZE"] = "8"
    os.environ["ROLE_TYPE"] = phase.upper()
