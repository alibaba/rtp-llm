"""Decode DCP execution switches for the K3.1 DSA target (FP8 Indexer, Page-RR KV over the TP group).

Each switch is read once, when the DSA workspace owner (DSA_DCP_PROLOGUE: the layer and its weights)
is built. The defaults are the validated decode path; set a switch explicitly to get the original
one, which is what the per-arm comparisons of the layer test and the DSA benchmark do.

DSA_DCP_TOPK
    a2a        Every rank all-to-alls its local FP32 Indexer scores, selects the exact top-k of its
               share of the rows, and all-gathers the global positions (the original path).
    mega_topk  MegaTopk (rtp_llm_ops.mega_topk) selects the exact global top-k inside one kernel and
               returns the global positions this rank owns; no score is exchanged (default).
DSA_DCP_COMBINE
    a2a        LSE-weighted head scatter of the decode partial outputs through an NCCL all-to-all
               (the original path).
    mega_a2a   MegaA2A: peers pull their heads' partial outputs from symmetric memory and merge them
               inside one kernel (default).
DSA_DCP_PROLOGUE
    dev           The layer's projections, Indexer and sparse-MLA query as they are (the original
                  path).
    serial        Every rank computes every query head (q_b and kc load all heads; no query
                  all-gather), the Indexer head gate is one BF16 GEMM over its FP32 weight split
                  into three BF16 parts, on one stream.
    multi_stream  The same operators, with the work MQA does not wait for on side streams (default).
DSA_SFA_DECODE
    sched          flash_mla_with_kvcache on the FP8 cache: host scheduler metadata, 64 query heads
                   per launch, and every row walking all `topk` slots (the original path).
    metadata_free  flash_mla_sparse_decode_fp8_kv: one launch, 64/96/128 query heads natively, and a
                   device `topk_length` so a row only walks the slots this owner holds (default).
                   Requires a FlashMLA build that exports the operator.
"""

import os


def _option(name, default, choices):
    value = os.environ.get(name, default).strip()
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def dcp_topk_mode():
    return _option("DSA_DCP_TOPK", "mega_topk", ("a2a", "mega_topk"))


def dcp_combine_backend():
    return _option("DSA_DCP_COMBINE", "mega_a2a", ("a2a", "mega_a2a"))


def dcp_prologue_mode():
    return _option("DSA_DCP_PROLOGUE", "multi_stream", ("dev", "serial", "multi_stream"))


def sfa_decode_backend():
    return _option("DSA_SFA_DECODE", "metadata_free", ("sched", "metadata_free"))
