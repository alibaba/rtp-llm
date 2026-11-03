"""Decode DCP execution switches for the K3.1 DSA target (FP8 Indexer, Page-RR KV over the TP group).

Each switch is read once, when the DSA workspace owner is built; the defaults keep the original path.

DSA_DCP_TOPK
    a2a        Every rank all-to-alls its local FP32 Indexer scores, selects the exact top-k of its
               share of the rows, and all-gathers the global positions (default).
    mega_topk  MegaTopk (rtp_llm_ops.mega_topk) selects the exact global top-k inside one kernel and
               returns the global positions this rank owns; no score is exchanged.
DSA_DCP_COMBINE
    a2a        LSE-weighted head scatter of the decode partial outputs through an NCCL all-to-all
               (default).
    mega_a2a   MegaA2A: peers pull their heads' partial outputs from symmetric memory and merge them
               inside one kernel.
"""

import os


def _option(name, default, choices):
    value = os.environ.get(name, default).strip()
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def dcp_topk_mode():
    return _option("DSA_DCP_TOPK", "a2a", ("a2a", "mega_topk"))


def dcp_combine_backend():
    return _option("DSA_DCP_COMBINE", "a2a", ("a2a", "mega_a2a"))
