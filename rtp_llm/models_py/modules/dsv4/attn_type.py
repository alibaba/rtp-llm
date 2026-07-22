"""DSV4 KV-pool attn_type IDs and framework cache tags.

The int constants are model-internal IDs used by DSV4 metadata and kernels.
Framework KVCache access uses semantic string tags via ``TAG_BY_ATTN_TYPE``.

Lookup helpers that actually read ``attn_inputs`` / ``kv_cache`` live in
:mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.
"""

# Canonical model-internal attn_type ids.
SWA_KV = 7
CSA_KV = 1
HCA_KV = 2
INDEXER_KV = 3
INDEXER_STATE = 4
CSA_STATE = 5
HCA_STATE = 6

TAG_BY_ATTN_TYPE = {
    CSA_KV: "csa_kv",
    HCA_KV: "hca_kv",
    INDEXER_KV: "indexer_kv",
    INDEXER_STATE: "indexer_state",
    CSA_STATE: "csa_state",
    HCA_STATE: "hca_state",
    SWA_KV: "swa_kv",
}
