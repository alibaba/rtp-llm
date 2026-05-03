"""DSV4 KV-pool attn_type IDs.

Pure int constants mirroring the C++ ``KVCacheRegionName`` enum in
``rtp_llm/models_py/bindings/OpDefs.h``. Safe to import from anywhere
(no torch / no .so dependencies).

Lookup helpers that actually read ``attn_inputs`` / ``kv_cache`` live in
:mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.
"""

# Canonical attn_type ids (mirror C++ KVCacheRegionName enum).
SWA_KV = 7
CSA_KV = 1
HCA_KV = 2
INDEXER_KV = 3
INDEXER_STATE = 4
CSA_STATE = 5
HCA_STATE = 6
