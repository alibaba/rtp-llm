"""Framework names at the native boundary; PPU codec names remain stable."""

import hashlib
import json

NATIVE_CACHE_PROTOCOL = "v41-ppu-native-cache-v2"
NATIVE_CACHE_KEY_HASH_SEED = 0x4453563431505032
FRAMEWORK_POOL_TAGS = {
    "global_kv_1": "hca_kv",
    "global_kv_2": "csa_kv",
    "indexer_kv": "indexer_kv",
    "swa_kv": "swa_kv",
    "csa_state": "csa_state",
}


def native_cache_protocol():
    contract = {
        "version": NATIVE_CACHE_PROTOCOL,
        "canonical_to_framework": dict(FRAMEWORK_POOL_TAGS),
        "global_row_bytes": 288,
        "indexer_row_bytes": 68,
        "indexer_layout": "planar-raw-token-capacity",
        "swa_row_bytes": 528,
        "csa_state_elements": 1024,
        "source_layout": "replicated-logical-cp-cache",
        "cache_key_hash_seed": NATIVE_CACHE_KEY_HASH_SEED,
    }
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    return contract
