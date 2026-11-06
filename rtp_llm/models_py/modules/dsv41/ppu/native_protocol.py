"""Framework names at the native boundary; PPU codec names remain stable."""

import hashlib
import json

from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy

NATIVE_CACHE_PROTOCOL = "v41-ppu-native-cache-v2"
NATIVE_CACHE_KEY_HASH_SEED = 0x4453563431505032
FRAMEWORK_POOL_TAGS = {
    "global_kv_1": "hca_kv",
    "global_kv_2": "csa_kv",
    "indexer_kv": "indexer_kv",
    "swa_kv": "swa_kv",
    "csa_state": "csa_state",
}


def framework_pool_tag(owner, tag, *, geometry, policy):
    return policy.framework_tag(owner, tag, geometry)


def native_cache_key_hash_seed(config=None):
    policy = PrefillPolicy.from_config(config)
    config = config or {}
    identity = config.get("ppu_model_identity", "target")
    capture = tuple(config.get("ppu_target_capture_layer_ids") or ())
    if identity == "dspark-draft" or capture:
        payload = json.dumps(
            {
                "base": NATIVE_CACHE_KEY_HASH_SEED,
                "policy": [policy.ced, policy.bounded_replay, policy.encoder_replay],
                "model": identity,
                "features": list(
                    capture or config.get("dspark_target_layer_ids") or ()
                ),
                "draft_layers": int(config.get("num_nextn_predict_layers", 3)),
                "layout": "independent-feature-swa-v1",
            },
            sort_keys=True,
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
            (1 << 63) - 1
        )
    if policy.encoder_replay:
        return 0x4453563431504531
    if policy.bounded_replay:
        return 0x4453563431504231
    return NATIVE_CACHE_KEY_HASH_SEED


def native_cache_protocol(config=None):
    policy = PrefillPolicy.from_config(config)
    contract = {
        "version": NATIVE_CACHE_PROTOCOL,
        "canonical_to_framework": dict(FRAMEWORK_POOL_TAGS),
        "global_row_bytes": 288,
        "indexer_row_bytes": 68,
        "indexer_layout": "planar-raw-token-capacity",
        "swa_row_bytes": 528,
        "csa_state_elements": 1024,
        "source_layout": "replicated-logical-cp-cache",
        "cache_key_hash_seed": native_cache_key_hash_seed(config),
        "model_identity": (config or {}).get("ppu_model_identity", "target"),
    }
    if policy.bounded_replay:
        contract["decoder_swa_tag"] = "decoder_swa_kv"
        contract["decoder_replay"] = "last-window-original-positions-v1"
    if policy.encoder_replay:
        contract["encoder_replay"] = "full-history-global-prefix-readonly-v1"
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    return contract
