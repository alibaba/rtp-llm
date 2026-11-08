"""Framework names at the native boundary; PPU codec names remain stable."""

import hashlib
import json
import os

from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import (
    PrefillPolicy,
    SharedPrefillPolicy,
)

NATIVE_CACHE_PROTOCOL = "v41-ppu-native-cache-v2"
FRAMEWORK_POOL_TAGS = {
    "global_kv_1": "hca_kv",
    "global_kv_2": "csa_kv",
    "indexer_kv": "indexer_kv",
    "swa_kv": "swa_kv",
    "csa_state": "csa_state",
}


def framework_pool_tag(owner, tag, *, geometry, policy):
    return policy.framework_tag(owner, tag, geometry)


def shared_cache_placement(config=None):
    """One compatible P-source/D-destination identity, independent of rank."""
    frozen = (config or {}).get("_ppu_cp_cache")
    if frozen is None:
        enabled = os.environ.get("PREFILL_CP_KV_CACHE_SHARDED", "0").lower() in ("1", "true", "yes", "on")
        size = int(os.environ.get("PREFILL_CP_SIZE", "1")) if enabled else 1
    else:
        enabled, size = bool(frozen["enabled"]), int(frozen["size"])
    if not enabled:
        return None
    if size <= 1:
        raise ValueError("sharded native identity requires source CP size > 1")
    return {
        "revision": "ppu-native-cp-placement-v1", "source_cp_size": size,
        "source_full": "physical-block-round-robin:b%CP,local=b//CP;kernel-pages-within-physical-block",
        "source_swa": "compact-native-full-stride-byte-slices",
        "source_csa": "compact-native-aligned-payload-slices",
        "destination": "DP-business-owner-full-native-pools",
        "transport": "native-CacheStore-RDMA",
    }


def native_cache_key_hash_seed(config=None):
    config = config or {}
    profile = {
        "layout": shared_cache_placement(config) or "shared-paged-replicated-ppu-v1",
        "policy": SharedPrefillPolicy.from_config(config).__dict__,
        "model": config.get("ppu_model_identity", "target"),
        "capture": tuple(config.get("ppu_target_capture_layer_ids") or ()),
    }
    speculative = _shared_dspark_contract(config)
    if speculative is not None:
        profile["dspark"] = speculative
    payload = json.dumps(profile, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


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
    contract["shared_profile"] = SharedPrefillPolicy.from_config(config).__dict__
    contract["model_path"] = "shared"
    placement = shared_cache_placement(config)
    if placement is not None:
        contract["placement"] = placement
        contract["source_layout"] = "per-tag-native-cp-sharded"
        contract["destination_layout"] = placement["destination"]
    if contract["shared_profile"]["max_rollback_tokens"]:
        contract["speculative_state"] = _shared_dspark_contract(config)
    if policy.bounded_replay:
        contract["decoder_replay"] = "compact140-original-positions-v1"
    contract["sha256"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()
    return contract


def _shared_dspark_contract(config):
    from rtp_llm.models_py.modules.dsv41.ppu.shared_dspark_modes import SharedDSparkMode

    mode = SharedDSparkMode.from_config(config)
    return mode.protocol() if mode is not None else None
