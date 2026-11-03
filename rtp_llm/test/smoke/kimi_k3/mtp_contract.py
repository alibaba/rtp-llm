"""MTP accounting for trained drafts and explicitly untrained smoke fixtures."""
import hashlib
import json
from pathlib import Path

from rtp_llm.utils.speculative_acceptance import checked_counters, summary


def native_mla_draft(checkpoint, target_config):
    """Validate the real nextn checkpoint, without rewriting its configuration."""
    path = Path(checkpoint).resolve(strict=True)
    data = (path / "config.json").read_bytes()
    config = json.loads(data)
    text = config.get("text_config", config)
    target = target_config.get("text_config", target_config)
    fixture = config.get("rtp_llm_fixture", {})
    if (config.get("model_type") != "kimi_k3" or fixture.get("synthetic")
            or fixture.get("trained") is False):
        raise ValueError("K3.1 MTP requires a real native K3 MLA draft; synthetic fallback is disabled")
    source = text.get("num_hidden_layers")
    schedule = text.get("linear_attn_config", {})
    if (type(source) is not int or source < 1
            or text.get("num_nextn_predict_layers") != 1
            or source + 1 not in schedule.get("full_attn_layers", [])
            or source + 1 in schedule.get("kda_layers", [])
            or text.get("mla_use_nope") is not True):
        raise ValueError("K3 MTP requires one NoPE full-MLA nextn layer")
    for field in ("hidden_size", "vocab_size"):
        if type(text.get(field)) is not int or text[field] != target.get(field):
            raise ValueError(f"incompatible target/draft {field}")
    index_data = (path / "model.safetensors.index.json").read_bytes()
    mapping = json.loads(index_data)["weight_map"]
    if any("indexer" in name.lower() for name in mapping):
        raise ValueError("native MLA draft must not contain Indexer weights")
    prefix = f"language_model.model.layers.{source}."
    required = ("embed_tokens.weight", "shared_head.head.weight", "shared_head.norm.weight",
                "enorm.weight", "hnorm.weight", "eh_proj.weight")
    for suffix in required:
        name = prefix + suffix
        if name not in mapping:
            raise ValueError(f"missing trained draft weight: {name}")
    shards = set(mapping.values())
    for shard in shards:
        resolved = (path / shard).resolve(strict=True)
        if not resolved.is_relative_to(path) or not resolved.is_file():
            raise ValueError("draft shard escapes checkpoint")
    return {"kind": "native_k3_mla_mtp", "synthetic": False, "trained": True,
            "checkpoint": str(path), "checkpoint_source_layer": source, "runtime_layer": 0,
            "config_sha256": hashlib.sha256(data).hexdigest(),
            "index_sha256": hashlib.sha256(index_data).hexdigest(),
            "shard_count": len(shards), "tensor_count": len(mapping),
            "query_cp": False, "kv_layout": "page_rr", "kv_dtype": "BASE",
            "compute": "BF16", "experts": "checkpoint_native_MXFP4",
            "global_weight_aliases": []}


def synthetic_fixture(checkpoint):
    path = Path(checkpoint).resolve(strict=True)
    data = (path / "config.json").read_bytes()
    config = json.loads(data)
    fixture = config.get("rtp_llm_fixture", {})
    text = config.get("text_config", config)
    if (config.get("model_type") != "kimi_k31"
            or fixture.get("kind") != "synthetic_dsa_mtp"
            or fixture.get("synthetic") is not True
            or fixture.get("native") is not False
            or fixture.get("trained") is not False
            or text.get("num_nextn_predict_layers") != 1):
        raise ValueError("untrained mode requires an explicit synthetic K3.1 DSA draft")
    return {"kind": "synthetic_untrained", "checkpoint": str(path),
            "config_sha256": hashlib.sha256(data).hexdigest(), "fixture": fixture}


def acceptance_contract(aux, *, untrained=False, required=False):
    metrics = aux.get("speculative_metrics") or {}
    if untrained and (required or int(aux.get("speculative_draft_rounds", 0)) > 0) and metrics.get("version") != 1:
        raise ValueError("synthetic draft requires authoritative A/V counters")
    counters = checked_counters(metrics)
    if counters["excluded_forced_rounds"]:
        raise ValueError("forced acceptance fixture cannot enter normal smoke")
    if untrained and required:
        if (int(aux.get("speculative_draft_rounds", 0)) <= 0
                or counters["proposal_rounds"] <= 0
                or counters["verify_rounds"] <= 0
                or counters["verified_candidates"] <= 0):
            raise ValueError("no actual proposal/verify with valid draft candidates")
    return summary(counters)
