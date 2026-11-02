"""MTP accounting for trained drafts and explicitly untrained smoke fixtures."""
import hashlib
import json
from pathlib import Path

from rtp_llm.utils.speculative_acceptance import checked_counters, summary


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
