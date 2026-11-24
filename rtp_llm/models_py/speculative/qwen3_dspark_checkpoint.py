"""Checkpoint conventions used by Qwen3 DSpARK drafts.

These functions do not select a loader, model runtime or sampling strategy.
"""

import torch


def target_layer_ids(config):
    """Return zero-based transformer layer IDs in execution order."""
    speculators = config.get("speculators_model_type") == "dspark"
    if "aux_hidden_state_layer_ids" in config:
        raw = config["aux_hidden_state_layer_ids"]
        offset = int(speculators)
    elif not speculators and "target_layer_ids" in config:
        raw = config["target_layer_ids"]
        offset = 0
    else:
        raise ValueError("Missing DSpARK target layer IDs")
    if (
        not isinstance(raw, (tuple, list))
        or not raw
        or any(type(i) is not int for i in raw)
    ):
        raise ValueError("DSpARK target layer IDs must be a non-empty integer list")
    result = [i - offset for i in raw]
    if result[0] < 0 or result != sorted(set(result)):
        raise ValueError(
            "DSpARK target layer IDs must be distinct, in order and non-negative after normalization"
        )
    if "target_layer_ids" in config and config["target_layer_ids"] != result:
        raise ValueError("Conflicting DSpARK target-layer metadata")
    return result


def absolute_d2t(name, source, draft_vocab, target_vocab):
    if (
        source.ndim != 1
        or source.numel() != draft_vocab
        or source.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "DSpARK d2t must be an integer map with one row per draft token"
        )
    # Clone even if dtype/device agree: FST may release the iterator storage.
    result = source.to(dtype=torch.int64, copy=True)
    if name == "d2t":
        result += torch.arange(draft_vocab, device=result.device)
    elif name != "model.d2t":
        raise ValueError(f"Unknown DSpARK d2t mapping {name}")
    if torch.any(result < 0) or torch.any(result >= target_vocab):
        raise ValueError("DSpARK d2t references a token outside target vocabulary")
    return result


def is_inference_weight(name):
    # Optional confidence predictor is training metadata for these checkpoints.
    # Main's DSpARK proposer uses the two Markov tables, never this classifier.
    # All other unexpected weight names still get checked by the actual owner.
    return name.removeprefix("model.") not in {
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    }
