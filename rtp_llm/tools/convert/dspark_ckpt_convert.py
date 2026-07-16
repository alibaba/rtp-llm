"""Convert a speculators-format DFlash/DSpark draft checkpoint to RTP-LLM layout.

Source layout (speculators, e.g. dspark_draft/):
    config.json                 speculators-format (DSparkSpeculatorConfig dump)
    model.safetensors           flat names: embed_tokens.weight, layers.N.*,
                                fc.weight, hidden_norm.weight, norm.weight,
                                lm_head.weight, markov_head.markov_w{1,2}.weight,
                                confidence_head.proj.{weight,bias}

Output layout (HF-style, what the qwen_3_dflash / qwen_3_dspark model types load):
    config.json                 flattened Qwen3 config + dspark extras at top level
    model.safetensors           model.-prefixed backbone names (mirrors the
                                qwen_3_moe_eagle3 precedent: backbone under
                                model.*, fc.weight / lm_head.weight top-level)

Weight name mapping:
    embed_tokens.weight              -> model.embed_tokens.weight
    norm.weight                      -> model.norm.weight
    hidden_norm.weight               -> model.hidden_norm.weight
    layers.{i}.*                     -> model.layers.{i}.*
    fc.weight                        -> fc.weight            (kept top-level)
    lm_head.weight                   -> lm_head.weight       (kept top-level)
    markov_head.markov_w{1,2}.weight -> unchanged            (top-level)
    confidence_head.*                -> unchanged (kept; phase-1 loader maps
                                        nothing to it, so it is simply ignored)

TP note: no sharding is baked in.  RTP-LLM splits at load time via the model's
weight styles; the vLLM reference replicates fc on every rank (ReplicatedLinear)
and vocab-shards markov_w1/markov_w2/lm_head, both of which are load-time
decisions.

Usage:
    python3 dspark_ckpt_convert.py --src /path/to/dspark_draft --dst /path/to/out
"""

import argparse
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BACKBONE_TOP_LEVEL = {"embed_tokens.weight", "norm.weight", "hidden_norm.weight"}
KEEP_TOP_LEVEL_PREFIXES = ("fc.", "lm_head.", "markov_head.", "confidence_head.", "d2t", "t2d")

# Per-layer tensors of a Qwen3 (qk-norm) decoder layer.
LAYER_TENSORS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def map_name(name: str) -> str:
    if name in BACKBONE_TOP_LEVEL or name.startswith("layers."):
        return "model." + name
    if name.startswith(KEEP_TOP_LEVEL_PREFIXES):
        return name
    raise ValueError(f"unexpected tensor name in source checkpoint: {name}")


def build_config(src_cfg: dict, has_markov: bool) -> dict:
    layer_cfg = src_cfg["transformer_layer_config"]
    spec_cfg = src_cfg.get("speculators_config", {})
    proposal = (spec_cfg.get("proposal_methods") or [{}])[0]

    rope_parameters = layer_cfg.get("rope_parameters") or {}
    out = dict(layer_cfg)  # flatten HF Qwen3 fields to top level
    out.pop("rope_parameters", None)
    out["rope_theta"] = rope_parameters.get("rope_theta", 1000000)
    out["torch_dtype"] = src_cfg.get("dtype", "bfloat16")
    out["architectures"] = (
        ["Qwen3DSparkForCausalLM"] if has_markov else ["Qwen3DFlashForCausalLM"]
    )

    # DFlash/DSpark extras consumed by the RTP-LLM draft model config.
    for key in (
        "aux_hidden_state_layer_ids",
        "block_size",
        "mask_token_id",
        "draft_vocab_size",
        "sliding_window_non_causal",
        "markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "tie_word_embeddings",
        "target_hidden_size",
    ):
        if key in src_cfg:
            out[key] = src_cfg[key]
    # Static proposal width: speculative_tokens (k).  The query block is
    # 1 + k wide (bonus anchor + k masks); k + 1 <= block_size must hold.
    out["speculative_tokens"] = proposal.get("speculative_tokens")
    out["proposal_type"] = proposal.get("proposal_type")

    # Provenance (not consumed by the loader).
    out["converted_from"] = {
        "speculators_model_type": src_cfg.get("speculators_model_type"),
        "speculators_version": src_cfg.get("speculators_version"),
        "verifier": spec_cfg.get("verifier", {}).get("name_or_path"),
    }
    return out


def validate_source(names: set, num_layers: int) -> None:
    missing = []
    for base in ("embed_tokens.weight", "norm.weight", "hidden_norm.weight",
                 "fc.weight", "lm_head.weight"):
        if base not in names:
            missing.append(base)
    for i in range(num_layers):
        for t in LAYER_TENSORS:
            k = f"layers.{i}.{t}"
            if k not in names:
                missing.append(k)
    if missing:
        raise ValueError(f"source checkpoint is missing tensors: {missing[:10]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="speculators-format draft ckpt dir")
    parser.add_argument("--dst", required=True, help="output dir (created if absent)")
    parser.add_argument(
        "--drop-confidence-head",
        action="store_true",
        help="omit confidence_head.* tensors from the output",
    )
    args = parser.parse_args()

    with open(os.path.join(args.src, "config.json")) as f:
        src_cfg = json.load(f)
    src_st = os.path.join(args.src, "model.safetensors")

    num_layers = src_cfg["transformer_layer_config"]["num_hidden_layers"]

    tensors = {}
    has_markov = False
    with safe_open(src_st, framework="pt", device="cpu") as f:
        names = set(f.keys())
        validate_source(names, num_layers)
        for name in sorted(names):
            if args.drop_confidence_head and name.startswith("confidence_head."):
                continue
            if name.startswith("markov_head."):
                has_markov = True
            tensors[map_name(name)] = f.get_tensor(name)

    # Shape sanity on the load-bearing extras.
    hidden = src_cfg["transformer_layer_config"]["hidden_size"]
    n_aux = len(src_cfg["aux_hidden_state_layer_ids"])
    fc = tensors["fc.weight"]
    assert fc.shape == (hidden, hidden * n_aux), (
        f"fc.weight shape {tuple(fc.shape)} != ({hidden}, {hidden * n_aux}); "
        "torch Linear [out, in] convention expected"
    )
    if has_markov:
        rank = src_cfg["markov_rank"]
        vocab = src_cfg["transformer_layer_config"]["vocab_size"]
        draft_vocab = src_cfg.get("draft_vocab_size") or vocab
        w1 = tensors["markov_head.markov_w1.weight"]
        w2 = tensors["markov_head.markov_w2.weight"]
        assert w1.shape == (vocab, rank), f"markov_w1 {tuple(w1.shape)}"
        assert w2.shape == (draft_vocab, rank), f"markov_w2 {tuple(w2.shape)}"

    os.makedirs(args.dst, exist_ok=True)
    out_cfg = build_config(src_cfg, has_markov)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2, ensure_ascii=False)

    dst_st = os.path.join(args.dst, "model.safetensors")
    save_file(tensors, dst_st, metadata={"format": "pt"})

    # Post-write verification: reopen and compare inventory + shapes + dtypes.
    with safe_open(dst_st, framework="pt", device="cpu") as f:
        out_names = set(f.keys())
        assert out_names == set(tensors.keys()), "output tensor inventory mismatch"
        for name in sorted(out_names):
            t = f.get_tensor(name)
            assert t.shape == tensors[name].shape and t.dtype == tensors[name].dtype
            if name in ("fc.weight", "lm_head.weight"):
                assert torch.equal(t, tensors[name]), f"{name} round-trip mismatch"

    arch = out_cfg["architectures"][0]
    print(f"converted {len(tensors)} tensors ({arch}) -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
