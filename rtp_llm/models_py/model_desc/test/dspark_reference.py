"""Pure-PyTorch dense reference for the DFlash/DSpark draft forward.

Executable spec for the qwen_3_dflash / qwen_3_dspark model implementation,
anchored line-by-line to the vLLM reference (dspark-work/vllm branch
feature/dspark-whale):
    vllm/model_executor/models/qwen3_dflash.py   (backbone, fc, context-KV pass)
    vllm/model_executor/models/qwen3_dspark.py   (Markov head)
    vllm/v1/worker/gpu/spec_decode/dspark/speculator.py (sequential sampling)

Independent of FlashInfer / paged caches / the RTP-LLM engine: dense GQA
attention, explicit per-layer context K/V, fp32 compute over bf16 weights.

Stages (each independently dumpable — the phase-1 gate-1 UT compares our
implementation against these tensors):
    A. combine:      fused = fc(concat(aux_hidden))                 [T, H]
    B. context KV:   hidden_norm -> per-layer K/V -> k_norm -> RoPE [L, T, nkv, hd]
    C. block fwd:    [anchor + k masks] -> backbone -> head hidden  [1+k, H]
                     base_logits = lm_head(hidden[mask positions])  [k, V]
    D. markov:       sequential greedy w/ low-rank transition bias  [k], [k, V]

Block layout is the speculators "bonus anchor" one (dspark_bonus_anchor=True):
the query block is 1+k wide, the anchor (last verified token) sits at position
committed_len, mask_j at committed_len+j, and predictions are sampled at the k
mask positions.  All positions are real future positions (no sharing).

Dump CLI (synthetic inputs, real weights):
    python3 dspark_reference.py --ckpt /path/to/dspark_draft \
        --out /path/to/golden_dir [--ctx-len 96] [--seed 42] [--device cuda]
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def load_dspark_ckpt(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Load (weights, config) from a draft ckpt dir.

    Accepts both the raw speculators layout (flat names) and the converted
    RTP-LLM layout (model.-prefixed backbone) — keys are normalized to the
    flat form used throughout this module.
    """
    from safetensors import safe_open

    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    # Converted config flattens transformer_layer_config to top level.
    layer_cfg = cfg.get("transformer_layer_config", cfg)

    weights: Dict[str, torch.Tensor] = {}
    with safe_open(
        os.path.join(path, "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        for name in f.keys():
            key = name[len("model."):] if name.startswith("model.") else name
            weights[key] = f.get_tensor(name)

    rope_params = layer_cfg.get("rope_parameters") or {}
    norm_cfg = {
        "num_hidden_layers": layer_cfg["num_hidden_layers"],
        "hidden_size": layer_cfg["hidden_size"],
        "num_attention_heads": layer_cfg["num_attention_heads"],
        "num_key_value_heads": layer_cfg["num_key_value_heads"],
        "head_dim": layer_cfg["head_dim"],
        "rms_norm_eps": layer_cfg["rms_norm_eps"],
        "vocab_size": layer_cfg["vocab_size"],
        "rope_theta": layer_cfg.get("rope_theta", rope_params.get("rope_theta", 1000000)),
        "aux_hidden_state_layer_ids": cfg["aux_hidden_state_layer_ids"],
        "mask_token_id": cfg["mask_token_id"],
        "speculative_tokens": cfg.get("speculative_tokens")
        or (cfg.get("speculators_config", {}).get("proposal_methods") or [{}])[0].get(
            "speculative_tokens"
        ),
        "markov_rank": cfg.get("markov_rank", 0),
    }
    return weights, norm_cfg


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 (vLLM RMSNorm.forward_native semantics)."""
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * weight.float()


def apply_rope_neox(
    x: torch.Tensor, positions: torch.Tensor, head_dim: int, theta: float
) -> torch.Tensor:
    """Neox-style RoPE in fp32.  x: [T, n_heads, head_dim], positions: [T]."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, dtype=torch.float32, device=x.device) / half)
    )
    freqs = positions.float()[:, None] * inv_freq[None, :]  # [T, half]
    cos = freqs.cos()[:, None, :]  # [T, 1, half]
    sin = freqs.sin()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class DSparkDraftReference:
    """Dense reference over raw ckpt weights.  All compute fp32."""

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        cfg: dict,
        device: str = "cpu",
    ) -> None:
        self.w = {k: v.to(device) for k, v in weights.items()}
        self.cfg = cfg
        self.device = device
        self.L = cfg["num_hidden_layers"]
        self.H = cfg["hidden_size"]
        self.nq = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["rope_theta"])
        self.group = self.nq // self.nkv

    # ---- Stage A: feature combine ------------------------------------

    def combine_hidden_states(self, aux_concat: torch.Tensor) -> torch.Tensor:
        """fc over the layer-concatenated aux hidden states.

        aux_concat: [T, len(aux_ids) * H], ordered by aux_hidden_state_layer_ids.
        Returns fused features [T, H] (fp32).
        """
        return F.linear(aux_concat.float(), self.w["fc.weight"].float())

    # ---- Stage B: context (feature) KV projection ---------------------

    def project_context_kv(
        self, fused: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """hidden_norm -> per-layer K/V -> per-head k_norm -> RoPE(K only).

        fused: [T, H] fused features at context positions; positions: [T].
        Returns (K, V) each [L, T, nkv, hd] fp32.  V is NOT rotated.
        """
        normed = rms_norm(fused, self.w["hidden_norm.weight"], self.eps)
        ks, vs = [], []
        for i in range(self.L):
            k = F.linear(normed, self.w[f"layers.{i}.self_attn.k_proj.weight"].float())
            v = F.linear(normed, self.w[f"layers.{i}.self_attn.v_proj.weight"].float())
            k = k.view(-1, self.nkv, self.hd)
            v = v.view(-1, self.nkv, self.hd)
            k = rms_norm(k, self.w[f"layers.{i}.self_attn.k_norm.weight"], self.eps)
            k = apply_rope_neox(k, positions, self.hd, self.theta)
            ks.append(k)
            vs.append(v)
        return torch.stack(ks), torch.stack(vs)

    # ---- Stage C: non-causal block forward -----------------------------

    def _attention(
        self,
        q: torch.Tensor,  # [B, nq, hd]
        k: torch.Tensor,  # [S, nkv, hd]
        v: torch.Tensor,  # [S, nkv, hd]
    ) -> torch.Tensor:
        """Dense non-causal GQA attention: every query sees every key."""
        k = k.repeat_interleave(self.group, dim=1)
        v = v.repeat_interleave(self.group, dim=1)
        scale = 1.0 / math.sqrt(self.hd)
        logits = torch.einsum("qhd,khd->hqk", q, k) * scale
        attn = torch.softmax(logits, dim=-1)
        return torch.einsum("hqk,khd->qhd", attn, v)

    def block_forward(
        self,
        input_ids: torch.Tensor,  # [B] = [anchor] + k * [mask_token_id]
        positions: torch.Tensor,  # [B] = committed_len + [0..k]
        ctx_k: torch.Tensor,  # [L, T, nkv, hd] from project_context_kv
        ctx_v: torch.Tensor,  # [L, T, nkv, hd]
    ) -> torch.Tensor:
        """Backbone over the query block.  Returns final hidden [B, H] fp32.

        The block's own K/V are appended after the context K/V (the paged-cache
        equivalent: block KV written at future positions).  All queries attend
        the full [context + block] window (non-causal).
        """
        x = self.w["embed_tokens.weight"].float()[input_ids]
        residual: Optional[torch.Tensor] = None
        for i in range(self.L):
            p = f"layers.{i}."
            if residual is None:
                residual = x
                normed = rms_norm(x, self.w[p + "input_layernorm.weight"], self.eps)
            else:
                x = x + residual
                residual = x
                normed = rms_norm(x, self.w[p + "input_layernorm.weight"], self.eps)

            q = F.linear(normed, self.w[p + "self_attn.q_proj.weight"].float())
            k = F.linear(normed, self.w[p + "self_attn.k_proj.weight"].float())
            v = F.linear(normed, self.w[p + "self_attn.v_proj.weight"].float())
            q = q.view(-1, self.nq, self.hd)
            k = k.view(-1, self.nkv, self.hd)
            v = v.view(-1, self.nkv, self.hd)
            q = rms_norm(q, self.w[p + "self_attn.q_norm.weight"], self.eps)
            k = rms_norm(k, self.w[p + "self_attn.k_norm.weight"], self.eps)
            q = apply_rope_neox(q, positions, self.hd, self.theta)
            k = apply_rope_neox(k, positions, self.hd, self.theta)

            k_all = torch.cat([ctx_k[i], k], dim=0)
            v_all = torch.cat([ctx_v[i], v], dim=0)
            attn_out = self._attention(q, k_all, v_all).reshape(-1, self.nq * self.hd)
            attn_out = F.linear(attn_out, self.w[p + "self_attn.o_proj.weight"].float())

            x = attn_out + residual
            residual = x
            normed = rms_norm(x, self.w[p + "post_attention_layernorm.weight"], self.eps)
            gate = F.linear(normed, self.w[p + "mlp.gate_proj.weight"].float())
            up = F.linear(normed, self.w[p + "mlp.up_proj.weight"].float())
            x = F.linear(F.silu(gate) * up, self.w[p + "mlp.down_proj.weight"].float())

        assert residual is not None
        return rms_norm(x + residual, self.w["norm.weight"], self.eps)

    def compute_base_logits(self, head_hidden: torch.Tensor) -> torch.Tensor:
        """lm_head over the k mask-position hiddens (bonus-anchor layout).

        head_hidden: [1+k, H] block output; rows 1..k are the mask positions.
        Returns [k, V] fp32.
        """
        return F.linear(head_hidden[1:], self.w["lm_head.weight"].float())

    # ---- Stage D: sequential Markov sampling ---------------------------

    def markov_greedy(
        self, base_logits: torch.Tensor, anchor_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-to-right greedy with the low-rank transition bias.

        bias_i = markov_w2 @ markov_w1[prev]; logits_i = base_i + bias_i;
        prev_0 = anchor.  Returns (tokens [k] int64, corrected_logits [k, V]).
        """
        w1 = self.w["markov_head.markov_w1.weight"].float()
        w2 = self.w["markov_head.markov_w2.weight"].float()
        k = base_logits.shape[0]
        tokens: List[int] = []
        corrected = torch.empty_like(base_logits)
        prev = anchor_id
        for i in range(k):
            bias = F.linear(w1[prev].unsqueeze(0), w2).squeeze(0)  # [V]
            corrected[i] = base_logits[i] + bias
            prev = int(corrected[i].argmax().item())
            tokens.append(prev)
        return torch.tensor(tokens, dtype=torch.int64), corrected

    # ---- Full pipeline --------------------------------------------------

    def run(
        self,
        aux_concat: torch.Tensor,  # [T, n_aux * H]
        anchor_id: int,
        committed_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        T = aux_concat.shape[0]
        committed_len = T if committed_len is None else committed_len
        k = self.cfg["speculative_tokens"]
        mask_id = self.cfg["mask_token_id"]
        dev = self.device

        ctx_positions = torch.arange(T, dtype=torch.int64, device=dev)
        fused = self.combine_hidden_states(aux_concat.to(dev))
        ctx_k, ctx_v = self.project_context_kv(fused, ctx_positions)

        input_ids = torch.tensor([anchor_id] + [mask_id] * k, device=dev)
        positions = committed_len + torch.arange(1 + k, dtype=torch.int64, device=dev)
        head_hidden = self.block_forward(input_ids, positions, ctx_k, ctx_v)
        base_logits = self.compute_base_logits(head_hidden)

        out = {
            "fused_features": fused,
            "ctx_k": ctx_k,
            "ctx_v": ctx_v,
            "head_hidden": head_hidden,
            "base_logits": base_logits,
        }
        if self.cfg.get("markov_rank"):
            tokens, corrected = self.markov_greedy(base_logits, anchor_id)
            out["draft_tokens"] = tokens
            out["corrected_logits"] = corrected
        return out


def main() -> int:
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="draft ckpt dir (raw or converted)")
    parser.add_argument("--out", required=True, help="golden output dir")
    parser.add_argument("--ctx-len", type=int, default=96, help="synthetic context length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-id", type=int, default=9707)  # arbitrary real token
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights, cfg = load_dspark_ckpt(args.ckpt)
    ref = DSparkDraftReference(weights, cfg, device=args.device)

    torch.manual_seed(args.seed)
    n_aux = len(cfg["aux_hidden_state_layer_ids"])
    # Synthetic aux hidden states: unit-scale gaussian is in-family for
    # RMSNorm'd residual streams and keeps softmaxes well-conditioned.
    aux_concat = torch.randn(args.ctx_len, n_aux * cfg["hidden_size"], dtype=torch.float32)

    with torch.no_grad():
        out = ref.run(aux_concat, args.anchor_id)

    os.makedirs(args.out, exist_ok=True)
    tensors = {"aux_concat": aux_concat}
    tensors.update({k: v.cpu().contiguous() for k, v in out.items()})
    save_file(tensors, os.path.join(args.out, "dspark_golden.safetensors"))

    manifest = {
        "ckpt": os.path.abspath(args.ckpt),
        "seed": args.seed,
        "ctx_len": args.ctx_len,
        "anchor_id": args.anchor_id,
        "committed_len": args.ctx_len,
        "block_layout": "bonus_anchor: [anchor] + k masks at committed_len + [0..k]",
        "speculative_tokens": cfg["speculative_tokens"],
        "mask_token_id": cfg["mask_token_id"],
        "aux_order": "concat by aux_hidden_state_layer_ids "
        + str(cfg["aux_hidden_state_layer_ids"]),
        "dtype_discipline": "fp32 compute over bf16 weights; fp32 dumps",
        "draft_tokens": out["draft_tokens"].tolist() if "draft_tokens" in out else None,
        "tensors": {k: list(v.shape) for k, v in tensors.items()},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"golden dumped to {args.out}")
    if "draft_tokens" in out:
        print(f"greedy draft tokens: {out['draft_tokens'].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
