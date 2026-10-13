"""gpt-oss real-checkpoint loader + MoE module comparison (M3 evidence).

Loads the real gpt-oss-20b checkpoint through the full framework loader
chain (CkptDatabase -> GptOssWeightInfo -> get_model_loader ->
load_weights) and compares against a direct safetensors reference:

  1. tensor-level: qkv merge/bias, o/bias, sinks, norms, router(+bias),
     MXFP4 packed blocks/scales/bias deinterleave, embedding, lm_head —
     exact (byte/bit-level after the declared transforms).
  2. module-level: Fp8Fp4MoeLayer forward vs the HF-math fp32 oracle
     (dequantized experts + topk_softmax router + quick-gelu activation).
     The RTP path quantizes activations to FP8, so the comparison reports
     the envelope (cosine / max rel) instead of exact equality.

Path comes from --test_env=GPT_OSS_CKPT (default /mnt/nas1/hf/gpt-oss-20b);
the test skips when the checkpoint or CUDA is unavailable.
"""

import json
import os
import unittest

import torch
from safetensors import safe_open

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_loader.loader import get_model_loader
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.utils.model_weight import W, merge_qkv_b, merge_qkv_hf

CKPT = os.environ.get("GPT_OSS_CKPT", "/mnt/nas1/hf/gpt-oss-20b")
E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _hf_weight(name: str, device="cpu") -> torch.Tensor:
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as reader:
        index = json.load(reader)["weight_map"]
    shard = os.path.join(CKPT, index[name])
    with safe_open(shard, framework="pt", device=device) as f:
        return f.get_tensor(name)


def _decode_mxfp4(packed: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """packed [.., K/2] uint8 + exps [.., K/32] uint8 -> fp32 [.., K]."""
    low = (packed & 0xF).long()
    high = (packed >> 4).long()
    lo = E2M1_MAGS[low & 7] * torch.where((low & 8) > 0, -1.0, 1.0)
    hi = E2M1_MAGS[high & 7] * torch.where((high & 8) > 0, -1.0, 1.0)
    out = torch.stack([lo, hi], dim=-1).flatten(-2)
    scales = torch.pow(2.0, exps.to(torch.float32) - 127.0)
    k = out.shape[-1]
    out = out.reshape(*out.shape[:-1], k // 32, 32) * scales.unsqueeze(-1)
    return out.reshape(*out.shape[:-2], k)


def _ref_moe_forward(x: torch.Tensor, layer: int, cfg: dict) -> torch.Tensor:
    """HF GptOssMLP math in fp32 (weights dequantized exactly)."""
    _, indices, weights, _ = _ref_moe_router(x, layer)
    return _ref_moe_forward_routed(x, layer, cfg, indices, weights)


def _ref_moe_router(x: torch.Tensor, layer: int):
    """fp32 router: biased-logit top-4 + softmax over the selection.

    Also returns the rank4/rank5 margin per token — the engine ranks on
    bf16-GEMM logits (fp32 epilogue), so tokens with a tiny margin can
    legitimately select a different 4th expert than this fp32 oracle.
    """
    p = f"model.layers.{layer}.mlp."
    logits = x @ _hf_weight(p + "router.weight").float().T + _hf_weight(
        p + "router.bias"
    ).float()
    top_value, indices = logits.topk(4, dim=-1)
    weights = torch.softmax(top_value, dim=-1)
    sorted_logits = logits.sort(dim=-1, descending=True).values
    margins = sorted_logits[:, 3] - sorted_logits[:, 4]
    return top_value, indices, weights, margins


def _ref_moe_forward_routed(
    x: torch.Tensor,
    layer: int,
    cfg: dict,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """fp32 expert compute over an externally supplied routing selection."""
    p = f"model.layers.{layer}.mlp."
    blocks = _hf_weight(p + "experts.gate_up_proj_blocks")
    scales = _hf_weight(p + "experts.gate_up_proj_scales")
    b_gu = _hf_weight(p + "experts.gate_up_proj_bias").float()
    blocks_d = _hf_weight(p + "experts.down_proj_blocks")
    scales_d = _hf_weight(p + "experts.down_proj_scales")
    b_d = _hf_weight(p + "experts.down_proj_bias").float()

    e, n, groups, sixteen = blocks.shape
    w_gu = _decode_mxfp4(blocks.reshape(e, n, -1), scales)  # [E, N, K]
    w_d = _decode_mxfp4(blocks_d.reshape(blocks_d.shape[0], n // 2, -1), scales_d)

    y = torch.zeros_like(x, dtype=torch.float32)
    limit = float(cfg.get("swiglu_limit", 7.0))
    for t in range(x.shape[0]):
        for k in range(4):
            eid = int(indices[t, k])
            gu = x[t] @ w_gu[eid].T + b_gu[eid]
            gate, up = gu[0::2], gu[1::2]
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            act = (gate * torch.sigmoid(1.702 * gate)) * (up + 1.0)
            out = act @ w_d[eid].T + b_d[eid]
            y[t] += float(weights[t, k]) * out
    return y


class GptOssWeightsCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(CKPT, "config.json")):
            raise unittest.SkipTest(f"checkpoint not found: {CKPT}")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        from rtp_llm.config.model_args import ModelArgs
        from rtp_llm.config.py_config_modules import LoraConfig
        from rtp_llm.model_factory import ModelFactory
        from rtp_llm.models.gpt_oss.gpt_oss import GptOss
        from rtp_llm.ops import ProfilingDebugLoggingConfig

        cls.cfg_json = json.load(open(os.path.join(CKPT, "config.json")))
        model_args = ModelArgs()
        model_args.ckpt_path = CKPT
        model_args.model_type = "gpt_oss"
        cls.config = ModelFactory.create_model_config(
            model_args,
            lora_config=LoraConfig(),
            kv_cache_config=KVCacheConfig(),
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        )
        weights_info = GptOss.get_weight_cls()(
            model_config=cls.config,
            parallelism_config=ParallelismConfig(),
            hw_kernel_config=HWKernelConfig(),
            kv_cache_config=KVCacheConfig(),
            merge_lora=False,
        )
        database = CkptDatabase(CKPT, None)
        loader = get_model_loader(cls.config, weights_info, None, database)
        cls.weights = loader.load_weights(device="cuda:0", global_weight_aliases={})

    def _layer(self, i: int) -> dict:
        return self.weights.weights[i]

    def test_attention_tensors(self):
        for i in (0, 5, 23):
            lw = self._layer(i)
            p = f"model.layers.{i}.self_attn."
            q = _hf_weight(p + "q_proj.weight")
            k = _hf_weight(p + "k_proj.weight")
            v = _hf_weight(p + "v_proj.weight")
            ref_qkv = merge_qkv_hf([q, k, v])
            self.assertTrue(
                torch.equal(lw[W.attn_qkv_w].cpu(), ref_qkv),
                f"qkv weight mismatch at layer {i}",
            )
            ref_b = merge_qkv_b(
                [_hf_weight(p + "q_proj.bias"), _hf_weight(p + "k_proj.bias"), _hf_weight(p + "v_proj.bias")]
            )
            self.assertTrue(torch.equal(lw[W.attn_qkv_b].cpu(), ref_b))
            ref_o = _hf_weight(p + "o_proj.weight").T.contiguous()
            self.assertTrue(torch.equal(lw[W.attn_o_w].cpu(), ref_o))
            self.assertTrue(
                torch.equal(lw[W.attn_o_b].cpu(), _hf_weight(p + "o_proj.bias"))
            )
            self.assertTrue(
                torch.equal(lw[W.attn_sink].cpu(), _hf_weight(p + "sinks")),
                f"sink mismatch at layer {i}",
            )

    def _assert_equal_report(self, got, ref, label):
        if got.dtype != ref.dtype:
            ref = ref.to(got.dtype)
        if not torch.equal(got.cpu(), ref):
            diff = (got.float().cpu() - ref.float()).abs()
            idx = diff.argmax().item()
            self.fail(
                f"{label} mismatch: dtype={got.dtype} max_diff={diff.max().item():.6f} "
                f"at flat idx {idx}: got={got.flatten()[idx].item()} "
                f"ref={ref.flatten()[idx].item()} shapes={tuple(got.shape)}/{tuple(ref.shape)}"
            )

    def test_norms_and_globals(self):
        lw = self._layer(3)
        self._assert_equal_report(
            lw[W.pre_ln_gamma],
            _hf_weight("model.layers.3.input_layernorm.weight"),
            "pre_ln_gamma",
        )
        self._assert_equal_report(
            lw[W.post_ln_gamma],
            _hf_weight("model.layers.3.post_attention_layernorm.weight"),
            "post_ln_gamma",
        )
        g = self.weights.global_weights
        self._assert_equal_report(
            g[W.final_ln_gamma], _hf_weight("model.norm.weight"), "final_ln_gamma"
        )
        emb = g[W.embedding] if W.embedding in g else g.get(W.embedding)
        self.assertIsNotNone(emb)
        self._assert_equal_report(
            emb[:100],
            _hf_weight("model.embed_tokens.weight")[:100],
            "embedding[:100]",
        )
        lm = g.get(W.lm_head)
        self._assert_equal_report(
            lm[:100], _hf_weight("lm_head.weight")[:100], "lm_head[:100]"
        )

    def test_moe_tensors_exact(self):
        problems = []
        for i in (0, 23):
            lw = self._layer(i)
            p = f"model.layers.{i}.mlp."
            blocks = _hf_weight(p + "experts.gate_up_proj_blocks")
            scales = _hf_weight(p + "experts.gate_up_proj_scales")
            b_gu = _hf_weight(p + "experts.gate_up_proj_bias")
            blocks_d = _hf_weight(p + "experts.down_proj_blocks")
            scales_d = _hf_weight(p + "experts.down_proj_scales")
            b_d = _hf_weight(p + "experts.down_proj_bias")

            def check(name, got, ref):
                got = got.cpu()
                if got.dtype != ref.dtype:
                    ref_c = ref.to(got.dtype)
                else:
                    ref_c = ref
                if not torch.equal(got, ref_c):
                    if got.shape != ref_c.shape:
                        problems.append(
                            f"L{i} {name}: shape {tuple(got.shape)} vs "
                            f"{tuple(ref_c.shape)} dtype {got.dtype}"
                        )
                    else:
                        if got.dtype in (torch.uint8, torch.int8):
                            diff = (got.int() != ref_c.int()).sum().item()
                            problems.append(
                                f"L{i} {name}: {diff} differing bytes "
                                f"({got.dtype} {tuple(got.shape)})"
                            )
                        else:
                            diff = (got.float() - ref_c.float()).abs()
                            problems.append(
                                f"L{i} {name}: max_diff {diff.max().item():.6f} "
                                f"({got.dtype} {tuple(got.shape)})"
                            )

            ref_w1 = torch.cat(
                [blocks[:, 0::2], blocks[:, 1::2]], dim=1
            ).reshape(blocks.shape[0], -1, blocks.shape[2] * blocks.shape[3])
            check("moe_w1", lw[W.moe_w1].view(torch.uint8), ref_w1)
            ref_s1 = torch.cat([scales[:, 0::2], scales[:, 1::2]], dim=1)
            check("moe_s1", lw[W.moe_s1].view(torch.uint8), ref_s1)
            ref_b1 = torch.cat([b_gu[:, 0::2], b_gu[:, 1::2]], dim=1)
            check("moe_b1", lw[W.moe_b1], ref_b1)
            ref_w2 = blocks_d.reshape(
                blocks_d.shape[0], blocks_d.shape[1], -1
            )
            check("moe_w2", lw[W.moe_w2].view(torch.uint8), ref_w2)
            check("moe_s2", lw[W.moe_s2].view(torch.uint8), scales_d)
            check("moe_b2", lw[W.moe_b2], b_d)
            check("moe_gate", lw[W.moe_gate], _hf_weight(p + "router.weight"))
            check(
                "moe_gate_bias", lw[W.moe_gate_bias], _hf_weight(p + "router.bias")
            )
        if problems:
            self.fail(
                "gpt-oss moe tensor mismatches:\n  " + "\n  ".join(problems)
            )

    def test_moe_module_forward_matches_oracle(self):
        layer = 23  # full-attention layer; MoE identical across layers
        lw = dict(self._layer(layer))
        moe = Fp8Fp4MoeLayer(
            layer_id=layer,
            dim=self.config.hidden_size,
            moe_inter_dim=self.config.moe_inter_size,
            n_routed_experts=self.config.expert_num,
            n_activated_experts=self.config.moe_k,
            n_shared_experts=0,
            score_func="topk_softmax",
            route_scale=1.0,
            swiglu_limit=7.0,
            n_hash_layers=0,
            vocab_size=self.config.vocab_size,
            layer_weights=lw,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=512,
            strategy="auto",
            model_type="gpt_oss",
            moe_w1_layout="gate_up",
            swiglu_alpha=1.702,
            swiglu_up_shift=1.0,
        )
        torch.manual_seed(3)
        x = (torch.randn(16, self.config.hidden_size, device="cuda") * 0.5).bfloat16()
        # Capture the engine's routing via the observer (identical math to
        # the observer-free path for gpt-oss: the fused gate-pack lane only
        # covers sqrtsoftplus routers).
        cap: dict = {}
        moe(x, None, observer=lambda tag, t: cap.__setitem__(tag, t.detach()))
        got = cap["final_y"].float().cpu()
        engine_ids = cap["topk_indices"].long().cpu()
        engine_wts = cap["topk_weights"].float().cpu()
        # Routing contract: the engine ranks bf16-GEMM logits (fp32 epilogue),
        # this oracle ranks fp32 logits — selections may differ only on
        # near-tie tokens (rank4/rank5 margin below epsilon). 120b (128
        # experts) legitimately carries one such flip on this input.
        _, ref_ids, _, margins = _ref_moe_router(x.float().cpu(), layer)
        flipped = (engine_ids != ref_ids).any(dim=-1).nonzero().flatten().tolist()
        for t in flipped:
            self.assertLess(
                float(margins[t]),
                1e-3,
                f"engine routing flip at token {t} is not a near-tie "
                f"(fp32 margin {float(margins[t]):.6f})",
            )
        # Expert-compute contract: fp32 oracle over the engine's exact
        # selection isolates loader+executor fidelity from router precision.
        ref = _ref_moe_forward_routed(
            x.float().cpu(), layer, self.cfg_json, engine_ids, engine_wts
        )
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), ref.flatten(), dim=0
        ).item()
        rel = ((got - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        # FP8-activation quantization envelope: wiring bugs produce garbage
        # (cos ~ 0 or rel > 0.5); quant noise stays well under this.
        self.assertGreater(cos, 0.98, f"cosine={cos:.4f}")
        self.assertLess(rel, 0.05, f"rel-l2={rel:.4f}")


if __name__ == "__main__":
    unittest.main()
