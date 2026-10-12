"""Upstream Block/V4Transformer integration with real SM120 component kernels."""

import json
import os
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.engram import (
    Engram,
    EngramLayout,
    HostEngramEmbedding,
    NgramHashState,
    _PinnedSharedTable,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
)
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCHead, DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.moe.shared_expert import MXFP8SharedExpertExecutor
from rtp_llm.models_py.modules.dsv4.prefill.forward import forward_layers
from rtp_llm.models_py.modules.dsv4.transformer import V4Args, V4Transformer
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear
from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts
from rtp_llm.utils.model_weight import W


def fp8(n, k, block=32):
    weight = (torch.randn(n, k, device="cuda") * 0.025).to(torch.float8_e4m3fn)
    scales = torch.ones(n // block, k // block, device="cuda").to(torch.float8_e8m0fnu)
    return weight, scales


def fixture(experts=4, *, full_schedule=False, max_seq_len=256):
    ratios = [0, 0] + [2] * 18 + [1] * 20 if full_schedule else [0, 0, 2, 2, 1, 1]
    kv_sources = [2, 8, 14, 20] if full_schedule else [2, 4]
    index_sources = [2, 8, 14, 20, 24, 28, 32, 36] if full_schedule else [2, 4]
    args = V4Args(
        vocab_size=128,
        dim=512,
        n_heads=4,
        n_layers=len(ratios),
        n_mtp_layers=0,
        q_lora_rank=256,
        o_groups=2,
        o_lora_rank=256,
        compress_ratios=ratios,
        index_n_heads=32,
        moe_inter_dim=512,
        n_routed_experts=experts,
        n_activated_experts=2,
        n_hash_layers=0,
        norm_eps=1e-20,
        max_batch_size=2,
        max_seq_len=max_seq_len,
        max_tokens_per_rank=max_seq_len,
        v41_config={
            "num_hidden_layers": len(ratios),
            "head_dim": 512,
            "index_head_dim": 128,
            "sliding_window": 128,
            "compress_ratios": ratios,
            "kv_source_layer_ids": kv_sources,
            "index_source_layer_ids": index_sources,
            "candidate_source_layer_id": kv_sources[-1],
            "candidate_topk_blocks": 2048 if full_schedule else 0,
            "candidate_block_size": 8 if full_schedule else 0,
        },
    )
    weights = []
    for layer_id in range(args.n_layers):
        w = {}
        for name, n, k in (
            ("wq_a", 256, 512),
            ("wq_b", 2048, 256),
            ("wkv", 512, 512),
            ("wo_a", 512, 1024),
            ("wo_b", 512, 512),
        ):
            w[getattr(W, f"v4_attn_{name}_w")], w[getattr(W, f"v4_attn_{name}_s")] = (
                fp8(n, k)
            )
        w[W.v4_attn_q_norm] = torch.ones(256, device="cuda", dtype=torch.bfloat16)
        w[W.v4_attn_kv_norm] = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        w[W.v4_attn_sink] = torch.linspace(-1, 1, 4, device="cuda")
        for part in ("attn", "ffn"):
            w[getattr(W, f"v4_{part}_norm")] = torch.ones(
                512, device="cuda", dtype=torch.bfloat16
            )
            w[getattr(W, f"v4_hc_{part}_fn")] = (
                torch.randn(24, 2048, device="cuda") * 0.01
            )
            w[getattr(W, f"v4_hc_{part}_base")] = torch.randn(24, device="cuda") * 0.1
            w[getattr(W, f"v4_hc_{part}_scale")] = torch.tensor(
                [0.5, 0.6, 0.7], device="cuda"
            )
        w[W.v4_router_w] = (torch.randn(experts, 512, device="cuda") * 0.025).bfloat16()
        w[W.v4_router_bias] = torch.randn(experts, device="cuda") * 0.1
        w[W.v4_shared_w13_w], w[W.v4_shared_w13_s] = fp8(1024, 512)
        w[W.v4_shared_w2_w], w[W.v4_shared_w2_s] = fp8(512, 512)
        for name in ("w1", "w2", "w3"):
            w[getattr(W, f"v4_routed_{name}_w")] = torch.randint(
                0, 256, (experts, 512, 256), device="cuda", dtype=torch.uint8
            )
            w[getattr(W, f"v4_routed_{name}_s")] = torch.full(
                (experts, 512, 16), 121, device="cuda", dtype=torch.uint8
            )
        if layer_id in kv_sources:
            w[W.v4_compressor_wkv] = (
                torch.randn(512, 512, device="cuda") * 0.025
            ).bfloat16()
            w[W.v4_compressor_wgate] = (
                torch.randn(512, 512, device="cuda") * 0.025
            ).bfloat16()
            w[W.v4_compressor_norm] = torch.ones(
                512, device="cuda", dtype=torch.bfloat16
            )
            w[W.v41_indexer_wk] = (
                torch.randn(128, 512, device="cuda") * 0.025
            ).bfloat16()
            w[W.v41_indexer_k_norm] = torch.ones(
                128, device="cuda", dtype=torch.bfloat16
            )
        if layer_id in index_sources:
            w[W.v4_indexer_wq_b_w], w[W.v4_indexer_wq_b_s] = fp8(4096, 256)
            w[W.v4_indexer_weights_proj_w] = torch.randn(
                32, 512, device="cuda", dtype=torch.bfloat16
            )
        weights.append(w)
    global_weights = {
        W.embedding: torch.randn(128, 512, device="cuda", dtype=torch.bfloat16),
        W.final_ln_gamma: torch.ones(512, device="cuda", dtype=torch.bfloat16),
        W.lm_head: torch.randn(128, 512, device="cuda", dtype=torch.bfloat16),
    }
    return args, SimpleNamespace(weights=weights, global_weights=global_weights)


def build(args, weights):
    model = V4Transformer(args, weights)
    model._prefill_ws_q_rows = 256
    model._prefill_ws_q_dim = args.n_heads * args.head_dim
    for layer in model.layers:
        layer.attn.reset_rope_cache(torch.device("cuda"))
    return model, NativeTaggedCache([layer.attn for layer in model.layers])


def nas_fixture():
    from safetensors import safe_open

    root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
    cfg = json.loads((root / "config.json").read_text())["text_config"]
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    experts = 8

    def read(name, rows=None):
        with safe_open(root / index[name], framework="pt", device="cpu") as file:
            value = (
                file.get_tensor(name) if rows is None else file.get_slice(name)[:rows]
            )
        return value.cuda()

    weights = {}
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        for kind, suffix in (("w", "weight"), ("s", "scale")):
            weights[getattr(W, f"v4_attn_{name}_{kind}")] = read(
                f"layers.0.attn.{name}.{suffix}"
            )
    for key, suffix in (
        (W.v4_attn_q_norm, "q_norm.weight"),
        (W.v4_attn_kv_norm, "kv_norm.weight"),
        (W.v4_attn_sink, "attn_sink"),
    ):
        value = read(f"layers.0.attn.{suffix}")
        weights[key] = value if key == W.v4_attn_sink else value.bfloat16()
    for part in ("attn", "ffn"):
        weights[getattr(W, f"v4_{part}_norm")] = read(
            f"layers.0.{part}_norm.weight"
        ).bfloat16()
        for name in ("fn", "base", "scale"):
            weights[getattr(W, f"v4_hc_{part}_{name}")] = read(
                f"layers.0.hc_{part}_{name}"
            ).float()
    weights[W.v4_router_w] = read("layers.0.ffn.gate.weight", experts).bfloat16()
    weights[W.v4_router_bias] = read("layers.0.ffn.gate.bias", experts).float()
    for kind, suffix in (("w", "weight"), ("s", "scale")):
        weights[getattr(W, f"v4_shared_w13_{kind}")] = torch.cat(
            [
                read(f"layers.0.ffn.shared_experts.{name}.{suffix}")
                for name in ("w1", "w3")
            ]
        )
        weights[getattr(W, f"v4_shared_w2_{kind}")] = read(
            f"layers.0.ffn.shared_experts.w2.{suffix}"
        )
        for name in ("w1", "w2", "w3"):
            # Stack raw bytes; FP4 and UE8M0 need no conversion on the host.
            weights[getattr(W, f"v4_routed_{name}_{kind}")] = torch.stack(
                [
                    read(f"layers.0.ffn.experts.{expert}.{name}.{suffix}").view(
                        torch.uint8
                    )
                    for expert in range(experts)
                ]
            )
    args = V4Args(
        vocab_size=128,
        dim=cfg["hidden_size"],
        n_heads=cfg["num_attention_heads"],
        n_layers=1,
        n_mtp_layers=0,
        q_lora_rank=cfg["q_lora_rank"],
        o_groups=cfg["o_groups"],
        o_lora_rank=cfg["o_lora_rank"],
        compress_ratios=[0],
        index_n_heads=cfg["index_n_heads"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_routed_experts=experts,
        n_activated_experts=cfg["num_experts_per_tok"],
        n_hash_layers=0,
        norm_eps=cfg["rms_norm_eps"],
        max_batch_size=2,
        max_seq_len=256,
        max_tokens_per_rank=256,
        v41_config=cfg,
    )
    # The bounded block check uses synthetic embed/head tensors and the first
    # eight real experts. It cannot qualify full-model logits or routing.
    global_weights = {
        W.embedding: torch.randn(128, args.dim, device="cuda", dtype=torch.bfloat16),
        W.final_ln_gamma: torch.ones(args.dim, device="cuda", dtype=torch.bfloat16),
        W.lm_head: torch.randn(128, args.dim, device="cuda", dtype=torch.bfloat16),
    }
    return args, SimpleNamespace(weights=[weights], global_weights=global_weights)


def run(model, cache, ids, lengths, prefixes=None, *, fast=True):
    prefixes = [0] * len(lengths) if prefixes is None else prefixes
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    pref = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
    positions = torch.cat(
        [torch.arange(p, p + n, device="cuda") for p, n in zip(prefixes, lengths)]
    )
    cu = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.int32), lens.cumsum(0).int())
    )
    tables = {
        tag: torch.tensor([[2], [1]][: len(lengths)], device="cuda", dtype=torch.int32)
        for tag in cache.layouts
    }
    attn_inputs = SimpleNamespace(
        input_lengths=lens,
        prefix_lengths=pref,
        is_prefill=True,
        cache_store_inputs=None,
    )
    with patch.dict(os.environ, {"DSV4_PREFILL_FAST_PATH": "1" if fast else "0"}):
        return forward_layers(model, cache, ids, positions, cu, tables, attn_inputs)


class V41TransformerSM120Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41121)
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        env = patch.dict(os.environ, {"DSV4_MOE_STRATEGY": "auto", "MOEDBG": "0"})
        env.start()
        self.addCleanup(env.stop)

    def assert_attention_scratch_released(self, model):
        shared = model.layers[0].attn._shared_attention
        self.assertEqual(shared["global"], {})
        self.assertEqual(shared["topk"], {})
        self.assertIsNone(shared["candidates"])
        self.assertEqual(set(shared), {"layers", "global", "topk", "candidates"})
        for layer in model.layers:
            self.assertIsNone(layer.attn._prefill_meta_shared)

    @torch.inference_mode()
    def test_production_prefill_loop_fast_normal_and_continuation(self):
        args, weights = fixture()
        model, cache = build(args, weights)
        self.assertIsInstance(model.head_hc, DelayedHCHead)
        self.assertIsNone(model.layers[0].attn_hc._previous_ref)
        for i, layer in enumerate(model.layers):
            self.assertIsInstance(layer.attn, AttentionV41FP8)
            self.assertIsInstance(layer.attn_hc, DelayedHCUnit)
            self.assertIs(layer.ffn_hc._previous_ref(), layer.attn_hc)
            if i:
                self.assertIs(layer.attn_hc._previous_ref(), model.layers[i - 1].ffn_hc)
            self.assertEqual(layer.ffn._strategy.cfg.expert_recipe, V41_EXPERT_RECIPE)
            self.assertTrue(layer.ffn.gate.fp32_logits)
            self.assertIsInstance(layer.ffn._shared_executor, MXFP8SharedExpertExecutor)
        prior = None
        for lengths, prefixes in (([5, 3], [0, 0]), ([3, 4], [5, 3]), ([1], [0])):
            ids = torch.randint(0, 128, (sum(lengths),), device="cuda")
            before = {
                key: group.kv_cache_base.clone() for key, group in cache.groups.items()
            }
            actual = run(model, cache, ids, lengths, prefixes)
            after = {
                key: group.kv_cache_base.clone() for key, group in cache.groups.items()
            }
            for key, raw in before.items():
                cache.groups[key].kv_cache_base.copy_(raw)
            expected = run(model, cache, ids, lengths, prefixes, fast=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.shape, (sum(lengths), args.dim))
            self.assertTrue(
                bool(torch.isfinite(F.linear(actual, model.head_weight)).all())
            )
            for key, raw in after.items():
                torch.testing.assert_close(
                    cache.groups[key].kv_cache_base, raw, rtol=0, atol=0
                )
            for layer in model.layers:
                self.assertEqual(layer.ffn_hc.pre_mix_out.shape, (sum(lengths), 4))
                self.assertIsNot(layer.ffn_hc.pre_mix_out, prior)
            self.assert_attention_scratch_released(model)
            prior = model.layers[-1].ffn_hc.pre_mix_out

    @torch.inference_mode()
    def test_identity_readout_does_not_alias_single_row_residual(self):
        args, weights = fixture()
        model, _ = build(args, weights)
        residual = torch.randn(1, 4, args.dim, device="cuda", dtype=torch.bfloat16)
        original = residual.clone()
        readout, _, _ = model.layers[0].attn_hc.pre(residual)
        torch.testing.assert_close(readout, original[:, 0], rtol=0, atol=0)
        readout.zero_()
        torch.testing.assert_close(residual, original, rtol=0, atol=0)

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"), "NAS checkpoint required"
    )
    @torch.inference_mode()
    def test_nas_l0_full_width_block(self):
        args, weights = nas_fixture()
        model, cache = build(args, weights)
        self.assertEqual((args.dim, args.moe_inter_dim), (5120, 2304))
        for tokens in (1, 3):
            ids = torch.arange(tokens, device="cuda")
            before = {
                key: group.kv_cache_base.clone() for key, group in cache.groups.items()
            }
            actual = run(model, cache, ids, [tokens])
            for key, raw in before.items():
                cache.groups[key].kv_cache_base.copy_(raw)
            expected = run(model, cache, ids, [tokens], fast=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            self.assertGreater(actual.float().abs().max().item(), 0)
            self.assert_attention_scratch_released(model)

    @torch.inference_mode()
    def test_moe_recipe_and_fp32_combine_match_component_oracle(self):
        args, weights = fixture()
        raw = weights.weights[0]
        routed = {
            "w13_w": torch.cat((raw[W.v4_routed_w1_w], raw[W.v4_routed_w3_w]), 1).cpu(),
            "w13_s": torch.cat((raw[W.v4_routed_w1_s], raw[W.v4_routed_w3_s]), 1).cpu(),
            "w2_w": raw[W.v4_routed_w2_w].cpu(),
            "w2_s": raw[W.v4_routed_w2_s].cpu(),
        }
        model, _ = build(args, weights)
        ffn = model.layers[0].ffn
        x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(3, device="cuda")
        route, experts = ffn.gate(x, ids)
        reference = V41LocalMoE(
            V41RoutedExperts(routed, backend="eager"), max_tokens=256, max_topk=2
        )
        routed_y = reference(x.cpu(), route.cpu(), experts.cpu())
        shared_y = ffn.shared_experts(x).cpu()
        expected = (routed_y.float() + shared_y.float()).bfloat16()
        torch.testing.assert_close(ffn(x, ids).cpu(), expected, rtol=0.025, atol=0.015)
        executor = ffn._shared_executor
        for factor in (0.0, 1e-12, 1.0):
            scaled = x * factor
            executor.start(ffn.shared_experts, scaled)
            torch.testing.assert_close(
                executor.finish(), ffn.shared_experts(scaled), rtol=0, atol=0
            )

    @torch.inference_mode()
    def test_engram_injection_precedes_delayed_mhc(self):
        args, weights = fixture()
        model, cache = build(args, weights)
        layer = model.layers[1]
        config = dict(
            engram_layer_ids=[1],
            engram_num_embeddings=[4000],
            engram_max_ngram_size=4,
            engram_n_heads=4,
            engram_head_dim=32,
            engram_compressed_vocab_size=128,
            engram_pad_token_id=2,
            engram_vocab_size=13,
        )
        layout = EngramLayout(config)
        table = torch.randn(4000, 32).to(torch.float8_e4m3fn)
        scale = torch.full((4000, 1), 121, dtype=torch.uint8)
        pinned = _PinnedSharedTable(
            table, scale, "v41-block-" + uuid.uuid4().hex, "cuda:0"
        )
        layer.engram = Engram(
            layout,
            0,
            HostEngramEmbedding(pinned.weight, pinned.scales, pinned=pinned),
            V41MXFP8Linear(*fp8(2560, 384)),
            torch.ones(4, 512, device="cuda", dtype=torch.bfloat16),
            torch.ones(4, 512, device="cuda", dtype=torch.bfloat16),
            1e-20,
        )
        ids = torch.tensor([7, 8, 9], device="cuda")
        with self.assertRaisesRegex(RuntimeError, "token history is missing"):
            run(model, cache, ids, [3])
        self.assert_attention_scratch_released(model)
        hashes = NgramHashState(layout, token_map=list(range(128)), device="cuda:0")
        layer.engram_hashes = hashes(
            torch.tensor(
                [[7, -1, -1, -1], [8, 7, -1, -1], [9, 8, 7, -1]], device="cuda"
            )
        )[:, 0].contiguous()
        layer.engram_token_mask = torch.tensor([True, False, True], device="cuda")
        observed = {}
        handles = [
            layer.register_forward_pre_hook(
                lambda module, inp: observed.update(before=inp[0].clone())
            ),
            layer.engram.register_forward_hook(
                lambda module, inp, out: observed.update(injected=out.clone())
            ),
        ]
        pre = layer.attn_hc._pre_impl

        def record_pre(hidden, *a, **kw):
            observed["mhc_input"] = hidden.clone()
            return pre(hidden, *a, **kw)

        try:
            with patch.object(layer.attn_hc, "_pre_impl", side_effect=record_pre):
                result = run(model, cache, ids, [3])
        finally:
            for handle in handles:
                handle.remove()
        self.assertIsNone(layer.prefill_fast_callable())
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(
            observed["injected"], observed["mhc_input"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            observed["before"][1], observed["injected"][1], rtol=0, atol=0
        )
        self.assertFalse(torch.equal(observed["before"][0], observed["injected"][0]))
        self.assertEqual(layer.engram.embed_tokens.weight.device.type, "cpu")

    def test_unintegrated_topologies_fail_before_weight_consumption(self):
        args, weights = fixture()
        for bad in (
            replace(args, pp_size=2),
            replace(args, pp_layer_ids=[3, 4, 5]),
            replace(args, commit_only=True),
            replace(args, fp8_kv_cache=False),
        ):
            with self.assertRaises(NotImplementedError):
                V4Transformer(bad, weights)
        # EP has a recipe-compatible adapter now, but these unsliced weights
        # must still be rejected before the local provider consumes them.
        with self.assertRaisesRegex(ValueError, "local expert shard"):
            V4Transformer(replace(args, ep_size=2), weights)
        self.assertIn(W.v4_routed_w1_w, weights.weights[0])


if __name__ == "__main__":
    unittest.main()
