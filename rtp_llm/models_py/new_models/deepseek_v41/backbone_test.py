"""Actual safetensors -> owned V4.1 tree, stopping before CUDA preparation."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41.backbone import OwnedV41Transformer
from rtp_llm.models_py.weight_mapper import get_all_weights


def args(rank=0, commit=False):
    return SimpleNamespace(
        n_layers=3,
        n_mtp_layers=0,
        dim=64,
        vocab_size=8,
        n_heads=2,
        q_lora_rank=32,
        head_dim=64,
        rope_head_dim=32,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=[0, 0, 0] if commit else [0, 2, 1],
        norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=0,
        max_seq_len=16,
        max_batch_size=2,
        max_tokens_per_rank=16,
        index_topk=2,
        index_n_heads=2,
        index_head_dim=32,
        tp_size=2,
        tp_rank=rank,
        ep_size=2,
        ep_rank=rank,
        dp_size=1,
        dp_rank=0,
        world_size=2,
        world_rank=rank,
        is_decode_role=False,
        n_shared_experts=1,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=64,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        swiglu_limit=10.0,
        moe_strategy="auto",
        n_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        fp8_kv_cache=True,
        commit_only=commit,
        lm_head_tp_size=2,
        lm_head_tp_rank=rank,
        embedding_tp_size=1,
        embedding_tp_rank=0,
        v41_config={
            "kv_source_layer_ids": [] if commit else [1],
            "index_source_layer_ids": [] if commit else [1],
            "engram_layer_ids": [],
        },
    )


def tensor(shape, dtype, seed=1):
    n = 1
    for size in shape:
        n *= size
    data = torch.arange(n).reshape(shape)
    if dtype == torch.float8_e8m0fnu:
        return (data % 3 + 124).to(torch.uint8).view(dtype)
    return (data % 5 + seed).to(dtype)


def checkpoint(commit=False, layer_prefix="layers"):
    a = args(commit=commit)
    values = {}

    def add(name, shape, dtype=torch.bfloat16, seed=1):
        values[name] = tensor(shape, dtype, seed)

    def fp8(name, output, input_):
        add(name + ".weight", (output, input_), torch.float8_e4m3fn)
        add(name + ".scale", (output // 32, input_ // 32), torch.float8_e8m0fnu)

    for i in range(3):
        p = f"{layer_prefix}.{i}."
        fp8(p + "attn.wkv", 64, 64)
        add(p + "attn.kv_norm.weight", (64,))
        if commit:
            continue
        for name, output, input_ in (
            ("wq_a", 32, 64),
            ("wq_b", 128, 32),
            ("wo_a", 64, 64),
            ("wo_b", 64, 64),
        ):
            fp8(p + "attn." + name, output, input_)
        add(p + "attn.q_norm.weight", (32,))
        add(p + "attn.attn_sink", (2,), torch.float32)
        if i == 1:
            for name, shape in (
                ("compressor.wkv.weight", (64, 64)),
                ("compressor.wgate.weight", (64, 64)),
                ("compressor.norm.weight", (64,)),
                ("indexer.wk.weight", (32, 64)),
                ("indexer.k_norm.weight", (32,)),
                ("indexer.weights_proj.weight", (2, 64)),
            ):
                add(p + "attn." + name, shape)
            fp8(p + "attn.indexer.wq_b", 64, 32)
        for kind in ("attn", "ffn"):
            add(p + kind + "_norm.weight", (64,))
            add(p + "hc_" + kind + "_fn", (8, 128), torch.float32)
            add(p + "hc_" + kind + "_base", (8,), torch.float32)
            add(p + "hc_" + kind + "_scale", (3,), torch.float32)
        for expert in range(4):
            for projection in ("w1", "w2", "w3"):
                key = p + f"ffn.experts.{expert}.{projection}"
                add(key + ".weight", (64, 32), torch.int8, expert + 1)
                add(key + ".scale", (64, 2), torch.float8_e8m0fnu)
        for projection in ("w1", "w2", "w3"):
            fp8(p + "ffn.shared_experts." + projection, 64, 64)
        add(p + "ffn.gate.weight", (4, 64))
        add(p + "ffn.gate.bias", (4,), torch.float32)
    if not commit:
        add("embed.weight", (8, 64))
        add("head.weight", (8, 64))
        add("norm.weight", (64,))
    return values


class BackboneCheckpointTest(unittest.TestCase):
    def test_native_hc_checkpoint_owners_preserve_fused_seam(self):
        from safetensors import safe_open

        from rtp_llm.models_py.module_base import RtpModule
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
        from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit
        from rtp_llm.models_py.modules.dsv41.hc.v41_jit_warmup import (
            _collect_v41_hc_configs,
        )
        from rtp_llm.models_py.new_models.deepseek_v41.backbone import (
            _checkpoint_cuda_norm,
            _checkpoint_native_hc,
        )

        config = args()
        config.dim, config.hc_mult = 5120, 4
        block = RtpModule()
        block.attn_hc = _checkpoint_native_hc(config, 0, "attn")
        block.ffn_hc = _checkpoint_native_hc(config, 0, "ffn")
        block.ffn_hc.set_previous(block.attn_hc)
        block.ffn_norm = _checkpoint_cuda_norm(
            config.dim, config.norm_eps, torch.bfloat16
        )
        self.assertIs(type(block.ffn_hc), DelayedHCUnit)
        self.assertIs(type(block.ffn_norm), RMSNorm)
        before = {k: id(p) for k, p in block.named_parameters()}
        weights = {
            k: torch.full_like(p, i + 1)
            for i, (k, p) in enumerate(block.named_parameters())
        }
        self.assertEqual(len(weights), 7)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            save_file(weights, path)
            with safe_open(path, framework="pt", device="cpu") as checkpoint_file:
                block.load_weights(
                    (k, checkpoint_file.get_tensor(k)) for k in checkpoint_file.keys()
                )
        for k, p in block.named_parameters():
            self.assertEqual(id(p), before[k])
            self.assertTrue(torch.equal(p, weights[k]))
        units, seams = _collect_v41_hc_configs(block)
        self.assertEqual((len(units), len(seams)), (1, 1))
        self.assertEqual(
            next(iter(seams.values())), (block.attn_hc, block.ffn_hc, block.ffn_norm)
        )

    def load(self, values, rank=0, commit=False, vision=False, native_fst=False):
        config = args(rank, commit)
        config.embedding_tp_size, config.embedding_tp_rank = 2, rank
        config.vision_router_bias_layers = {"layers": (1,)} if vision else {}
        model = OwnedV41Transformer(config, layer_prefix="mtp" if commit else "layers")
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            save_file(values, path)
            loader = NewModelLoader(
                SimpleNamespace(model_type="deepseek_v41"),
                NewLoaderConfig(
                    device="cpu", tp_size=2, tp_rank=rank, ep_size=2, ep_rank=rank
                ),
                model_path=root,
            )
            name_filter = model.checkpoint_weight_name_filter()
            plan = {}
            resolver = loader._checkpoint_read_spec_resolver(
                model, name_filter, None, plan
            )
            model._rtp_checkpoint_load_plan = plan
            weights = (
                loader._fastsafetensors_weights_iterator(
                    [path], "cpu", name_filter, None, resolver
                )
                if native_fst
                else get_all_weights(
                    [path], name_filter=name_filter, read_spec_resolver=resolver
                )
            )
            model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)
        return model

    @unittest.skipUnless(
        os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1", "requires native FST"
    )
    def test_native_fst_full_tree_matches_scratch_every_owned_parameter(self):
        import fastsafetensors.cpp as cpp

        self.assertTrue(cpp.__file__.endswith(".so"))
        for commit in (False, True):
            values = checkpoint(commit, "mtp" if commit else "layers")
            values["layers.1.engram.embed.weight"] = torch.ones(1)
            for rank in (0, 1):
                with self.subTest(commit=commit, rank=rank):
                    scratch = self.load(values, rank, commit)
                    fst = self.load(values, rank, commit, native_fst=True)
                    expected = dict(scratch.named_parameters())
                    actual = dict(fst.named_parameters())
                    self.assertEqual(set(actual), set(expected))
                    for name, parameter in actual.items():
                        self.assertEqual(parameter.dtype, expected[name].dtype, name)
                        self.assertTrue(
                            torch.equal(
                                parameter.contiguous().view(torch.uint8),
                                expected[name].contiguous().view(torch.uint8),
                            ),
                            name,
                        )

    def test_actual_checkpoint_full_tree_tp_ep_rank_local_owners(self):
        values = checkpoint()
        # Host Engram belongs to the independent CPU-source loader, not this
        # ordinary device stream. Invalid payload proves pre-read exclusion.
        values["layers.1.engram.embed.weight"] = torch.ones(1)
        for rank in (0, 1):
            model = self.load(values, rank)
            # OldFST allocates a multiple of eight rows per LM-head rank.
            # With eight real tokens, rank one is entirely padding.
            expected_head = values["head.weight"]
            if rank == 1:
                expected_head = torch.zeros_like(expected_head)
            torch.testing.assert_close(model.head.weight, expected_head)
            torch.testing.assert_close(
                model.embed.weight,
                values["embed.weight"][:, rank * 32 : (rank + 1) * 32],
            )
            self.assertEqual(model.layers[1].attn.wq_b.scale.shape, (2, 1))
            self.assertEqual(model.layers[2].attn.kv_source_layer_id, 1)
            self.assertEqual(list(model.layers[2].attn.sources.parameters()), [])
            self.assertIs(
                model.layers[1].attn.global_wkv,
                model.layers[1].attn.sources.compressor.wkv.weight,
            )
            self.assertIs(
                model.layers[0].ffn_hc._previous_ref(), model.layers[0].attn_hc
            )
            self.assertIs(
                model.layers[1].attn_hc._previous_ref(), model.layers[0].ffn_hc
            )
            self.assertIs(model.head_hc._last_ffn_ref(), model.layers[-1].ffn_hc)
            self.assertFalse(
                any("hc_head" in name for name, _ in model.named_parameters())
            )
            for i, layer in enumerate(model.layers):
                for local in range(2):
                    expected = values[
                        f"layers.{i}.ffn.experts.{rank * 2 + local}.w1.weight"
                    ]
                    torch.testing.assert_close(
                        layer.ffn.routed_w13[local, :64], expected
                    )
            self.assertEqual(
                model.layers[1].attn._pool_spec["swa_kv"], (torch.uint8, 528)
            )
            self.assertEqual(
                model.layers[1].attn._pool_spec["global_kv_2"], (torch.uint8, 288)
            )

    def test_required_attention_source_moe_hc_and_global_omissions_fail(self):
        for missing in (
            "layers.0.attn.wkv.scale",
            "layers.1.attn.compressor.wgate.weight",
            "layers.1.attn.indexer.wk.weight",
            "layers.1.attn.indexer.wq_b.scale",
            "layers.0.ffn.experts.0.w1.scale",
            "layers.2.hc_ffn_scale",
            "head.weight",
        ):
            values = checkpoint()
            del values[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(
                RuntimeError, "[Mm]issing"
            ):
                self.load(values)

    def test_commit_only_has_exact_native_wkv_norm_and_v41_cache_abi(self):
        values = checkpoint(commit=True, layer_prefix="mtp")
        values["mtp.0.attn.wq_a.weight"] = torch.ones(1)
        model = self.load(values, commit=True)
        self.assertEqual(len(list(model.parameters())), 9)
        self.assertIsNone(model.embed)
        for layer in model.layers:
            self.assertIsNone(layer.ffn)
            self.assertEqual(layer.attn.wkv.scale.shape, (2, 2))
            self.assertEqual(layer.attn._pool_spec["swa_kv"], (torch.uint8, 528))

    def test_optional_vision_bias_is_loaded_and_required_when_declared(self):
        values = checkpoint()
        values["layers.1.ffn.gate.bias"] = torch.tensor([100.0, 20.0, 0.0, 0.0])
        values["layers.1.ffn.gate.bias_vl"] = torch.tensor([0.0, 0.0, 100.0, 200.0])
        model = self.load(values, vision=True)
        owner = model.layers[1].ffn
        torch.testing.assert_close(
            owner.router_bias_vl, values["layers.1.ffn.gate.bias_vl"]
        )
        del values["layers.1.ffn.gate.bias_vl"]
        with self.assertRaisesRegex(RuntimeError, "gate.bias_vl"):
            self.load(values, vision=True)

    def test_vision_bias_reaches_main_gate_when_native_runtime_is_available(self):
        try:
            from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import (
                Gate,
            )
        except ModuleNotFoundError as error:
            if error.name == "libth_transformer_config":
                self.skipTest("matching main native config library is unavailable")
            raise
        values = checkpoint()
        values["layers.1.ffn.gate.bias"] = torch.tensor([100.0, 20.0, 0.0, 0.0])
        values["layers.1.ffn.gate.bias_vl"] = torch.tensor([0.0, 0.0, 100.0, 200.0])
        owner = self.load(values, vision=True).layers[1].ffn
        gate = Gate(
            layer_id=1,
            dim=64,
            n_routed_experts=4,
            n_activated_experts=2,
            vocab_size=8,
            weight_owner=owner,
        )
        self.assertIs(gate.bias_vl, owner.router_bias_vl)
        gate._has_visual_tokens = True
        _, experts = gate(
            torch.zeros(2, 64, dtype=torch.bfloat16), torch.tensor([0, 11])
        )
        torch.testing.assert_close(experts, torch.tensor([[0, 1], [3, 2]]))

    def test_checkpoint_metadata_selects_target_and_draft_vision_bias_separately(self):
        import json

        from rtp_llm.models_py.new_models.deepseek_v41.owner import vision_router_layers

        with tempfile.TemporaryDirectory() as folder:
            names = {
                "layers.1.ffn.gate.bias_vl": "s.safetensors",
                "mtp.2.ffn.gate.bias_vl": "d.safetensors",
                "layers.1.ffn.gate.bias": "s.safetensors",
            }
            with open(
                os.path.join(folder, "model.safetensors.index.json"), "w"
            ) as stream:
                json.dump({"weight_map": names}, stream)
            self.assertEqual(
                vision_router_layers(folder), {"layers": (1,), "mtp": (2,)}
            )


if __name__ == "__main__":
    unittest.main()
