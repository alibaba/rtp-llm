"""Pure checkpoint-owned V4.1 attention tree; not GPU attention execution."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v41.attention import (
    OwnedV41Attention,
    OwnedV41CommitAttention,
)
from rtp_llm.models_py.weight_mapper import get_all_weights


def config(rank=0):
    return SimpleNamespace(
        dim=64,
        n_heads=2,
        o_groups=2,
        q_lora_rank=32,
        o_lora_rank=32,
        head_dim=64,
        rope_head_dim=32,
        window_size=8,
        norm_eps=1e-6,
        compress_ratios=[2, 1],
        tp_size=2,
        tp_rank=rank,
        index_topk=2,
        index_n_heads=2,
        index_head_dim=32,
        v41_config={"kv_source_layer_ids": [0], "index_source_layer_ids": [0]},
        compress_rope_theta=160000.0,
        rope_theta=10000.0,
        rope_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=0,
        max_seq_len=16,
    )


def weights(source=True):
    values = {}
    for name, output, input_ in (
        ("wq_a", 32, 64),
        ("wq_b", 128, 32),
        ("wkv", 64, 64),
        ("wo_a", 64, 64),
        ("wo_b", 64, 64),
    ):
        values[name + ".weight"] = (
            torch.arange(output * input_).reshape(output, input_) % 7
        ).to(torch.float8_e4m3fn)
        values[name + ".scale"] = torch.full(
            (output // 32, input_ // 32), 125, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    values["q_norm.weight"] = torch.ones(32, dtype=torch.bfloat16)
    values["kv_norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    values["attn_sink"] = torch.tensor([2.0, 3.0])
    if source:
        for name, shape in (
            ("compressor.wkv.weight", (64, 64)),
            ("compressor.wgate.weight", (64, 64)),
            ("compressor.norm.weight", (64,)),
            ("indexer.wk.weight", (32, 64)),
            ("indexer.k_norm.weight", (32,)),
            ("indexer.weights_proj.weight", (2, 64)),
        ):
            values[name] = torch.ones(shape, dtype=torch.bfloat16)
        values["indexer.wq_b.weight"] = torch.ones((64, 32), dtype=torch.float8_e4m3fn)
        values["indexer.wq_b.scale"] = torch.full((2, 1), 125, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
    return values


def load(owner, values):
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.safetensors")
        save_file(values, path)
        loader = NewModelLoader(
            SimpleNamespace(model_type="deepseek_v41"),
            NewLoaderConfig(device="cpu", tp_size=owner.tp_size, tp_rank=owner.tp_rank),
            model_path=folder,
        )
        plan = {}
        resolver = loader._checkpoint_read_spec_resolver(owner, None, None, plan)
        owner._rtp_checkpoint_load_plan = plan
        owner.load_weights(get_all_weights([path], read_spec_resolver=resolver))
    NewModelLoader._validate_loaded_weights(owner)
    return owner


class AttentionCheckpointTest(unittest.TestCase):
    def test_consumer_selects_nearest_source_regardless_of_list_order(self):
        cfg = config()
        cfg.compress_ratios = [1] * 5
        cfg.v41_config = {
            "kv_source_layer_ids": [2, 0, 0],
            "index_source_layer_ids": [1, 0, 0],
        }
        owner = OwnedV41Attention(cfg, 4, {})
        self.assertEqual(
            (owner.kv_source_layer_id, owner.index_source_layer_id), (2, 1)
        )
        self.assertEqual(list(owner.sources.parameters()), [])

    def test_qkv_runtime_reuses_loaded_parameters_and_raw_scale_views(self):
        owner = load(OwnedV41Attention(config(), 0, {}), weights())
        initial = dict(owner.named_parameters())
        q, kv = owner.wq_a, owner.wkv
        original = [p.view(torch.uint8).clone() for p in (q.weight, kv.weight)]

        # CPU storage-only stand-in for the native SM100 GEMM. The source
        # values and direct-dispatch parameter tree come from the real loader.
        def merge(qw, qs, kw, ks):
            weight = torch.cat([x.view(torch.uint8) for x in (qw, kw)]).view(qw.dtype)
            scale = torch.cat([x.view(torch.uint8) for x in (qs, ks)]).view(qs.dtype)
            return (
                torch.nn.Identity(),
                weight.split([qw.size(0), kw.size(0)]),
                scale.split([qs.size(0), ks.size(0)]),
            )

        owner._prepare_qkv(merge)
        self.assertIsNotNone(owner.wq_a_wkv)
        self.assertNotIn("wq_a_wkv", dict(owner.named_modules()))
        for name, p in owner.named_parameters():
            self.assertIs(initial[name], p)
        for a, b in ((q.weight, kv.weight), (q.scale, kv.scale)):
            self.assertEqual(
                a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
            )
            self.assertEqual(b.storage_offset(), a.numel())
        for idx, p in enumerate((q.weight, kv.weight)):
            self.assertTrue(torch.equal(p.view(torch.uint8), original[idx]))
        owner.to("cpu")
        self.assertIsNone(owner.wq_a_wkv)

    def test_grouped_wo_a_keeps_sources_without_a_standalone_executor(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                owner = load(OwnedV41Attention(config(rank), 0, {}), weights())
                leaf = owner.wo_a
                with patch.object(leaf, "validate_runtime_device") as preflight:
                    NewModelLoader._run_post_load_hooks(leaf)
                preflight.assert_called_once_with(leaf.weight.device)
                self.assertIsNone(leaf._executor)
                self.assertIs(owner.wo_a_w, leaf.weight)
                self.assertIs(owner.wo_a_s, leaf.scale)
                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    leaf.process_weights_after_loading()

    def test_source_consumer_shards_are_owned_and_runtime_views_alias(self):
        for rank in (0, 1):
            shared = {}
            source = load(OwnedV41Attention(config(rank), 0, shared), weights())
            consumer = load(OwnedV41Attention(config(rank), 1, shared), weights(False))
            self.assertIs(shared["layers"][0], source)
            self.assertIs(shared["layers"][1], consumer)
            self.assertEqual(
                (consumer.kv_source_layer_id, consumer.index_source_layer_id), (0, 0)
            )
            self.assertEqual(list(consumer.sources.parameters()), [])
            self.assertIs(source.global_wkv, source.sources.compressor.wkv.weight)
            self.assertIs(source.index_wq, source.sources.indexer.wq_b)
            self.assertEqual(source.attn_sink.item(), 2.0 + rank)
            self.assertEqual(source.wq_b.weight.shape, (64, 32))
            self.assertEqual(source.wo_a.scale.shape, (1, 2))
            self.assertEqual(source.wo_b.scale.shape, (2, 1))
            self.assertEqual(source._pool_spec["swa_kv"], (torch.uint8, 528))
            self.assertEqual(source._pool_spec["indexer_kv"], (torch.uint8, 68))

    def test_every_checkpoint_tensor_required(self):
        original = weights()
        for missing in original:
            values = dict(original)
            del values[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(
                RuntimeError, "missing required"
            ):
                load(OwnedV41Attention(config(), 0, {}), values)

    def test_commit_only_native32_scale_and_v41_cache(self):
        args = config()
        args.compress_ratios = [0, 0]
        selected = {
            name: value
            for name, value in weights(False).items()
            if name in ("wkv.weight", "wkv.scale", "kv_norm.weight")
        }
        owner = load(OwnedV41CommitAttention(args, 0, {}), selected)
        self.assertEqual(len(list(owner.parameters())), 3)
        self.assertEqual(owner.wkv.scale.shape, (2, 2))
        self.assertEqual(owner._pool_spec["swa_kv"], (torch.uint8, 528))

    def test_runtime_source_properties_follow_parameter_rebinding(self):
        owner = load(OwnedV41Attention(config(), 0, {}), weights())
        replacement = torch.nn.Parameter(
            torch.full((64, 64), 4.0, dtype=torch.bfloat16), requires_grad=False
        )
        owner.sources.compressor.wkv.weight = replacement
        self.assertIs(owner.global_wkv, replacement)
        self.assertEqual(
            sum(
                name.endswith("compressor.wkv.weight")
                for name, _ in owner.named_parameters()
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
