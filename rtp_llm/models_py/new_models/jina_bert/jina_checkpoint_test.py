"""Standalone checkpoint and packed-token output regression; no mocked native modules."""

import os
import tempfile
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.jina_bert.encoder import (
    JinaEncoder,
    JinaEncoderLayer,
    _packed_attention,
)
from rtp_llm.models_py.registry import register_model


@register_model("_jina_encoder_cpu_test")
class _CpuEncoder(JinaEncoder):
    pass


def config(heads=3, inter=17):
    return SimpleNamespace(
        model_type="_jina_encoder_cpu_test",
        num_layers=2,
        hidden_size=12,
        attn_config=SimpleNamespace(head_num=heads),
        inter_size=inter,
        layernorm_eps=1e-5,
        type_vocab_size=2,
        vocab_size=29,
    )


def checkpoint(inter=17):
    rng = torch.Generator().manual_seed(71)
    rand = lambda *shape: torch.randn(*shape, generator=rng) * 0.2
    out = {
        "embeddings.word_embeddings.weight": rand(29, 12),
        "embeddings.token_type_embeddings.weight": rand(2, 12),
        "embeddings.LayerNorm.weight": 1 + rand(12),
        "embeddings.LayerNorm.bias": rand(12),
    }
    for layer in range(2):
        prefix = f"encoder.layer.{layer}."
        for k in (
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
        ):
            out[prefix + k + ".weight"], out[prefix + k + ".bias"] = rand(12, 12), rand(
                12
            )
        for k in (
            "attention.self.layer_norm_q",
            "attention.self.layer_norm_k",
            "attention.output.LayerNorm",
            "layer_norm_1",
            "layer_norm_2",
        ):
            out[prefix + k + ".weight"], out[prefix + k + ".bias"] = 1 + rand(12), rand(
                12
            )
        out[prefix + "mlp.up_gated_layer.weight"] = rand(2 * inter, 12)
        out[prefix + "mlp.down_layer.weight"], out[prefix + "mlp.down_layer.bias"] = (
            rand(12, inter),
            rand(12),
        )
    return out


def exact_forward(w, ids, types, slopes=(1 / 16, 1 / 256, 1 / 4)):
    def norm(x, k):
        return F.layer_norm(x, (12,), w[k + ".weight"], w[k + ".bias"], 1e-5)

    def linear(x, k):
        return F.linear(x, w[k + ".weight"], w.get(k + ".bias"))

    x = norm(
        w["embeddings.word_embeddings.weight"][ids]
        + w["embeddings.token_type_embeddings.weight"][types],
        "embeddings.LayerNorm",
    )
    for i in range(2):
        p = f"encoder.layer.{i}."
        q = norm(
            linear(x, p + "attention.self.query"), p + "attention.self.layer_norm_q"
        )
        k = norm(linear(x, p + "attention.self.key"), p + "attention.self.layer_norm_k")
        v = linear(x, p + "attention.self.value")
        heads = []
        pos = torch.arange(len(ids))
        # Three-head ALiBi from the reference recurrence, not the model's helper.
        width = 12 // len(slopes)
        for h, slope in enumerate(slopes):
            sl = slice(width * h, width * (h + 1))
            scores = (
                q[:, sl] @ k[:, sl].T / width**0.5 - slope * (pos[:, None] - pos).abs()
            )
            heads.append(scores.softmax(-1) @ v[:, sl])
        a = norm(
            x + linear(torch.cat(heads, -1), p + "attention.output.dense"),
            p + "attention.output.LayerNorm",
        )
        x = norm(x + a, p + "layer_norm_1")
        up, gate = linear(x, p + "mlp.up_gated_layer").chunk(2, -1)
        x = norm(
            x + linear(up * F.gelu(gate), p + "mlp.down_layer"), p + "layer_norm_2"
        )
    return x


def tp_worker(rank, directory):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method="file://" + os.path.join(directory, "pg"),
    )
    try:
        for method, (attn, ffn) in product(
            checkpoint_readers("scratch", "fastsafetensors"), ((2, 2), (1, 2), (2, 1))
        ):
            load = NewLoaderConfig(
                device="cpu",
                tp_size=2,
                tp_rank=rank,
                attn_tp_size=attn,
                attn_tp_rank=rank if attn > 1 else 0,
                ffn_tp_size=ffn,
                ffn_tp_rank=rank if ffn > 1 else 0,
                compute_dtype=torch.float32,
                load_method=method,
            )
            loader = NewModelLoader(config(2, 18), load, model_path=directory)
            model = loader.load()
            assert loader._selected_load_method.value == method
            local = model.encoder.layer[0]
            assert local.attention.self.query.weight.shape == (12 // attn, 12)
            assert local.mlp.up_gated_layer.weight.shape == (36 // ffn, 12)
            ids = torch.tensor([4, 9, 2])
            types = torch.tensor([1, 0, 1])
            result = model.encode(ids[None], types[None], pg=dist.group.WORLD)
            expected = exact_forward(checkpoint(18), ids, types, (1 / 16, 1 / 256))
            torch.testing.assert_close(result[0], expected)
    finally:
        dist.destroy_process_group()


class JinaCheckpointTest(unittest.TestCase):
    def test_alibi_matches_oldloader_tp_partition(self):
        from rtp_llm.utils.model_weight import slopes, split_slopes_tp

        heads, tp = 12, 2
        for rank in range(tp):
            layer = JinaEncoderLayer(config(heads), torch.float32, tp, rank)
            expected = split_slopes_tp(slopes([], heads), heads, tp, rank)
            torch.testing.assert_close(torch.tensor(layer.slopes), expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_fused_packed_alibi_replays_live_boundaries(self):
        from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

        if not can_use_flash_attn():
            self.skipTest("requires packed FlashAttention-2")
        import flash_attn

        total, heads, width = 8192, 3, 64
        ids = torch.zeros(total, dtype=torch.int32, device="cuda")
        q, k, v = torch.randn(
            3, 1, heads, total, width, device="cuda", dtype=torch.float16
        )
        slopes = torch.tensor([1 / 16, 1 / 256, 1 / 4], device="cuda")
        ends = torch.tensor([0, 512, 512, 7680, 7936], device="cuda", dtype=torch.int32)

        def forward():
            attention = _packed_attention(ids, ends, None, True, q.dtype, width)
            return attention(q, k, v, slopes)

        with patch.object(
            flash_attn, "flash_attn_varlen_func", wraps=flash_attn.flash_attn_varlen_func
        ) as calls:
            forward()
            self.assertEqual(calls.call_count, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward()
        for offsets in ([0, 512, 512, 7680, 7936], [0, total, total, total, total]):
            ends.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
            graph.replay()
            for a, b in zip(offsets, offsets[1:] + [total]):
                if a == b:
                    continue
                pos = torch.arange(b - a, device="cuda")
                # Check every short request/padding row and sample the long
                # request at both ends and in the middle, without a dense 8K² mask.
                rows = pos if b - a <= 512 else pos[[0, 1, (b - a) // 2, -2, -1]]
                bias = -slopes[:, None, None] * (rows[:, None] - pos).abs()
                expected = F.scaled_dot_product_attention(
                    q[:, :, a + rows].float(),
                    k[:, :, a:b].float(),
                    v[:, :, a:b].float(),
                    attn_mask=bias,
                )
                torch.testing.assert_close(
                    actual[:, :, a + rows].float(), expected, atol=1e-3, rtol=1e-2
                )

    def test_long_packed_attention_respects_boundaries_and_tile_budget(self):
        # Includes an empty graph slot and physical padding after the last end.
        lengths = [512, 768, 256, 512]
        ids = torch.arange(sum(lengths), dtype=torch.int32) % 29
        types = torch.arange(ids.numel(), dtype=torch.int32) % 2
        ends = torch.tensor([0, 512, 512, 1280, 1536], dtype=torch.int32)
        data = checkpoint()
        model = self._load(data, "scratch")
        with patch.object(
            F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
        ) as calls:
            actual = model.encode_cumulative(ids, types, ends)
        expected = torch.cat(
            [
                exact_forward(data, seq, kind)
                for seq, kind in zip(ids.split(lengths), types.split(lengths))
            ]
        )
        torch.testing.assert_close(actual, expected)
        for call in calls.call_args_list:
            self.assertLessEqual(call.args[1].size(2), max(lengths))
            self.assertLessEqual(call.kwargs["attn_mask"].numel(), 1 << 20)
        # Five real SDPA tiles/layer, versus 13 dense packed tiles/layer.
        self.assertEqual(calls.call_count, 5 * len(model.encoder.layer))

    def test_graph_attention_uses_live_ends_not_host_fixture(self):
        data = checkpoint()
        model = self._load(data, "scratch")
        ids = torch.tensor([1, 4, 8, 2, 6, 3, 7])
        types = torch.zeros_like(ids)
        stale = torch.tensor([0, 7, 7], dtype=torch.int32)
        for boundary in (1, 4):
            ends = torch.tensor([0, boundary, 6], dtype=torch.int32)
            result = model.encode_cumulative(
                ids, types, ends, host_ends=stale, is_graph=True
            )
            expected = torch.cat(
                [
                    exact_forward(data, ids[a:b], types[a:b])
                    for a, b in ((0, boundary), (boundary, 6), (6, 7))
                ]
            )
            torch.testing.assert_close(result, expected)

    def test_split_prefill_cannot_silently_run_local_bidirectional_attention(self):
        parallel = SimpleNamespace(prefill_cp_config=SimpleNamespace(is_enabled=lambda: True))
        load = NewLoaderConfig(
            device="cpu", compute_dtype=torch.float32, tp_size=2,
            parallelism_config=parallel,
        )
        with self.assertRaisesRegex(ValueError, "complete sequence"):
            _CpuEncoder(config(heads=2, inter=18), load)

    def test_tp2_gloo(self):
        with tempfile.TemporaryDirectory() as d:
            save_file(checkpoint(18), os.path.join(d, "model.safetensors"))
            torch.multiprocessing.spawn(tp_worker, args=(d,), nprocs=2, join=True)

    def _load(self, data, method, mappings=()):
        with tempfile.TemporaryDirectory() as d:
            save_file(data, os.path.join(d, "model.safetensors"))
            loader = NewModelLoader(
                config(),
                NewLoaderConfig(
                    device="cpu",
                    compute_dtype=torch.float32,
                    load_method=method,
                    custom_weight_mappings=mappings,
                ),
                model_path=d,
            )
            model = loader.load()
            self.assertEqual(loader._selected_load_method.value, method)
        return model

    def test_load_forward(self):
        data = checkpoint()
        ids = torch.tensor([2, 8, 24, 1, 5, 6, 17])
        types = torch.tensor([1, 0, 1, 0, 0, 0, 1])
        lengths = [3, 1, 3]
        expected = torch.cat(
            [
                exact_forward(data, seq, kind)
                for seq, kind in zip(ids.split(lengths), types.split(lengths))
            ]
        )
        for method, alias in checkpoint_readers(("scratch", False), ("fastsafetensors", True)):
            with self.subTest(method=method):
                payload = {}
                for k, v in data.items():
                    if alias and any(
                        n in k
                        for n in (
                            "embeddings.LayerNorm",
                            "attention.output.LayerNorm",
                            "layer_norm_2",
                        )
                    ):
                        k = k.replace(".weight", ".gamma").replace(".bias", ".beta")
                    payload[("bert." if alias else "") + k] = v
                payload["cls.predictions.bias"] = torch.ones(29)
                model = self._load(payload, method)
                torch.testing.assert_close(
                    model.encode_packed(ids, types, lengths), expected
                )
                seq_ids = torch.arange(len(lengths)).repeat_interleave(
                    torch.tensor(lengths)
                )
                packed = model.encode(ids[None], types[None], sequence_ids=seq_ids)
                torch.testing.assert_close(packed.squeeze(0), expected)
                for key, param in model.named_parameters():
                    torch.testing.assert_close(param, data[key], rtol=0, atol=0)

    def test_missing_layer_norm(self):
        data = checkpoint()
        del data["encoder.layer.1.attention.self.layer_norm_k.bias"]
        with self.assertRaisesRegex(RuntimeError, "layer_norm_k.bias"):
            self._load(data, "scratch")

    def test_owned_downstream(self):
        data = checkpoint()
        data["classifier.weight"] = torch.arange(24, dtype=torch.float32).reshape(2, 12)
        mapping = (("__custom__.classifier.weight", "classifier.weight"),)
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            model = self._load(data, method, mapping)
            torch.testing.assert_close(
                model.custom_weight_view()["__custom__.classifier.weight"],
                data["classifier.weight"],
            )
        del data["classifier.weight"]
        with self.assertRaisesRegex(RuntimeError, "missing Jina downstream"):
            self._load(data, "scratch", mapping)

    def test_downstream_declared_dtype(self):
        data = checkpoint()
        source = "classifier.weight"
        data[source] = torch.rand(2, 12)
        for method in checkpoint_readers("scratch", "fastsafetensors"):
            with self.subTest(method=method):
                model = self._load(
                    data, method, (("__custom__.head", source, torch.float16),)
                )
                torch.testing.assert_close(
                    model.custom_weight_view()["__custom__.head"], data[source].half()
                )


if __name__ == "__main__":
    unittest.main()
