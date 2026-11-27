"""Standalone checkpoint and packed-token output regression; no mocked native modules."""

import os
import tempfile
import unittest
from itertools import product
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from rtp_llm.models_py.checkpoint_reader_test_support import checkpoint_readers
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.jina_bert.encoder import JinaEncoder
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
