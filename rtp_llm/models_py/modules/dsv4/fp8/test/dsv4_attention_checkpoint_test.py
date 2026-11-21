"""Real attention parameter construction/loading; no GPU executor simulation."""

import os
import tempfile
import types
import unittest

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.weight_mapper import get_all_weights


class Dsv4AttentionCheckpointTest(unittest.TestCase):
    def _layer(self, ratio=0, rank=0):
        return AttentionFP8(
            layer_id=0,
            dim=256,
            n_heads=2,
            q_lora_rank=128,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=128,
            o_groups=2,
            window_size=128,
            compress_ratio=ratio,
            compress_rope_theta=10000.0,
            rope_theta=10000.0,
            rope_factor=1.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=0,
            max_batch_size=1,
            max_seq_len=128,
            index_n_heads=2,
            index_head_dim=128,
            index_topk=2,
            tp_size=2,
            tp_rank=rank,
        )

    def _weights(self, ratio=0):
        weights = {}
        for name, shape in {
            "wq_a": (128, 256),
            "wq_b": (1024, 128),
            "wkv": (512, 256),
            "wo_a": (256, 512),
            "wo_b": (256, 256),
        }.items():
            weights[name + ".weight"] = (
                torch.arange(shape[0] * shape[1]).reshape(shape) % 16
            ).to(torch.float8_e4m3fn)
            sshape = (shape[0] // 128, shape[1] // 128)
            weights[name + ".scale"] = (
                (torch.arange(sshape[0] * sshape[1]).reshape(sshape) + 120)
                .to(torch.uint8)
                .view(torch.float8_e8m0fnu)
            )
        weights["q_norm.weight"] = torch.full((128,), 2, dtype=torch.bfloat16)
        weights["kv_norm.weight"] = torch.full((512,), 3, dtype=torch.bfloat16)
        weights["attn_sink"] = torch.tensor([1.000123, 2.000123])

        def compressor(prefix, head_dim):
            out_dim = head_dim * (2 if ratio == 4 else 1)
            weights[prefix + "ape"] = torch.full(
                (ratio, out_dim), 1.000123, dtype=torch.float32
            )
            weights[prefix + "wkv.weight"] = torch.full(
                (out_dim, 256), 2, dtype=torch.bfloat16
            )
            weights[prefix + "wgate.weight"] = torch.full(
                (out_dim, 256), 3, dtype=torch.bfloat16
            )
            weights[prefix + "norm.weight"] = torch.full(
                (head_dim,), 4, dtype=torch.bfloat16
            )

        if ratio:
            compressor("compressor.", 512)
        if ratio == 4:
            weights["indexer.wq_b.weight"] = torch.full(
                (256, 128), 2, dtype=torch.float8_e4m3fn
            )
            weights["indexer.wq_b.scale"] = torch.tensor(
                [[127], [128]], dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
            weights["indexer.weights_proj.weight"] = torch.full(
                (2, 256), 3, dtype=torch.bfloat16
            )
            compressor("indexer.compressor.", 128)
        return weights

    def _load(self, weights, ratio=0, rank=0):
        layer = self._layer(ratio, rank)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pytorch_model.bin")
            torch.save(weights, path)
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="deepseek_v4"),
                NewLoaderConfig(device="cpu", tp_size=2, tp_rank=rank),
                model_path=folder,
            )
            resolver = loader._checkpoint_read_spec_resolver(layer, None, None)
            layer.load_weights(get_all_weights([path], read_spec_resolver=resolver))
        NewModelLoader._validate_loaded_weights(layer)
        return layer

    def test_real_checkpoint_swa_csa_hca_tp2_parameter_owners(self):
        for ratio in (0, 4, 128):
            weights = self._weights(ratio)
            for rank in (0, 1):
                with self.subTest(ratio=ratio, rank=rank):
                    layer = self._load(weights, ratio, rank)
                    for name, param in layer.named_parameters():
                        ckpt_name = {
                            "wo_a_w": "wo_a.weight",
                            "wo_a_s": "wo_a.scale",
                            "q_norm": "q_norm.weight",
                            "kv_norm": "kv_norm.weight",
                            "indexer.weights_proj": "indexer.weights_proj.weight",
                        }.get(name, name)
                        expected = weights[ckpt_name]
                        if (
                            ckpt_name.startswith(("wq_b.", "wo_a."))
                            or ckpt_name == "attn_sink"
                        ):
                            expected = expected.chunk(2, dim=0)[rank]
                        elif ckpt_name.startswith("wo_b."):
                            expected = expected.chunk(2, dim=1)[rank]
                        torch.testing.assert_close(
                            param.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
                    self.assertEqual(layer.attn_sink.dtype, torch.float32)
                    self.assertEqual(layer.wo_a_s.dtype, torch.float8_e8m0fnu)
                    self.assertIsNone(layer._wo_a_stk_w)
                    self.assertEqual(layer._can_reuse_qkv_input_quant(), True)

    def test_grouped_scale_missing_fails_completeness(self):
        weights = self._weights()
        del weights["wo_a.scale"]
        with self.assertRaisesRegex(RuntimeError, "wo_a_s"):
            self._load(weights)

    def test_grouped_scale_wrong_dtype_is_not_reinterpreted(self):
        weights = self._weights()
        weights["wo_a.scale"] = torch.ones(2, 4)
        with self.assertRaisesRegex(TypeError, "Dtype mismatch"):
            self._load(weights)

    def test_wrong_rank_sink_readspec_fails(self):
        source = self._layer(rank=0)
        dest = self._layer(rank=1)
        value = source.checkpoint_read_spec("attn_sink", (2,)).materialize(
            torch.tensor([1.0, 2.0])
        )
        with self.assertRaisesRegex(ValueError, "different shard"):
            dest.load_weights({"attn_sink": value})

    def test_cpu_is_not_reported_as_valid_gpu_runtime(self):
        layer = self._load(self._weights())
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            layer.process_weights_after_loading()

    def test_parent_load_plan_routes_original_checkpoint_names(self):
        parent = RtpModule()
        parent.attn = self._layer(ratio=4, rank=1)
        weights = {"attn." + name: tensor for name, tensor in self._weights(4).items()}
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "pytorch_model.bin")
            torch.save(weights, path)
            loader = NewModelLoader(
                types.SimpleNamespace(model_type="deepseek_v4"),
                NewLoaderConfig(device="cpu", tp_size=2, tp_rank=1),
                model_path=folder,
            )
            plan = {}
            resolver = loader._checkpoint_read_spec_resolver(parent, None, None, plan)
            parent._rtp_checkpoint_load_plan = plan
            parent.load_weights(get_all_weights([path], read_spec_resolver=resolver))
        NewModelLoader._validate_loaded_weights(parent)
        torch.testing.assert_close(parent.attn.attn_sink, weights["attn.attn_sink"][1:])
        torch.testing.assert_close(
            parent.attn.indexer.weights_proj,
            weights["attn.indexer.weights_proj.weight"],
        )


if __name__ == "__main__":
    unittest.main()
