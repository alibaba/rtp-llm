"""CPU checks for M3.1 DSpARK checkpoint and main model ownership."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.minimax_m31 import MiniMaxM31
from rtp_llm.models.minimax_m31_dspark import MiniMaxM31DSpark, MiniMaxM31DSparkWeight
from rtp_llm.models_py.modules.minimax_m31.kernels.fa4_import import (
    prefer_flash_attn_4_cute,
)
from rtp_llm.ops import DataType, HWKernelConfig, ParallelismConfig, SpeculativeType
from rtp_llm.utils.model_weight import W


def draft_config():
    return {
        "architectures": ["DSparkMiniMaxDraftModel"],
        "quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32]},
        "text_config": {
            "layer_types": ["sliding_attention"] * 5,
            "moe_layer_freq": [0] * 5,
            "use_gemma_norm": True,
            "hidden_size": 6144,
            "vocab_size": 200064,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "partial_rotary_factor": 0.5,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "dense_intermediate_size": 12288,
            "dspark_noise_token_id": 200058,
            "dspark_target_layer_ids": [3, 17, 31, 45, 59],
            "dspark_markov_rank": 256,
            "dspark_block_size": 7,
            "sliding_window": 4096,
        },
    }


class MiniMaxM31DSparkTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_packed_mxfp8_scale_dequantizes_independent_rows(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MiniMax-M3.1 MXFP8 requires Blackwell CUDA")
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

        x = torch.randn((7, 128), device="cuda", dtype=torch.bfloat16)
        quantized, packed = mxfp8_quant_act_packed(x)
        self.assertEqual(tuple(packed.shape), (7, 1))
        scale_bytes = (
            (packed.to(torch.int32)[:, :, None] >> (8 * torch.arange(4, device="cuda")))
            & 0xFF
        ).reshape(7, 4)
        scale = torch.exp2(scale_bytes.float() - 127).repeat_interleave(32, dim=1)
        torch.testing.assert_close(
            quantized.float() * scale, x.float(), rtol=0.15, atol=0.15
        )
        self.assertEqual(mxfp8_quant_act_packed(x[:0])[0].shape, (0, 128))

    def test_context_rope_calls_flashinfer_with_zero_query_heads(self):
        from rtp_llm.models_py.modules.minimax_m31.dspark.minimax_m31_dspark_layer import (
            MiniMaxM31DSparkLayer,
        )

        q = torch.empty((2, 0, 128), dtype=torch.bfloat16)
        k = torch.empty((2, 4, 128), dtype=torch.bfloat16)
        positions = torch.arange(2, dtype=torch.int32)
        with patch("flashinfer.apply_rope_pos_ids_inplace") as apply:
            MiniMaxM31DSparkLayer._rope(
                SimpleNamespace(rope_dim=64, rope_theta=10000), q, k, positions
            )
        apply.assert_called_once_with(
            q, k, positions, rotary_dim=64, interleave=False, rope_theta=10000
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_context_rope_accepts_zero_query_heads_on_pinned_flashinfer(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MiniMax-M3.1 DSpARK requires Blackwell CUDA")
        from rtp_llm.models_py.modules.minimax_m31.dspark.minimax_m31_dspark_layer import (
            MiniMaxM31DSparkLayer,
        )

        q = torch.empty((2, 0, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.ones((2, 4, 128), dtype=torch.bfloat16, device="cuda")
        positions = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        MiniMaxM31DSparkLayer._rope(
            SimpleNamespace(rope_dim=64, rope_theta=10000), q, k, positions
        )
        self.assertTrue(torch.equal(k[0], torch.ones_like(k[0])))
        self.assertTrue(torch.isfinite(k).all())

    def test_packed_mxfp8_rejects_incompatible_scale_geometry(self):
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed
        from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
            pack_flashinfer_mxfp8_scale_triton,
        )

        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            mxfp8_quant_act_packed(torch.empty(1, 96))
        with self.assertRaisesRegex(ValueError, "row-major uint8"):
            pack_flashinfer_mxfp8_scale_triton(torch.empty(1, 4), 1, 128)
        empty = pack_flashinfer_mxfp8_scale_triton(
            torch.empty((0, 4), dtype=torch.uint8), 0, 128
        )
        self.assertEqual(tuple(empty.shape), (0, 1))

    def test_dspark_rejects_wrong_speculative_type_before_loading(self):
        engine = SimpleNamespace(
            sp_config=SimpleNamespace(
                type=SpeculativeType.MTP,
                checkpoint_path="/nonexistent",
                model_type="minimax_m31_dspark",
            )
        )
        with self.assertRaisesRegex(ValueError, "SP_TYPE=dspark"):
            ModelFactory.create_propose_model_config(engine, None, None)

    def test_draft_rope_and_swiglu_metadata_are_validated_at_load(self):
        for edit, expected in (
            ({"partial_rotary_factor": 1.0}, "rotary_dim=64"),
            ({"swiglu_alpha": 0.0}, "positive SwiGLU"),
        ):
            raw = draft_config()
            raw["text_config"].update(edit)
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, expected):
                    MiniMaxM31DSpark._create_config(directory)
        raw = draft_config()
        del raw["text_config"]["swiglu_limit"]
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "SwiGLU checkpoint parameters"):
                MiniMaxM31DSpark._create_config(directory)

    def test_fa4_checks_declared_jit_dependencies_before_changing_import_path(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory, "flash_attn")
            (package / "cute").mkdir(parents=True)
            (package / "cute" / "interface.py").touch()
            distribution = SimpleNamespace(
                locate_file=lambda _: package,
                requires=[
                    "nvidia-cutlass-dsl==4.6.0.dev0",
                    "quack-kernels>=0.5.3",
                    'nvidia-cutlass-dsl[cu13]==4.6.0.dev0; extra == "cu13"',
                ],
            )
            flash = ModuleType("flash_attn")
            flash.__path__ = ["original"]
            prefix = "rtp_llm.models_py.modules.minimax_m31.kernels.fa4_import"
            with patch(
                f"{prefix}.importlib.metadata.distribution", return_value=distribution
            ), patch.dict(sys.modules, {"flash_attn": flash, "flash_attn.cute": None}):
                for versions, expected in (
                    (
                        {"nvidia-cutlass-dsl": "4.4.2", "quack-kernels": "0.4.1"},
                        "nvidia-cutlass-dsl",
                    ),
                    (
                        {"nvidia-cutlass-dsl": "4.6.0.dev0", "quack-kernels": "0.4.1"},
                        "quack-kernels",
                    ),
                ):
                    with self.subTest(versions=versions), patch(
                        f"{prefix}.importlib.metadata.version",
                        side_effect=versions.__getitem__,
                    ):
                        with self.assertRaisesRegex(ImportError, expected):
                            prefer_flash_attn_4_cute()
                        self.assertEqual(flash.__path__, ["original"])
                versions = {
                    "nvidia-cutlass-dsl": "4.6.0.dev0",
                    "quack-kernels": "0.5.3",
                }
                with patch(
                    f"{prefix}.importlib.metadata.version",
                    side_effect=versions.__getitem__,
                ):
                    prefer_flash_attn_4_cute()
                self.assertEqual(flash.__path__, [str(package), "original"])

    def test_namespace_fa4_module_fails_with_import_error(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory, "flash_attn")
            (package / "cute").mkdir(parents=True)
            (package / "cute" / "interface.py").touch()
            distribution = type(
                "Distribution", (), {"locate_file": lambda self, _: package}
            )()
            imported_cute = ModuleType("flash_attn.cute")
            imported_cute.__file__ = None
            with patch(
                "rtp_llm.models_py.modules.minimax_m31.kernels.fa4_import.importlib.metadata.distribution",
                return_value=distribution,
            ), patch.dict(sys.modules, {"flash_attn.cute": imported_cute}):
                with self.assertRaises(ImportError):
                    prefer_flash_attn_4_cute()

    def test_dense_draft_uses_main_cache_and_weight_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(draft_config()))
            config = MiniMaxM31DSpark._create_config(directory)
            kv_config = KVCacheConfig()
            kv_config.seq_size_per_block = 64
            config.init_precision_config(kv_config, "BF16")
            config.attn_config.tokens_per_block = kv_config.seq_size_per_block
            config.attn_config.kernel_tokens_per_block = kv_config.seq_size_per_block
            MiniMaxM31DSpark._apply_kv_cache_config(config, kv_config)
            MiniMaxM31DSpark._post_build_model_config(config)
            self.assertEqual(config.attn_config.sliding_window, 4096)
            self.assertEqual(config.dspark_target_layer_ids, [3, 17, 31, 45, 59])
            self.assertEqual(
                [[desc.tag for desc in layer] for layer in config.kv_cache_spec_descs],
                [["m31_kv"]] * 5,
            )
            self.assertEqual(config.kv_cache_spec_descs[0][0].dtype, DataType.TYPE_BF16)
            self.assertEqual(config.attn_config.tokens_per_block, 128)
            self.assertEqual(config.attn_config.kernel_tokens_per_block, 128)
            weights = MiniMaxM31DSparkWeight(
                config, ParallelismConfig(), HWKernelConfig(), KVCacheConfig()
            )._get_weight_info()
            self.assertEqual(len(weights.layer_weights), 5)
            self.assertEqual(
                {weight.name for weight in weights.weights} & {W.embedding, W.lm_head},
                {W.embedding, W.lm_head},
            )
            first_sources = [
                info.name
                for weight in weights.layer_weights[0]
                for component in weight.get_components()
                for info in getattr(component, "weights", ())
            ]
            self.assertIn(
                "language_model.model.dspark.layers.{i}.decoder_layer.self_attn.q_proj.weight",
                first_sources,
            )
            self.assertFalse(any("index_" in name for name in first_sources))
            target = MiniMaxM31.__new__(MiniMaxM31)
            target.model_config = config
            self.assertEqual(
                MiniMaxM31DSpark.speculative_weight_alias_names(target, config),
                (W.embedding, W.lm_head),
            )

    def test_swa_kernel_selection_with_and_without_fa4(self):
        from rtp_llm.models_py.modules.minimax_m31.dspark.dspark_swa import (
            paged_gqa_swa,
        )
        from rtp_llm.models_py.modules.minimax_m31.dspark.minimax_m31_dspark_layer import (
            _select_swa_kernel,
        )

        fake_fa4 = ModuleType("flash_attn.cute.interface")
        fake_fa4._flash_attn_fwd = object()
        fake_swa = ModuleType(
            "rtp_llm.models_py.modules.minimax_m31.dspark.dspark_swa_fa4"
        )
        fake_swa.paged_gqa_swa_fa4 = object()
        with patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with patch.dict(
                sys.modules,
                {
                    "flash_attn.cute.interface": fake_fa4,
                    fake_swa.__name__: fake_swa,
                },
            ):
                self.assertIs(_select_swa_kernel("cuda:0"), fake_swa.paged_gqa_swa_fa4)
            with patch.dict(sys.modules, {"flash_attn.cute.interface": None}):
                self.assertIs(_select_swa_kernel("cuda:0"), paged_gqa_swa)
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            self.assertIs(_select_swa_kernel("cuda:0"), paged_gqa_swa)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_native_swa_matches_reference_and_replays_after_fa4_rejection(self):
        from rtp_llm.models_py.modules.minimax_m31.dspark.dspark_swa import (
            paged_gqa_swa,
        )

        torch.manual_seed(17)
        q = torch.randn(2, 7, 8, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 7, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache = torch.randn(4, 2, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
        table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
        prefixes = torch.tensor([64, 200], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([7, 3], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q)

        def run():
            return paged_gqa_swa(
                q,
                k,
                v,
                cache,
                table,
                prefixes,
                lengths,
                window_left=63,
                causal=False,
                out=out,
            )

        def reference(prefix_values, live_values):
            expected = torch.zeros_like(q)
            for b, (prefix, live) in enumerate(zip(prefix_values, live_values)):
                past = (
                    cache[table[b].long()]
                    .permute(1, 0, 3, 2, 4)
                    .reshape(2, 256, 2, 128)
                )
                keys = (
                    torch.cat((past[0, :prefix], k[b, :live]))
                    .float()
                    .repeat_interleave(4, dim=1)
                )
                values = (
                    torch.cat((past[1, :prefix], v[b, :live]))
                    .float()
                    .repeat_interleave(4, dim=1)
                )
                for row in range(live):
                    positions = torch.arange(prefix + live, device="cuda")
                    allowed = (positions >= prefix + row - 63) & (
                        positions <= prefix + row + 63
                    )
                    scores = (
                        torch.einsum("hd,thd->ht", q[b, row].float(), keys) / 128**0.5
                    )
                    scores.masked_fill_(~allowed, -torch.inf)
                    expected[b, row] = torch.einsum(
                        "ht,thd->hd", scores.softmax(-1), values
                    ).to(q.dtype)
            return expected

        torch.testing.assert_close(
            run(), reference([64, 200], [7, 3]), atol=0.02, rtol=0.02
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        prefixes.copy_(torch.tensor([48, 160], device="cuda", dtype=torch.int32))
        lengths.copy_(torch.tensor([5, 0], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.testing.assert_close(
            out, reference([48, 160], [5, 0]), atol=0.02, rtol=0.02
        )

    def test_mxfp8_kernel_imports_on_current_main(self):
        from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
            mxfp8_quant_act_triton,
        )

        self.assertTrue(callable(mxfp8_quant_act_triton))

    def test_rejects_sparse_draft_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            value = draft_config()
            value["text_config"]["moe_layer_freq"][0] = 1
            Path(directory, "config.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "dense Gemma"):
                MiniMaxM31DSpark.create_config(directory)


if __name__ == "__main__":
    unittest.main()
