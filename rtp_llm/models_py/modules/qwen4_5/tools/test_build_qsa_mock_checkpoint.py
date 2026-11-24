"""Unit tests for build_qsa_mock_checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from rtp_llm.config.qwen4_5_config import (
    parse_qwen4_5_config_json,
    parse_qwen4_5_quant_config,
)
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
    SafetensorsShardReader,
    write_safetensors_file,
)
from rtp_llm.models_py.modules.qwen4_5.tokenizer import Qwen2BPETokenizer
from rtp_llm.models_py.modules.qwen4_5.tools.build_qsa_mock_checkpoint import (
    QSA_SHARD_NAME,
    QSAConfigParams,
    SYNTHETIC_ACTIVE_IDS,
    _parse_args,
    build_overlay_40l,
    build_synthetic_8l,
    validate_checkpoint,
)
from rtp_llm.models_py.modules.qwen4_5.yoco import YoCoSharingPlan


class TestBuildQSAMockCheckpoint(unittest.TestCase):
    def test_default_qsa_params_and_cli_override(self) -> None:
        default_qsa = QSAConfigParams()
        self.assertEqual(default_qsa.indexer_n_heads, 8)
        self.assertEqual(default_qsa.indexer_head_dim, 128)
        self.assertEqual(default_qsa.indexer_kv_heads, 1)
        self.assertEqual(default_qsa.indexer_compress_ratio, 4)
        self.assertEqual(default_qsa.indexer_budget, 2048)

        args_default = _parse_args(["--mode", "synthetic-8l"])
        self.assertEqual(args_default.indexer_n_heads, 8)
        args_override = _parse_args(["--mode", "synthetic-8l", "--indexer-n-heads", "4"])
        self.assertEqual(args_override.indexer_n_heads, 4)

    def test_protected_mtp_test_path_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected path"):
            build_synthetic_8l(
                output_dir=Path("/3fs-data/3fs/mtp_test/bad_out"),
                qsa=QSAConfigParams(),
                qsa_seed=20261124,
                base_seed=2027,
                hidden_size=512,
                max_position_embeddings=8192,
            )

    def test_synthetic_8l_contract_and_determinism_both_head_counts(self) -> None:
        for n_heads in (8, 4):
            qsa = QSAConfigParams(indexer_n_heads=n_heads)
            expected_qk_out = (n_heads + 1) * 128
            with self.subTest(indexer_n_heads=n_heads):
                with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
                    out1 = Path(tmp1) / "8l_a"
                    out2 = Path(tmp2) / "8l_b"
                    s1 = build_synthetic_8l(
                        output_dir=out1,
                        qsa=qsa,
                        qsa_seed=20261124,
                        base_seed=2027,
                        hidden_size=512,
                        max_position_embeddings=8192,
                    )
                    s2 = build_synthetic_8l(
                        output_dir=out2,
                        qsa=qsa,
                        qsa_seed=20261124,
                        base_seed=2027,
                        hidden_size=512,
                        max_position_embeddings=8192,
                    )
                    self.assertEqual(s1["full_attention_layers"], [3, 7])
                    self.assertEqual(s1["tensor_count"], s2["tensor_count"])
                    self.assertEqual(s1["total_size"], s2["total_size"])

                    m1 = json.loads((out1 / "mock_qsa_manifest.json").read_text())
                    m2 = json.loads((out2 / "mock_qsa_manifest.json").read_text())
                    self.assertEqual(m1["qsa_tensors"], m2["qsa_tensors"])
                    self.assertEqual(m1["qsa_tensor_count"], 6)
                    self.assertEqual(m1["qsa_config"]["indexer_n_heads"], n_heads)

                    cfg_json = json.loads((out1 / "config.json").read_text())
                    self.assertEqual(cfg_json["indexer_n_heads"], n_heads)
                    arch = parse_qwen4_5_config_json(cfg_json)
                    parse_qwen4_5_quant_config(cfg_json)
                    plan = YoCoSharingPlan.from_config(arch)
                    self.assertTrue(plan.is_full_attn_source(3))
                    self.assertTrue(plan.is_full_attn_consumer(7))
                    self.assertEqual(plan.source_for(7), 3)

                    tok = Qwen2BPETokenizer(out1)
                    encoded = tok.encode("ADGJMPSV")
                    self.assertEqual(encoded, list(SYNTHETIC_ACTIVE_IDS))
                    self.assertEqual(tok.decode(encoded, skip_special_tokens=False), "ADGJMPSV")

                    reader = SafetensorsShardReader(out1, prefer_safetensors=False)
                    try:
                        for layer_idx in (3, 7):
                            qk = reader.get_tensor(f"model.layers.{layer_idx}.self_attn.indexer.index_qk_proj.weight")
                            qn = reader.get_tensor(f"model.layers.{layer_idx}.self_attn.indexer.q_layernorm.weight")
                            kn = reader.get_tensor(f"model.layers.{layer_idx}.self_attn.indexer.k_layernorm.weight")
                            self.assertEqual(tuple(qk.shape), (expected_qk_out, 512))
                            self.assertEqual(qk.dtype, torch.bfloat16)
                            self.assertAlmostEqual(float(qk.float().std().item()), 1.0 / math.sqrt(512), delta=0.01)
                            self.assertTrue(torch.equal(qn, torch.zeros(128, dtype=torch.bfloat16)))
                            self.assertTrue(torch.equal(kn, torch.zeros(128, dtype=torch.bfloat16)))
                    finally:
                        reader.close()

    def test_overlay_mode_preserves_base_and_creates_qsa_shard(self) -> None:
        for n_heads in (8, 4):
            expected_qk_out = (n_heads + 1) * 128
            with self.subTest(indexer_n_heads=n_heads):
                with tempfile.TemporaryDirectory() as tmp:
                    base = Path(tmp) / "base"
                    out = Path(tmp) / "overlay"
                    base.mkdir()
                    t0 = torch.randn((16, 128), dtype=torch.bfloat16)
                    shard_name = "model-00001-of-00001.safetensors"
                    write_safetensors_file({"model.embed_tokens.weight": t0}, base / shard_name)
                    base_cfg = {
                        "architectures": ["Qwen4_5ForCausalLM"],
                        "model_type": "qwen4_5",
                        "vocab_size": 16,
                        "hidden_size": 128,
                        "num_hidden_layers": 8,
                        "intermediate_size": 256,
                        "first_k_dense_replace": 1,
                        "full_attention_interval": 4,
                        "layer_types": [
                            "linear_attention", "linear_attention", "linear_attention", "full_attention",
                            "linear_attention", "linear_attention", "linear_attention", "full_attention",
                        ],
                        "num_attention_heads": 4,
                        "num_key_value_heads": 1,
                        "head_dim": 256,
                        "partial_rotary_factor": 0.25,
                        "rms_norm_eps": 1e-6,
                        "routed_expert_hidden_size": 64,
                        "moe_intermediate_size": 64,
                        "shared_expert_intermediate_size": 32,
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "linear_num_key_heads": 2,
                        "linear_key_head_dim": 64,
                        "linear_num_value_heads": 4,
                        "linear_value_head_dim": 64,
                        "hc_count": 4,
                        "hc_lowrank": 16,
                        "ple_layer_ids": [2],
                        "ple_embed_dim": 128,
                        "ple_conv_kernel_size": 4,
                        "ngram_size": 3,
                        "heads_per_ngram": 4,
                        "ngram_vocab_size_base": 64,
                        "make_ngram_vocab_size_divisible_by": 8,
                        "split_ngram_parts": 4,
                        "share_map": "4->8",
                        "quantization_config": {
                            "quant_method": "mxfp4",
                            "ple_quant_scheme": "fp8_pertensor",
                        },
                    }
                    (base / "config.json").write_text(json.dumps(base_cfg))
                    (base / "model.safetensors.index.json").write_text(
                        json.dumps({
                            "metadata": {"total_size": 16 * 128 * 2},
                            "weight_map": {"model.embed_tokens.weight": shard_name},
                        })
                    )
                    (base / "tokenizer_config.json").write_text("{}")

                    qsa = QSAConfigParams(indexer_n_heads=n_heads)
                    summary = build_overlay_40l(base_dir=base, output_dir=out, qsa=qsa, qsa_seed=20261124)
                    self.assertEqual(summary["symlinked_shards"], 1)
                    self.assertEqual(summary["shard_count"], 2)
                    self.assertEqual(summary["tensor_count"], 7)
                    self.assertTrue((out / shard_name).is_symlink())
                    self.assertTrue((out / "tokenizer_config.json").is_symlink())
                    self.assertTrue((out / QSA_SHARD_NAME).is_file())
                    self.assertFalse((out / "config.json").is_symlink())
                    self.assertFalse((out / "model.safetensors.index.json").is_symlink())
                    out_cfg = json.loads((out / "config.json").read_text())
                    self.assertEqual(out_cfg["indexer_n_heads"], n_heads)
                    validate_checkpoint(out, qsa=qsa, check_all_shard_headers=True)

                    reader = SafetensorsShardReader(out, prefer_safetensors=False)
                    try:
                        qk = reader.get_tensor("model.layers.3.self_attn.indexer.index_qk_proj.weight")
                        self.assertEqual(tuple(qk.shape), (expected_qk_out, 128))
                    finally:
                        reader.close()


if __name__ == "__main__":
    unittest.main()
