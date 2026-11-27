"""Load a real Qwen3.5 MoE owner from floating and prequantized checkpoints."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import CompressedW8A8Int8PerChannelQuantConfig
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3_next.language import Qwen3NextForCausalLM
from rtp_llm.models_py.new_models.qwen3_next.test.main_checkpoint_test import (
    _config,
    _dense_weights,
    _load_config,
)
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.models_py.quant_methods.w8a8_load_source import W8A8LoadSource
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.w8a8_int8_quant import (
    is_recipe_weight,
    quantize_weight_per_output_channel,
    scale_name_for,
)


def _checkpoint(linear):
    def values(shape):
        v = torch.arange(torch.Size(shape).numel()).float().reshape(shape) - 11
        return v.to(torch.bfloat16)

    cfg = _config(
        HybridAttentionType.LINEAR if linear else HybridAttentionType.NONE, tie=False
    )
    cfg.model_type, cfg.moe_layer_index, cfg.data_type = "qwen35_moe", [0], "bf16"
    cfg.moe_inter_size = 8
    root = "model.language_model.layers.0."
    raw = {
        key.replace("model.", "model.language_model.", 1): tensor.to(torch.bfloat16)
        for key, tensor in _dense_weights(True).items()
        if ".mlp." not in key and not (linear and ".self_attn." in key)
    }
    for name, shape in (
        ("gate.weight", (2, 4)),
        ("experts.gate_up_proj", (2, 16, 4)),
        ("experts.down_proj", (2, 4, 8)),
        ("shared_expert_gate.weight", (1, 4)),
        ("shared_expert.gate_proj.weight", (4, 4)),
        ("shared_expert.up_proj.weight", (4, 4)),
        ("shared_expert.down_proj.weight", (4, 4)),
    ):
        raw[root + "mlp." + name] = values(shape)
    if linear:
        for name, shape in (
            ("in_proj_qkv.weight", (12, 4)),
            ("in_proj_z.weight", (4, 4)),
            ("in_proj_b.weight", (2, 4)),
            ("in_proj_a.weight", (2, 4)),
            ("conv1d.weight", (12, 1, 2)),
            ("out_proj.weight", (4, 4)),
            ("A_log", (2,)),
            ("dt_bias", (2,)),
            ("norm.weight", (2,)),
        ):
            raw[root + "linear_attn." + name] = values(shape)
    return cfg, raw


class Qwen35LoadTimeW8A8Test(unittest.TestCase):
    def test_float_qkv_exception_in_online_and_prequantized_checkpoints(self):
        cfg, raw = _checkpoint(False)
        prefix = "model.language_model.layers.0.self_attn."
        physical = [prefix + key for key in ("q_proj", "k_proj", "v_proj")]
        q_gate = raw[prefix + "q_proj.weight"].reshape(2, 2, 2, 4)
        for encoded, ignored in (
            (False, physical),
            (False, [prefix + "qkv_proj"]),
            (True, physical),
        ):
            with self.subTest(encoded=encoded), tempfile.TemporaryDirectory() as path:
                source = CompressedW8A8Int8PerChannelQuantConfig(
                    is_quanted=encoded, ignore_patterns=ignored
                )
                cfg.quant_config = source
                quant = QuantizationConfig(source.get_method(), source)
                tensors = {}
                for key, weight in raw.items():
                    if (
                        encoded
                        and is_recipe_weight(key)
                        and key.removesuffix(".weight") not in physical
                    ):
                        tensors[key], tensors[scale_name_for(key)] = (
                            quantize_weight_per_output_channel(weight)
                        )
                    else:
                        tensors[key] = weight
                save_file(tensors, str(Path(path) / "model.safetensors"))
                options = replace(
                    _load_config(quant_config=quant), compute_dtype=torch.bfloat16
                )
                model = NewModelLoader(cfg, options, model_path=path).load()
                attn = model.layers[0].self_attn
                torch.testing.assert_close(
                    attn.gate.weight, q_gate[:, 1].reshape(4, 4), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    attn.qkv_proj.weight[:4], q_gate[:, 0].reshape(4, 4), rtol=0, atol=0
                )
                self.assertEqual(attn.o_proj.weight.dtype, torch.int8)

    def test_fp8_exclusions_follow_the_physical_q_projection(self):
        cfg = _config()
        prefix = "model.layers.0.self_attn."
        qkv = [prefix + key for key in ("q_proj", "k_proj", "v_proj")]

        def owner(ignored):
            opts = replace(
                _load_config(
                    quant_config=QuantizationConfig("fp8_online", ignored_layers=ignored)
                ),
                compute_dtype=torch.bfloat16,
            )
            return Qwen3NextForCausalLM(cfg, opts)

        raw = {k: v.bfloat16() for k, v in _dense_weights().items()}
        with tempfile.TemporaryDirectory() as path:
            checkpoint = str(Path(path) / "model.safetensors")
            save_file(raw, checkpoint)
            for ignored in (qkv, [prefix + "qkv_proj"], []):
                with self.subTest(ignored=ignored):
                    model = owner(ignored)
                    model.load_weights(get_all_weights([checkpoint]))
                    NewModelLoader._validate_loaded_weights(model)
                    attn = model.layers[0].self_attn
                    self.assertEqual(
                        isinstance(attn.gate.quant_method, UnquantizedLinearMethod),
                        bool(ignored),
                    )
                    self.assertIs(
                        type(attn.gate.quant_method), type(attn.qkv_proj.quant_method)
                    )
                    self.assertNotIsInstance(
                        attn.o_proj.quant_method, UnquantizedLinearMethod
                    )
        with self.assertRaisesRegex(ValueError, "partially match fused layer"):
            owner(qkv[:1])

    def test_real_model_hybrid_attention_shared_and_routed_experts_both_tp_ranks(self):
        for linear in (False, True):
            cfg, raw = _checkpoint(linear)
            compressed = {}
            for key, tensor in raw.items():
                if is_recipe_weight(key):
                    compressed[key], compressed[scale_name_for(key)] = (
                        quantize_weight_per_output_channel(tensor)
                    )
                else:
                    compressed[key] = tensor
            with tempfile.TemporaryDirectory() as directory:
                for label, tensors in (("float", raw), ("encoded", compressed)):
                    root = Path(directory) / label
                    root.mkdir()
                    save_file(tensors, str(root / "model.safetensors"))
                for rank in (0, 1):
                    with self.subTest(linear=linear, rank=rank):
                        states = []
                        for label in ("float", "encoded"):
                            source = CompressedW8A8Int8PerChannelQuantConfig(
                                is_quanted=label == "encoded", load_chunk_rows=2
                            )
                            cfg.quant_config = source
                            options = replace(
                                _load_config(
                                    tp_size=2,
                                    tp_rank=rank,
                                    quant_config=QuantizationConfig(
                                        source.get_method(), source_config=source
                                    ),
                                ),
                                compute_dtype=torch.bfloat16,
                            )
                            path = Path(directory) / label
                            if linear:
                                # GDN forward is accelerator-only. Stage and validate
                                # the real model's weights on CPU, without bypassing
                                # NewModelLoader's runtime check or pretending to run it.
                                model = Qwen3NextForCausalLM(cfg, options)
                                files = [str(path / "model.safetensors")]
                                view = (
                                    W8A8LoadSource.from_files(
                                        files, options.quant_config
                                    )
                                    if label == "float"
                                    else None
                                )
                                model.load_weights(
                                    get_all_weights(files, w8a8_source=view)
                                )
                                NewModelLoader._validate_loaded_weights(model)
                            else:
                                model = NewModelLoader(
                                    cfg, options, model_path=str(path)
                                ).load()
                            states.append(model.state_dict())
                        actual, reference = states
                        self.assertEqual(actual.keys(), reference.keys())
                        self.assertTrue(
                            any(t.dtype == torch.int8 for t in actual.values())
                        )
                        for key, expected in reference.items():
                            torch.testing.assert_close(
                                actual[key], expected, rtol=0, atol=0
                            )


if __name__ == "__main__":
    unittest.main()
