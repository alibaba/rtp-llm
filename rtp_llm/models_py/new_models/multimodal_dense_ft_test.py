"""Deployed text-only owners read actual FT export shards, not an HF rename."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig, VitConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.weight_mapper import get_all_weights
from rtp_llm.ops import (
    FMHAConfig,
    HWKernelConfig,
    MoeConfig,
    ParallelismConfig,
    TaskType,
    VitSeparation,
)
from rtp_llm.tools.convert.ft_export_job import create_ft_export_job
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index


class MultimodalDenseFtTest(unittest.TestCase):
    def round_trip(self, tag, *, cache_fp8=False, fp8_activation=None):
        qwen3 = tag == "qwen3_vl"
        vl = tag in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")
        text = dict(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_hidden_layers=1,
            vocab_size=16,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope_theta=10000,
            tie_word_embeddings=False,
        )
        if qwen3:
            hf = Qwen3ForCausalLM(Qwen3Config(**text))
        else:
            hf = Qwen2ForCausalLM(Qwen2Config(**text))
        weights = dict(hf.state_dict())
        config = dict(text, model_type=tag)
        if fp8_activation is not None:
            weights = {name: tensor.bfloat16() for name, tensor in weights.items()}
            scheme = {"num_bits": 8, "type": "float", "strategy": "tensor"}
            config["quantization_config"] = {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": dict(scheme, dynamic=False),
                        "input_activations": dict(
                            scheme, dynamic=fp8_activation == "dynamic"
                        ),
                    }
                },
                "ignore": ["lm_head"],
            }
            for name in list(weights):
                if name.endswith("_proj.weight"):
                    weights[name] = weights[name].to(torch.float8_e4m3fn)
                    stem = name.removesuffix(".weight")
                    weights[stem + ".weight_scale"] = torch.ones(1)
                    if fp8_activation == "static":
                        step = {"o_proj": 0.125, "down_proj": 0.0625}.get(
                            stem.split(".")[-1], 0.25
                        )
                        weights[stem + ".input_scale"] = torch.tensor([step])
        if tag == "cosyvoice_qwen":
            config["input_vocab_size"] = 27
            weights["model.embed_tokens.weight"] = torch.randn(27, 8)
        elif vl:
            text["rope_scaling"] = {"mrope_section": [1, 1, 0]}
            config = dict(text_config=text) if qwen3 else dict(text)
            config.update(
                model_type=tag,
                vision_start_token_id=2,
                vision_end_token_id=3,
                vision_config={},
            )
            if qwen3:
                weights = {
                    name.replace("model.", "model.language_model.", 1): value
                    for name, value in weights.items()
                }
        with tempfile.TemporaryDirectory() as folder:
            source, ft = Path(folder) / "hf", Path(folder) / "ft"
            source.mkdir()
            ft.mkdir()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel(
                        {"[UNK]": 0, **{str(i): i for i in range(1, 16)}},
                        unk_token="[UNK]",
                    )
                ),
                unk_token="[UNK]",
            )
            tokenizer.save_pretrained(source)
            tokenizer.save_pretrained(ft)
            (source / "config.json").write_text(json.dumps(config))
            shutil.copyfile(source / "config.json", ft / "config.json")
            save_file(weights, source / "model.safetensors")
            cls = ModelFactory.get_model_cls(tag)

            def service(path, method, *, writing=False):
                cfg = cls.create_config(str(path))
                cfg.model_type, cfg.tokenizer_path = tag, str(path)
                dtype = "bf16" if fp8_activation is not None else "fp32"
                cfg.data_type, cfg.task_type = dtype, TaskType.LANGUAGE_MODEL
                cfg.render_config, cfg.generate_env_config = (
                    RenderConfig(),
                    GenerateEnvConfig(),
                )
                cache = KVCacheConfig()
                cache.fp8_kv_cache = int(cache_fp8 and writing)
                cfg.init_precision_config(cache, dtype)
                pc = ParallelismConfig()
                pc.ep_size = 1
                vit = VitConfig()
                if vl:
                    vit.vit_separation = VitSeparation.VIT_SEPARATION_REMOTE
                return cls(
                    model_config=cfg,
                    parallelism_config=pc,
                    kv_cache_config=cache,
                    hw_kernel_config=HWKernelConfig(),
                    fmha_config=FMHAConfig(),
                    moe_config=MoeConfig(),
                    vit_config=vit,
                    merge_lora=False,
                    max_generate_batch_size=1,
                    load_method=method,
                )

            create_ft_export_job(
                service(source, "scratch", writing=True)
            ).dump_weight_as_ft_style("cpu", str(ft))
            generate_ft_index(str(ft), {"FP8_KV_CACHE": int(cache_fp8)})

            def load(path, method):
                wrapper = service(path, method)
                if fp8_activation is not None:
                    # Exercise real readers, destination owners and strict coverage.
                    # A CPU worker cannot activate a serving FP8 executor.
                    cfg = wrapper.model_config
                    quant = QuantizationConfig(
                        cfg.quant_config.get_method(), source_config=cfg.quant_config
                    )
                    opts = NewLoaderConfig.from_runtime(
                        wrapper.parallelism_config,
                        device="cpu",
                        compute_dtype=torch.bfloat16,
                        quant_config=quant,
                        load_method=method,
                    )
                    loader = NewModelLoader(cfg, opts, model_path=str(path))
                    cp = (
                        FTCheckpoint.open(str(path), CheckpointTopology())
                        if path == ft else None
                    )
                    owner = loader._create_checkpoint_model(cp)
                    if path == ft:
                        self.assertIsNotNone(cp)
                        cp.read_backend = method
                        owner.load_ft_checkpoint(cp)
                    else:
                        files = [str(path / "model.safetensors")]
                        owner.load_weights(get_all_weights(files))
                    loader._validate_loaded_weights(owner)
                    return owner
                # Select the CPU device only; run the normal public model
                # load, completeness checks and runtime weight publication.
                with patch.object(wrapper, "_get_device_str", return_value="cpu"):
                    wrapper.load()
                self.assertEqual(
                    wrapper.model_weights_loader.loaded_method.value, method
                )
                return wrapper.py_model

            reference = load(source, "scratch").state_dict()
            methods = (
                ("scratch",)
                if os.getenv("NEWLOADER_TEST_NATIVE_FST") == "0"
                else ("scratch", "fastsafetensors")
            )
            for method in methods:
                with self.subTest(model=tag, method=method):
                    model = load(ft, method)
                    actual = model.state_dict()
                    self.assertEqual(reference.keys(), actual.keys())
                    for name, expected in reference.items():
                        if fp8_activation == "static" and name.endswith(".input_scale"):
                            # The HF fused owners defer merging their input scales
                            # until device activation. Compare the archive's value
                            # directly to the calibration supplied to the exporter.
                            physical = name.replace("qkv_proj", "q_proj").replace(
                                "gate_up_proj", "gate_proj"
                            )
                            expected = weights["model." + physical]
                        if expected.dtype == torch.float8_e4m3fn:
                            self.assertEqual(actual[name].dtype, expected.dtype)
                            torch.testing.assert_close(
                                actual[name].view(torch.uint8), expected.view(torch.uint8)
                            )
                        else:
                            torch.testing.assert_close(
                                actual[name], expected, rtol=0, atol=0
                            )
                    if tag == "cosyvoice_qwen":
                        self.assertEqual(
                            (
                                model.embed_tokens.weight.shape[0],
                                model.lm_head.weight.shape[0],
                            ),
                            (27, 16),
                        )
            if cache_fp8 and fp8_activation != "static":
                name = "rank_00_00_00.layers.0.self_attention_weights.attention_output_weight.static_quant_reciprocal"
                layout = json.loads((ft / "model.safetensors.index.json").read_text())
                shard = ft / layout["weight_map"][name]
                # Rewrite independent tensors, never mmap aliases of the file
                # being overwritten. The malformed file still uses the real FT index.
                tensors = {
                    key: value.clone() for key, value in load_file(shard).items()
                }
                torch.testing.assert_close(
                    tensors[name], torch.ones(1, dtype=torch.float32)
                )
                for bad in (
                    torch.tensor([2.0]),
                    torch.ones(2),
                    torch.ones(1, dtype=torch.float16),
                ):
                    tensors[name] = bad
                    save_file(tensors, shard)
                    with self.assertRaisesRegex(
                        ValueError, "KV-cache dummy reciprocal"
                    ):
                        load(ft, "scratch")
                if fp8_activation == "dynamic":
                    tensors[name] = torch.ones(1)
                    for forbidden in (
                        name.removesuffix("_reciprocal"),
                        name.replace(
                            "self_attention_weights.attention_output_weight",
                            "pre_layernorm_weights",
                        ),
                    ):
                        save_file({**tensors, forbidden: torch.ones(1)}, shard)
                        generate_ft_index(str(ft), {"FP8_KV_CACHE": 1})
                        with self.assertRaisesRegex(ValueError, "static source disagrees"):
                            load(ft, "scratch")

    def test_cosyvoice_separate_vocabularies(self):
        self.round_trip("cosyvoice_qwen")

    def test_dense_cache_fp8_metadata(self):
        self.round_trip("qwen_2", cache_fp8=True)

    def test_dynamic_fp8_ft_with_fp8_kv_cache(self):
        self.round_trip("qwen_2", cache_fp8=True, fp8_activation="dynamic")

    def test_static_fp8_ft_preserves_calibration_with_fp8_kv_cache(self):
        self.round_trip("qwen_2", cache_fp8=True, fp8_activation="static")

    def test_remote_vision_text_owners(self):
        for tag in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
            with self.subTest(model=tag):
                self.round_trip(tag)


if __name__ == "__main__":
    unittest.main()
