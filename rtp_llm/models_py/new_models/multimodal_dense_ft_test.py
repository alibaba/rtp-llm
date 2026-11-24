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
    def round_trip(self, tag, *, cache_fp8=False):
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
                cfg.data_type, cfg.task_type = "fp32", TaskType.LANGUAGE_MODEL
                cfg.render_config, cfg.generate_env_config = (
                    RenderConfig(),
                    GenerateEnvConfig(),
                )
                cache = KVCacheConfig()
                cache.fp8_kv_cache = int(cache_fp8 and writing)
                cfg.init_precision_config(cache, "fp32")
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
            if cache_fp8:
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

    def test_cosyvoice_separate_vocabularies(self):
        self.round_trip("cosyvoice_qwen")

    def test_dense_cache_fp8_metadata(self):
        self.round_trip("qwen_2", cache_fp8=True)

    def test_remote_vision_text_owners(self):
        for tag in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
            with self.subTest(model=tag):
                self.round_trip(tag)


if __name__ == "__main__":
    unittest.main()
