"""Keep configuration and frontend preprocessing independent of GPU libraries."""

import importlib.abc
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def check_cpu_frontend():
    gpu_imports = []

    class NoGpuLibraries(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {
                "librtp_compute_ops",
                "libth_transformer",
                "rtp_llm.ops.compute_ops",
                "rtp_llm.models_py.modules",
            }:
                gpu_imports.append(fullname)
                raise AssertionError(f"CPU frontend imported GPU code: {fullname}")

    sys.meta_path.insert(0, NoGpuLibraries())
    import torch
    from PIL import Image

    from rtp_llm.config.py_config_modules import PyEnvConfigs
    from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
    from rtp_llm.frontend.frontend_server import FrontendServer
    from rtp_llm.model_factory import ModelFactory
    from rtp_llm.models.multimodal.deepseek_v41_processor import (
        V41ImageProcessorConfig,
        preprocess_image,
    )
    from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer

    assert not torch.cuda.is_available()
    text = dict(
        num_hidden_layers=2,
        hidden_size=128,
        vocab_size=129280,
        num_attention_heads=2,
        head_dim=64,
        qk_rope_head_dim=32,
        compress_ratios=[1, 2],
        o_groups=1,
        o_lora_rank=32,
        index_head_dim=32,
        index_n_heads=2,
        index_topk=16,
        routed_scaling_factor=1.0,
        num_experts_per_tok=2,
        n_routed_experts=4,
        moe_intermediate_size=64,
        n_shared_experts=1,
        kv_source_layer_ids=[0, 1],
        index_source_layer_ids=[0],
        max_position_embeddings=1048576,
    )
    with tempfile.TemporaryDirectory() as checkpoint:
        Path(checkpoint, "config.json").write_text(
            json.dumps({"text_config": text, "dtype": "bfloat16"})
        )
        env = PyEnvConfigs()
        env.model_args.model_type = "deepseek_v41"
        env.model_args.ckpt_path = checkpoint
        env.model_args.tokenizer_path = checkpoint
        config = ModelFactory.create_model_config(
            model_args=env.model_args,
            lora_config=env.lora_config,
            kv_cache_config=env.kv_cache_config,
            profiling_debug_logging_config=env.profiling_debug_logging_config,
        )
        assert config.is_deepseek_v41
        assert config.mm_model_config.is_multimodal
        assert config.hidden_size == 128

    patches, *_ = preprocess_image(
        Image.new("RGB", (42, 84)), V41ImageProcessorConfig()
    )
    assert patches.device.type == "cpu"
    assert not torch.cuda.is_initialized()
    assert not gpu_imports, f"CPU frontend attempted GPU imports: {gpu_imports}"
    assert FrontendServer and DashScInferenceServicer and DeepseekV41Renderer


class CpuFrontendTest(unittest.TestCase):
    def test_fresh_process_without_gpu_libraries(self):
        # A fresh process prevents earlier tests from hiding eager imports.
        result = subprocess.run(
            [sys.executable, __file__, "--check-cpu-frontend"],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--check-cpu-frontend" in sys.argv:
        check_cpu_frontend()
    else:
        unittest.main()
