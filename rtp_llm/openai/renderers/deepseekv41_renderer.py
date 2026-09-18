"""V4.1 chat formatting from the checkpoint's authoritative encoder."""

import importlib.util
import os
import sys

from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer


class DeepseekV41Renderer(DeepseekV4Renderer):
    def _load_encoding_module(self, ckpt_path: str):
        script = os.path.join(ckpt_path, "encoding", "encoding.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"V4.1 checkpoint encoder is missing: {script}")
        spec = importlib.util.spec_from_file_location(
            "rtp_deepseek_v41_encoding", script
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load V4.1 checkpoint encoder: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


register_renderer("deepseek_v41", DeepseekV41Renderer)
