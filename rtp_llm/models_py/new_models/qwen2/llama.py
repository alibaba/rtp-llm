"""Bias-free HF Llama checkpoint owner, including compatible Mistral HF weights."""

import json
from pathlib import Path

from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM


class LlamaForCausalLM(Qwen2ForCausalLM):
    # The HF Q/K rotary channel ordering, gated MLP and TP slicing are the
    # same as Qwen2. Do not create (or falsely require) a packed QKV bias.
    QKV_BIAS = False

    def load_meta_checkpoint(self, checkpoint):
        from rtp_llm.models_py.new_models.qwen2.llama_meta_checkpoint import (
            load_meta_checkpoint,
        )

        load_meta_checkpoint(self, checkpoint)

    def __init__(self, model_config, load_config):
        get = (
            model_config.get
            if isinstance(model_config, dict)
            else lambda name, default=None: getattr(model_config, name, default)
        )
        path = get("ckpt_path")
        hf = model_config if isinstance(model_config, dict) else {}
        if path:
            root = Path(path)
            if (root / "config.json").is_file():
                with (root / "config.json").open() as stream:
                    hf = json.load(stream)
            elif not (root / "params.json").is_file():
                # FT exports may have only a rank index; architecture and
                # topology come from the caller and the FT importer. Never
                # mistake an ordinary HF shard index for an FT snapshot.
                index = root / "model.safetensors.index.json"
                if index.is_file():
                    with index.open() as stream:
                        is_ft = json.load(stream).get("is_ft_style_weight") is True
                else:
                    is_ft = False
                if not is_ft:
                    raise ValueError("Missing Llama config.json or Meta/FT metadata")
        family = hf.get("model_type", "llama")
        expected = {"llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM"}
        if family not in expected or expected[family] not in hf.get(
            "architectures", [expected.get(family)]
        ):
            raise ValueError(f"Unsupported HF decoder for llama checkpoint: {family!r}")
        if (
            hf.get("attention_bias")
            or hf.get("mlp_bias")
            or hf.get("hidden_act", "silu") != "silu"
        ):
            raise ValueError("The llama checkpoint owner requires bias-free SwiGLU")
        # This is the explicitly requested llama route, not a Mistral runtime
        # registration. Full and sliding causal attention agree only within
        # one window; never silently run full attention past that point.
        window = hf.get("sliding_window")
        if window is not None and not 0 < get("max_seq_len", 0) <= window:
            raise ValueError("The llama route requires max_seq_len <= sliding_window")
        super().__init__(model_config, load_config)
