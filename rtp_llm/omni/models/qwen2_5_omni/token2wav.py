import json
import logging
import os
from dataclasses import dataclass, field
from typing import List

from rtp_llm.model_factory_register import register_model

logger = logging.getLogger(__name__)


@dataclass
class Token2WavConfig:
    ckpt_path: str = ""
    dit_depth: int = 22
    dit_dim: int = 1024
    dit_heads: int = 16
    dit_head_dim: int = 64
    mel_dim: int = 80
    dit_num_embeds: int = 8193
    dit_ff_mult: int = 2
    upsample_rates: List[int] = field(default_factory=lambda: [5, 3, 2, 2, 2, 2])
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])


class Qwen2_5OmniToken2Wav:
    @classmethod
    def _create_config(cls, ckpt_path: str) -> Token2WavConfig:
        config = Token2WavConfig()
        config.ckpt_path = ckpt_path

        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config

        with open(config_path) as f:
            root_config = json.load(f)

        t2w_config = root_config.get("token2wav_config", {})
        dit_config = t2w_config.get("dit_config", {})
        bigvgan_config = t2w_config.get("bigvgan_config", {})

        config.dit_depth = dit_config.get("depth", config.dit_depth)
        config.dit_dim = dit_config.get("dim", config.dit_dim)
        config.dit_heads = dit_config.get("heads", config.dit_heads)
        config.dit_head_dim = dit_config.get("head_dim", config.dit_head_dim)
        config.mel_dim = dit_config.get("mel_dim", config.mel_dim)
        config.dit_num_embeds = dit_config.get("num_embeds", config.dit_num_embeds)
        config.dit_ff_mult = dit_config.get("ff_mult", config.dit_ff_mult)

        config.upsample_rates = bigvgan_config.get(
            "upsample_rates", config.upsample_rates
        )
        config.upsample_initial_channel = bigvgan_config.get(
            "upsample_initial_channel", config.upsample_initial_channel
        )
        config.resblock_kernel_sizes = bigvgan_config.get(
            "resblock_kernel_sizes", config.resblock_kernel_sizes
        )

        logger.info(
            f"Token2Wav config: DiT depth={config.dit_depth} dim={config.dit_dim}, "
            f"BigVGAN upsample_rates={config.upsample_rates}"
        )
        return config

    @staticmethod
    def get_weight_cls():
        return None


register_model("qwen2_5_omni_token2wav", Qwen2_5OmniToken2Wav)
