from typing import Any, Dict, List, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.omni.models.qwen2_5_omni.token2wav_bigvgan import BigVGAN, BigVGANConfig
from rtp_llm.omni.models.qwen2_5_omni.token2wav_dit import OmniDiT, OmniDiTConfig
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity
from rtp_llm.utils.util import get_config_from_path


class Qwen25OmniToken2WavWeight(ModelDeployWeightInfo):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def _get_weight_info(self):
        weights: List[AtomicWeight] = []
        return ModelWeightInfo(layer_weights=[], weights=weights)


class Qwen25OmniToken2Wav(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path

        config_json = get_config_from_path(ckpt_path)
        if config_json and "token2wav_config" in config_json:
            t2w_cfg = config_json["token2wav_config"]
            dit_cfg = t2w_cfg.get("dit_config", {})
            bigvgan_cfg = t2w_cfg.get("bigvgan_config", {})

            config.hidden_size = dit_cfg.get("dim", 1024)
            config.num_layers = dit_cfg.get("depth", 22)
            config.vocab_size = dit_cfg.get("num_embeds", 8193)

        return config

    @staticmethod
    def get_weight_cls():
        return Qwen25OmniToken2WavWeight

    def _create_python_model(self):
        config_json = get_config_from_path(self.model_config.ckpt_path)
        if config_json and "token2wav_config" in config_json:
            t2w_cfg = config_json["token2wav_config"]
            self.dit = OmniDiT(OmniDiTConfig.from_dict(t2w_cfg.get("dit_config", {})))
            self.bigvgan = BigVGAN(
                BigVGANConfig.from_dict(t2w_cfg.get("bigvgan_config", {}))
            )
        else:
            self.dit = OmniDiT(OmniDiTConfig())
            self.bigvgan = BigVGAN(BigVGANConfig())

    @torch.inference_mode()
    def generate_audio(
        self,
        codec_ids: torch.Tensor,
        num_steps: int = 32,
        spk_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = codec_ids.shape
        mel_dim = self.dit.config.mel_dim

        x = torch.randn(
            batch_size, seq_len, mel_dim, device=codec_ids.device
        )

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_val = i * dt
            t = torch.full(
                (batch_size,), t_val, device=codec_ids.device, dtype=x.dtype
            )
            v = self.dit(x, t, codec_ids, spk_emb)
            x = x + v * dt

        mel = x.transpose(1, 2)
        waveform = self.bigvgan(mel)
        return waveform


register_model("qwen2_5_omni_token2wav", Qwen25OmniToken2Wav)
