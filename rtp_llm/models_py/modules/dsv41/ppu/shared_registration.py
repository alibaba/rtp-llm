"""Compatibility name for the inherited V4.1 PPU draft."""

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.deepseek_v41 import DeepSeekV41DSpark

DeepSeekV41SharedPPUDSpark = DeepSeekV41DSpark
register_model("deepseek_v41_ppu_shared_dspark", DeepSeekV41SharedPPUDSpark)
