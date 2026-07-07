from rtp_llm.models_py.new_models.deepseek_vl2.language import DeepSeekVLV2ForCausalLM
from rtp_llm.models_py.new_models.deepseek_vl2.model import (
    DeepSeekVLV2ForConditionalGeneration,
)

# Registration is done in rtp_llm.models_py.__init__ to keep a single
# registration site consistent with all other new-loader models.

__all__ = [
    "DeepSeekVLV2ForConditionalGeneration",
    "DeepSeekVLV2ForCausalLM",
]
