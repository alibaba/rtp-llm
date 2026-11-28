from rtp_llm.models_py.new_models.glm4_moe.language import GqaMoeForCausalLM
from rtp_llm.models_py.new_models.qwen3_moe.parameters import (
    install_qwen3_moe_parameters,
)


class Qwen3MoeForCausalLM(GqaMoeForCausalLM):
    _install_parameters = staticmethod(install_qwen3_moe_parameters)
