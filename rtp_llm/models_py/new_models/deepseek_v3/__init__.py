from .language import DeepSeekV32ForCausalLM

# DeepSeekV32ForCausalLM also serves as the new-loader implementation for
# deepseek2, deepseek3, deepseek_v31, deepseek_v32, glm_5, glm4_moe_lite, and
# kimi_k2. These are architectural variants of the same MLA + MoE family and
# are handled by the same class via config-driven parameterisation in
# extract_config_values.

__all__ = ["DeepSeekV32ForCausalLM"]
