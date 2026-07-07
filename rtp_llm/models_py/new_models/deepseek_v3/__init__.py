from .language import DeepSeekV32ForCausalLM

# DeepSeekV32ForCausalLM also serves as the new-loader implementation for
# deepseek2, deepseek3, deepseek_v31, and glm_5.  All four are architectural
# variants of the same MLA + MoE family and are handled by the same class
# via config-driven parameterisation in _extract_config_values.

__all__ = ["DeepSeekV32ForCausalLM"]
