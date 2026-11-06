"""Pure-Python validation of the pinned, text-only Qwen3.8 checkpoint schema."""
from copy import deepcopy


def validate_flat_qwen38_config(source):
    config = deepcopy(source)
    if config.get("model_type") != "qwen3_5_moe_text" or "text_config" in config:
        raise ValueError("Qwen3.8 requires the flat qwen3_5_moe_text checkpoint schema")
    if config.get("dtype") != "bfloat16" or "quantization_config" in config:
        raise ValueError("This Qwen3.8 path requires unquantized BF16 checkpoint inputs")
    if config.get("architectures") != ["Qwen3_5MoeForCausalLM"]:
        raise ValueError("Unexpected Qwen3.8 text architecture")
    rope = config.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default" or "mrope_section" in rope:
        raise ValueError("Qwen3.8 text supports ordinary default partial RoPE here")
    factor = rope.get("partial_rotary_factor", config.get("partial_rotary_factor"))
    if not isinstance(factor, (float, int)) or not 0 < factor <= 1:
        raise ValueError("Invalid partial rotary factor")
    rotary_dim = int(config["head_dim"] * factor)
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("Partial rotary dimensions must be positive and even")
    if not config.get("attn_output_gate", False):
        raise ValueError("The existing native Qwen weight descriptor requires the Q output gate")
    if config.get("hidden_act") != "silu":
        raise ValueError("Unsupported routed/shared-expert activation")
    layers = config["num_hidden_layers"]
    interval = config["full_attention_interval"]
    expected = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(layers)]
    if config.get("layer_types") != expected:
        raise ValueError("The checkpoint layer types differ from the native hybrid interval")
    config["partial_rotary_factor"] = factor
    config["rope_parameters"]["partial_rotary_factor"] = factor
    return config
