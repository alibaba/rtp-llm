import torch


def checkpoint(*, qk_norm=True, correction=True):
    weights = {}

    def put(name, shape):
        weights[name] = torch.arange(
            torch.tensor(shape).prod().item(), dtype=torch.float32
        ).reshape(shape) / 10 + len(weights)

    put("model.embed_tokens.weight", (8, 4))
    put("model.norm.weight", (4,))
    put("lm_head.weight", (8, 4))
    for layer in range(2):
        base = f"model.layers.{layer}."
        put(base + "input_layernorm.weight", (4,))
        put(base + "post_attention_layernorm.weight", (4,))
        for role, size in (("q", 4), ("k", 2), ("v", 2)):
            put(base + f"self_attn.{role}_proj.weight", (size, 4))
            put(base + f"self_attn.{role}_proj.bias", (size,))
        put(base + "self_attn.o_proj.weight", (4, 4))
        if qk_norm:
            put(base + "self_attn.q_norm.weight", (2,))
            put(base + "self_attn.k_norm.weight", (2,))
        if layer == 0:
            ffns = [("mlp", 8)]
        else:
            put(base + "mlp.gate.weight", (4, 4))
            if correction:
                put(base + "mlp.gate.e_score_correction_bias", (4,))
            ffns = [("mlp.shared_experts", 4)] + [
                (f"mlp.experts.{i}", 4) for i in range(4)
            ]
        for prefix, width in ffns:
            put(base + prefix + ".gate_proj.weight", (width, 4))
            put(base + prefix + ".up_proj.weight", (width, 4))
            put(base + prefix + ".down_proj.weight", (4, width))
    return weights
