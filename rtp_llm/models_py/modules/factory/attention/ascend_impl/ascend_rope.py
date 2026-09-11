import torch


def apply_rope_pos_ids_nhd(q, k, cos_sin_cache, pos_ids, is_neox_style=False):
    """
    Apply RoPE to query and key tensors using a precomputed cos/sin cache.

    Args:
        q: [tokens, num_heads, head_dim]
        k: [tokens, num_kv_heads, head_dim]
        cos_sin_cache: [max_pos, rope_dim] non-interleaved [cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}]
        pos_ids: [tokens] int32 position indices
        is_neox_style: interleave mode

    Returns:
        (q_out, k_out): RoPE-transformed tensors (in-place on input tensors)
    """
    rope_dim = cos_sin_cache.shape[-1]
    half_dim = rope_dim // 2

    embedding = cos_sin_cache[pos_ids]
    cos = embedding[:, :half_dim]
    sin = embedding[:, half_dim:]

    q_rope = q[..., :rope_dim]
    k_rope = k[..., :rope_dim]

    if is_neox_style:
        q_rope_2 = q_rope.reshape(*q_rope.shape[:-1], -1, 2)
        k_rope_2 = k_rope.reshape(*k_rope.shape[:-1], -1, 2)

        q_neg = q_rope_2[..., 1]
        q_pos = q_rope_2[..., 0]
        k_neg = k_rope_2[..., 1]
        k_pos = k_rope_2[..., 0]

        cos_exp = cos.unsqueeze(-2).unsqueeze(-1)
        sin_exp = sin.unsqueeze(-2).unsqueeze(-1)

        q_rot = torch.stack([
            q_pos * cos_exp.squeeze(-1) - q_neg * sin_exp.squeeze(-1),
            q_pos * sin_exp.squeeze(-1) + q_neg * cos_exp.squeeze(-1),
        ], dim=-1).flatten(start_dim=-2)
        k_rot = torch.stack([
            k_pos * cos_exp.squeeze(-1) - k_neg * sin_exp.squeeze(-1),
            k_pos * sin_exp.squeeze(-1) + k_neg * cos_exp.squeeze(-1),
        ], dim=-1).flatten(start_dim=-2)
    else:
        q1, q2 = q_rope[..., :half_dim], q_rope[..., half_dim:]
        k1, k2 = k_rope[..., :half_dim], k_rope[..., half_dim:]

        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    q[..., :rope_dim] = q_rot
    k[..., :rope_dim] = k_rot

    return q, k
