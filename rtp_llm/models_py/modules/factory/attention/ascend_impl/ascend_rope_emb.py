import torch
import torch_npu

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope import apply_rope_pos_ids_nhd
from rtp_llm.ops import AttentionConfigs, RopeStyle, check_rope_cache, get_rope_cache_once


def _gen_base_halves_cache(rope_dim: int, rope_theta: float, max_pos: int) -> torch.Tensor:
    """Python-side halves [cos|sin] cache, mirroring C++ genBaseCache
    (non-interleaved). MRoPE reuses the base rotary formula: the three-axis
    positions index into this cache per frequency section."""
    device = torch.device("npu") if torch.npu.is_available() else torch.device("cpu")
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


class AscendRotaryEmbeddingOp:
    """Ascend RoPE using pure PyTorch implementation (replaces flashinfer.rope)."""

    def __init__(self, attn_config: AttentionConfigs, cos_sin_cache: torch.Tensor | None = None):
        self.head_size = attn_config.size_per_head
        self.token_per_block = attn_config.kernel_tokens_per_block
        self.rope_config = attn_config.rope_config
        # rope_config.is_neox_style=True means NeoX half-split (repo default),
        # False means GPT-J interleaved. The helper's flag has inverted naming
        # (True = interleaved) and always consumes a halves [cos|sin] cache.
        rope_neox = bool(self.rope_config.is_neox_style) if self.rope_config is not None else True
        self.is_interleaved = not rope_neox
        self.rope_style = self.rope_config.style if self.rope_config is not None else None
        self.mrope_section = None
        self.mrope_interleaved = False
        if self.rope_style == RopeStyle.Mrope:
            # Pair-unit T/H/W frequency sections (Qwen style, e.g. [16,24,24]);
            # sum == rotary_dim/2. Interleaved MRoPE (Qwen3.5) takes the
            # pure-torch path below until npu_mrope's interleave mode is
            # validated against the reference.
            self.mrope_section = [
                int(self.rope_config.mrope_dim1),
                int(self.rope_config.mrope_dim2),
                int(self.rope_config.mrope_dim3),
            ]
            self.mrope_interleaved = bool(
                getattr(self.rope_config, "mrope_interleaved", False)
            )
        self.cos_sin_cache = cos_sin_cache
        if self.cos_sin_cache is None and self.rope_config is not None:
            if self.rope_style == RopeStyle.Mrope:
                # C++ RopeCache only generates Base/Yarn; MRoPE builds the
                # same halves cache in Python with the base rotary formula
                # (three-axis positions index into it per frequency section).
                self.cos_sin_cache = _gen_base_halves_cache(
                    int(self.rope_config.dim),
                    float(self.rope_config.base),
                    int(attn_config.max_seq_len),
                )
            else:
                rope_cache = get_rope_cache_once(
                    self.rope_config,
                    max_position_embeddings=attn_config.max_seq_len,
                    is_cuda=True,  # Ascend can also set to True
                    interleave=False,
                )
                if not check_rope_cache(self.rope_config, rope_cache):
                    # Fail at construction with the real cause instead of a
                    # confusing "requires cos_sin_cache" on every forward.
                    raise RuntimeError(
                        f"Ascend RoPE has no cache for rope style "
                        f"{self.rope_config.style} (only Base/Yarn generate a "
                        f"cache in C++ RopeCache); use a Base/Yarn config on Ascend"
                    )
                self.cos_sin_cache = rope_cache.data
        self.num_heads = attn_config.head_num
        self.num_kv_heads = attn_config.kv_head_num
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, qkv):
        qkv = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(qkv, [
            self.head_size * self.num_heads,
            self.head_size * self.num_kv_heads,
            self.head_size * self.num_kv_heads,
        ], dim=-1)
        query = q.reshape(q.shape[0], self.num_heads, self.head_size)
        key = k.reshape(k.shape[0], self.num_kv_heads, self.head_size).contiguous()
        value = v.reshape(v.shape[0], self.num_kv_heads, self.head_size).contiguous()

        self._apply_rope(query, key, self.params)

        return query, key, value

    def _apply_rope(self, query, key, rope_params):
        if self.cos_sin_cache is None:
            raise RuntimeError("AscendRotaryEmbeddingOp requires cos_sin_cache")
        if self.mrope_section is not None:
            self._apply_mrope(query, key, rope_params)
            return
        device = rope_params.positions_d.device
        if self.cos_sin_cache.device != device:
            self.cos_sin_cache = self.cos_sin_cache.to(device, non_blocking=True)
        apply_rope_pos_ids_nhd(
            query, key,
            self.cos_sin_cache,
            rope_params.positions_d,
            is_neox_style=self.is_interleaved,
        )

    def _apply_mrope(self, query, key, rope_params):
        positions = rope_params.mrope_positions_d
        if positions is None or positions.dim() != 2:
            raise RuntimeError(
                f"MRoPE expects [num_tokens, 3] t/h/w positions, got "
                f"{None if positions is None else tuple(positions.shape)}; "
                "combo_position_ids must be produced upstream (PositionIdsGenerator MROPE style)"
            )
        device = query.device
        if self.cos_sin_cache.device != device:
            self.cos_sin_cache = self.cos_sin_cache.to(device, non_blocking=True)

        # Non-interleaved MRoPE: torch_npu.npu_mrope fast path.
        if not self.mrope_interleaved:
            pos_t = positions.t().to(torch.int64).contiguous()          # (3, T)
            q2 = query.reshape(query.shape[0], -1)
            k2 = key.reshape(key.shape[0], -1)
            q_out, k_out = torch_npu.npu_mrope(
                pos_t, q2, k2, self.cos_sin_cache,
                self.head_size,
                mrope_section=self.mrope_section,
                rotary_mode="half",
                cache_mode="default",
            )
            query.copy_(q_out.view_as(query))
            key.copy_(k_out.view_as(key))
            return

        # Interleaved MRoPE (Qwen3.5): pure-torch reference path. Global
        # adjacent-pair (2f, 2f+1) rotation; frequency f picks its axis
        # section's position. Only the first rope_dim dims are rotated
        # (partial_rotary_factor < 1 supported).
        cache = self.cos_sin_cache
        rope_dim = cache.shape[-1]
        half = rope_dim // 2
        t_end = self.mrope_section[0]
        h_end = t_end + self.mrope_section[1]
        axis_of_freq = torch.zeros(half, dtype=torch.long, device=positions.device)
        axis_of_freq[t_end:h_end] = 1
        axis_of_freq[h_end:] = 2
        pos = positions.long()[:, axis_of_freq]          # [T, half]
        # gather per-frequency cos/sin: result[t, f] = cache[pos[t, f], f].
        # Both indices must be advanced (a plain slice would hoist the
        # advanced dims and change the result shape).
        freq_idx = torch.arange(half, device=pos.device)
        cos = cache[pos, freq_idx].unsqueeze(1)          # [T, 1, half]
        sin = cache[pos, half + freq_idx].unsqueeze(1)   # [T, 1, half]

        def rot(x):
            r = x[..., :rope_dim]
            x1 = r[..., 0::2]
            x2 = r[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos
            out = x.clone()
            out[..., 0:rope_dim:2] = o1
            out[..., 1:rope_dim:2] = o2
            return out

        query.copy_(rot(query))
        key.copy_(rot(key))

    def _prepare_warmup_cache_indices(self, num_tokens, device):
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)

        max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block
        kv_page_indices = positions // self.token_per_block
        kv_page_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device=device)

        last_page_len = num_tokens % self.token_per_block
        if last_page_len == 0:
            last_page_len = self.token_per_block
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

        return batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages
