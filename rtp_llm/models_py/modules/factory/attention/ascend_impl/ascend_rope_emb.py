import torch

try:
    import torch_npu
except ImportError:  # allow import on non-Ascend hosts (e.g. CPU-only CI tests)
    torch_npu = None

from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_rope import apply_rope_pos_ids_nhd
from rtp_llm.ops import AttentionConfigs, RopeStyle, check_rope_cache, get_rope_cache_once


def _npu_available() -> bool:
    return torch_npu is not None and torch.npu.is_available()


_HALVES_CACHE = {}


def _gen_base_halves_cache(rope_dim: int, rope_theta: float, max_pos: int) -> torch.Tensor:
    """Python-side halves [cos|sin] cache, mirroring C++ genBaseCache
    (non-interleaved). MRoPE reuses the base rotary formula: the three-axis
    positions index into this cache per frequency slot.

    Cached per (dim, theta, max_pos, device): the attention impl is rebuilt
    per batch on the eager path and once per capture bucket under aclgraph —
    rebuilding the table each time would waste both startup and steady-state
    latency, so every construction shares one resident tensor."""
    device = torch.device("npu") if _npu_available() else torch.device("cpu")
    key = (rope_dim, rope_theta, max_pos, str(device))
    cached = _HALVES_CACHE.get(key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()
    _HALVES_CACHE[key] = cache
    return cache


def mrope_interleaved_axis_of_freq(mrope_section, rotary_pairs: int, device) -> torch.Tensor:
    """Axis (0=T/1=H/2=W) of each rotary frequency slot under the repo-wide
    interleaved-MRoPE contract — a literal mirror of the kernel's
    get_mrope_position_dim: slot tidx is H when tidx%%3==1 and tidx<3*dim2, W
    when tidx%%3==2 and tidx<3*dim3, otherwise T. Sections must sum to
    rotary_pairs and fit the strided slots (same as the ROCm host validation).

    Built and validated on CPU (no device sync), then moved to ``device`` —
    this function runs inside the aclgraph capture region where host
    readbacks of device tensors are not allowed."""
    if (len(mrope_section) != 3
            or any(int(s) < 0 for s in mrope_section)
            or sum(int(s) for s in mrope_section) != rotary_pairs):
        raise RuntimeError(
            f"interleaved MRoPE sections {mrope_section} must have 3 entries "
            f"summing to rotary_pairs={rotary_pairs}")
    tidx = torch.arange(rotary_pairs)
    axes = torch.zeros(rotary_pairs, dtype=torch.long)
    axes[(tidx % 3 == 1) & (tidx < 3 * int(mrope_section[1]))] = 1
    axes[(tidx % 3 == 2) & (tidx < 3 * int(mrope_section[2]))] = 2
    if int((axes == 1).sum()) != int(mrope_section[1]) or int((axes == 2).sum()) != int(mrope_section[2]):
        raise RuntimeError(
            f"interleaved MRoPE sections {mrope_section} exceed the THW strided "
            f"slot capacity of rotary_pairs={rotary_pairs}")
    return axes.to(device)


def apply_mrope_interleaved(query, key, positions, mrope_section, cos_sin_cache, axes=None):
    """Interleaved MRoPE (mrope_interleaved=true, Qwen3-VL/Qwen3.5) following
    the repo-wide contract: THW slot-cycling axis assignment (see
    mrope_interleaved_axis_of_freq) and lo/hi half-split pairing within
    rope_dim, identical to the CUDA/ROCm kernels and the reference test in
    rocm_impl/test/test_aiter_prefill_op.py. cos_sin_cache is the halves
    [cos|sin] cache of total width rope_dim; dims beyond rope_dim pass through.
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    if axes is None:
        axes = mrope_interleaved_axis_of_freq(mrope_section, half, positions.device)
    pos = positions.long()[:, axes]                      # [T, half]
    freq_idx = torch.arange(half, device=pos.device)
    cos = cos_sin_cache[pos, freq_idx].unsqueeze(1)      # [T, 1, half]
    sin = cos_sin_cache[pos, half + freq_idx].unsqueeze(1)

    def rot(x):
        lo = x[..., :half]
        hi = x[..., half:rope_dim]
        out = torch.empty_like(x)
        out[..., :half] = lo * cos - hi * sin
        out[..., half:rope_dim] = hi * cos + lo * sin
        out[..., rope_dim:] = x[..., rope_dim:]
        return out

    query.copy_(rot(query))
    key.copy_(rot(key))


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
        # Per-device cached axis map: built on first (eager) use, reused
        # during aclgraph capture/replay so no host->device copy happens
        # inside the capture region.
        self._mrope_axes_cache = {}

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
            if torch_npu is None:
                raise RuntimeError("non-interleaved MRoPE requires torch_npu")
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

        # Interleaved MRoPE (mrope_interleaved=true): repo-wide contract —
        # THW slot-cycling axis assignment + lo/hi half-split pairing, same
        # math as the CUDA/ROCm kernels (see apply_mrope_interleaved).
        axes = self._mrope_axes_cache.get(query.device)
        if axes is None:
            axes = mrope_interleaved_axis_of_freq(
                self.mrope_section, self.cos_sin_cache.shape[-1] // 2, query.device)
            self._mrope_axes_cache[query.device] = axes
        apply_mrope_interleaved(query, key, positions, self.mrope_section, self.cos_sin_cache, axes=axes)

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
