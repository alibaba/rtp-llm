"""Opt-in PPU MegaAttention with channelwise FP8 output feeding ``wo_a``.

The ABI follows FlashMLA-for-sail as consumed by SGLang 631e12a9. It fuses
Q RoPE, sparse attention, inverse O RoPE and whole-4096-channel output
quantization. ``wo_a`` remains a separate DeepGEMM per-channel GEMM. No
SGLang Python package or checkout is required at runtime.

Prepare before graph capture. The original projection weights are never
modified: a private Wq_b row permutation can emit packed Q directly, while
ordinary callers can pass their unrotated Q and let this adapter pack it.
The caller must skip its Q RoPE, O RoPE and wo_a when using either forward
entry, then pass the returned BF16 [rows, groups, rank] to wo_b. KV writers,
causal visibility, CP gathering and Verify journal staging remain owned by
the existing attention coordinator.

All eligibility checks are metadata-only. The caller still guarantees
in-range live token positions and physical cache slots, and keeps cache
writes/journal staging alive through the native call. Native execution
failures propagate; there is no retry after a partial device execution.
"""

from __future__ import annotations

import importlib
import inspect
import os
from functools import lru_cache
from types import CodeType, SimpleNamespace
from weakref import WeakValueDictionary

import torch
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import env_value as optimization_env_value


ENV = "RTP_V41_PPU_MEGA_ATTENTION"
LAUNCH_REUSE_ENV = "RTP_V41_PPU_DECODE_LAUNCH_REUSE"
HEAD_DIM = 512
ROPE_DIM = 64
GROUP_WIDTH = 4096
_PHASES = frozenset(("prefill", "decode", "verify"))
_COS_SIN_TABLES = WeakValueDictionary()


def enabled():
    value = optimization_env_value(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value == "1"


def _normal(tensor, dtype=None):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and (dtype is None or tensor.dtype == dtype)
        and tensor.is_cuda
        and not tensor.requires_grad
        and not tensor.is_conj()
        and not tensor.is_neg()
    )


def _key(tensor):
    # Captured models have immutable weights, including inference tensors
    # without version counters. Reloads require preparing a fresh adapter.
    version = None if torch.is_inference(tensor) else tensor._version
    return (
        id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.dtype, tensor.device, version,
    )


def _accepts_interleaved_flag(operation):
    parameters = inspect.signature(operation).parameters
    return "use_legacy" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )


def _cos_sin_table(freqs_cis):
    # Layers sharing the same RoPE table must not each retain another large
    # FP32 table. Weak values disappear with the last prepared model owner.
    key = _key(freqs_cis)
    table = _COS_SIN_TABLES.get(key)
    if table is None:
        table = torch.cat((freqs_cis.real, freqs_cis.imag), dim=-1).contiguous()
        _COS_SIN_TABLES[key] = table
    return table


def _readonly_prefill_binding(module, operation, *, name="prefill", entry_name="fused_norm_rope_attn_rope_cast_fwd"):
    # The installed/preflight-qualified FlashMLA wrapper passes positions to
    # a const int32_t* native input. Bind that actual chain once; a custom
    # adapter or a changed wrapper/native entry must keep independent casts.
    native = vars(module).get("_backend")
    entry = getattr(native, entry_name, None)
    if (
        module.__name__ != "flash_mla.fused_norm_rope_attn_rope_cast"
        or not inspect.isfunction(operation)
        or operation.__globals__ is not vars(module)
        or vars(module).get(name) is not operation
        or getattr(native, "__name__", None) != "flash_mla_cuda"
        or not inspect.isbuiltin(entry)
        or getattr(entry, "__module__", None) != "flash_mla_cuda"
        or getattr(entry, "__name__", None) != entry_name
    ):
        return None
    receipt = os.environ.get("RTP_V41_PPU_FLASHMLA_BUILD_RECEIPT")
    if not receipt:
        return None
    try:
        from importlib.metadata import distribution
        from pathlib import Path

        from rtp_llm.tools.v41.preflight import flashmla_build_identity

        # Reuse the existing pinned wrapper/binary authenticator once on cold
        # backend preparation, with no file hashing on a forward cache hit.
        flashmla_build_identity(distribution("flash-mla"), receipt)
        # Disk identity alone cannot certify a function whose live __code__
        # was replaced before preparation. Compile, without executing, the
        # authenticated wrapper and compare its original function body.
        compiled = compile(
            Path(module.__file__).read_bytes(), operation.__code__.co_filename,
            "exec", dont_inherit=True,
        )
        original = next((value for value in compiled.co_consts
                         if isinstance(value, CodeType) and value.co_name == name), None)
        if operation.__code__ != original:
            return None
    except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return module, operation, operation.__code__, native, entry


@lru_cache(maxsize=1)
def _backend():
    """Resolve the installed wheel before launching or changing Q semantics."""
    try:
        module = importlib.import_module("flash_mla.fused_norm_rope_attn_rope_cast")
        deep_gemm = importlib.import_module("deep_gemm")
    except ImportError:
        return None
    prefill = getattr(module, "prefill", None)
    decode = getattr(module, "decode", None)
    einsum = getattr(deep_gemm, "fp8_einsum", None)
    if not all(callable(op) for op in (prefill, decode, einsum)):
        return None
    from rtp_llm.models_py.kernels.ppu.runtime import install_deep_gemm_build_lock

    install_deep_gemm_build_lock()
    # PPU wheels expose a single token-interleaved kernel set and omit this
    # keyword. Newer dual-layout wheels require use_legacy=False for our pools.
    return SimpleNamespace(
        prefill=prefill, decode=decode, einsum=einsum,
        prefill_kwargs={"use_legacy": False} if _accepts_interleaved_flag(prefill) else {},
        decode_kwargs={"use_legacy": False} if _accepts_interleaved_flag(decode) else {},
        readonly_prefill=_readonly_prefill_binding(module, prefill),
        readonly_decode=_readonly_prefill_binding(
            module, decode, name="decode", entry_name="fused_norm_rope_attn_rope_cast_decode",
        ),
    )


def eligibility_reason(attn, freqs_cis):
    """Inspect resident weights/table only; no activation or cache is changed."""
    if not enabled():
        return "disabled"
    from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

    g = getattr(attn, "geometry", None)
    wq_b, wo_a = getattr(attn, "wq_b", None), getattr(attn, "wo_a", None)
    if g is None or type(wq_b) is not PPUPTPCLinear:
        return "query_projection_not_ptpc"
    if (
        g.n_heads not in (64, 128) or g.head_dim != HEAD_DIM
        or g.rope_head_dim != ROPE_DIM or g.o_groups * 8 != g.n_heads
        or g.o_lora_rank <= 0
    ):
        return "unsupported_geometry"
    qw, qs = wq_b.weight, wq_b.scale
    if not (
        _normal(qw, torch.float8_e4m3fn) and _normal(qs, torch.float32)
        and qw.shape == (g.n_heads * HEAD_DIM, g.q_lora_rank)
        and qs.shape == (qw.shape[0], 1) and qw.is_contiguous() and qs.is_contiguous()
        and qs.device == qw.device and callable(wq_b._gemm)
    ):
        return "unsupported_query_weight"
    if torch.version.hip is not None or "ZW" not in torch.cuda.get_device_name(qw.device):
        return "not_ppu"
    major, minor = torch.cuda.get_device_capability(qw.device)
    if major * 10 + minor < 89:
        return "requires_ppu_sm89"
    weight = getattr(wo_a, "weight", None)
    if not (
        _normal(weight) and weight.device == qw.device and weight.is_contiguous()
        and weight.shape == (g.o_groups * g.o_lora_rank, GROUP_WIDTH)
    ):
        return "unsupported_output_weight"
    if type(wo_a) is PPUPTPCLinear:
        scale = wo_a.scale
        if not (
            weight.dtype == torch.float8_e4m3fn and _normal(scale, torch.float32)
            and scale.device == qw.device and scale.is_contiguous()
            and scale.shape == (weight.shape[0], 1)
        ):
            return "unsupported_output_scale"
    elif weight.dtype != torch.bfloat16:
        return "unsupported_output_quantization"
    if not (
        _normal(freqs_cis, torch.complex64) and freqs_cis.device == qw.device
        and freqs_cis.ndim == 2 and freqs_cis.shape[1] == ROPE_DIM // 2
        and freqs_cis.shape[0] > 0 and freqs_cis.is_contiguous()
    ):
        return "unsupported_rope_table"
    sink = getattr(attn, "attn_sink", None)
    if not (
        _normal(sink, torch.float32) and sink.device == qw.device
        and sink.shape == (g.n_heads,) and sink.is_contiguous()
    ):
        return "unsupported_sink"
    return None


def prepare(attn, freqs_cis=None):
    """Return a prepared adapter, or None before activation/cache work.

    The opt-in changes attention/output quantization and therefore requires
    separate correctness and performance acceptance. It defaults to off.
    """
    if freqs_cis is None:
        freqs_cis = getattr(attn, "freqs_cis", None)
    if eligibility_reason(attn, freqs_cis) is not None:
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("PPU MegaAttention must be prepared before graph capture")
    backend = _backend()
    if backend is None:
        return None
    result = PPUMegaAttention(attn, freqs_cis, backend)
    binding = vars(backend).get("readonly_prefill") if type(backend) is SimpleNamespace else None
    if binding is not None:
        result._readonly_prefill = backend, binding
    binding = vars(backend).get("readonly_decode") if type(backend) is SimpleNamespace else None
    if binding is not None:
        result._readonly_decode = backend, binding
    return result


class PPUMegaAttention:
    def __init__(self, attn, freqs_cis, backend):
        from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
        from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import quantize_ptpc_activation

        self.owner, self.backend = attn, backend
        # Direct/custom adapter construction retains the independent-cast ABI.
        # Only prepare() can attach its authenticated resolver result.
        self._readonly_prefill = None
        self._readonly_decode = None
        self.heads, self.groups = attn.geometry.n_heads, attn.geometry.o_groups
        self.rank, self.q_rank = attn.geometry.o_lora_rank, attn.geometry.q_lora_rank
        self.device = attn.wq_b.weight.device
        self.softmax_scale = attn.geometry.sm_scale()
        self.freqs_cis = freqs_cis
        self.cos_sin_cache = _cos_sin_table(freqs_cis)
        self.sink = attn.attn_sink
        # A PTPC scale belongs to each output row, so its permutation is
        # identical to the weight's. Neither original owner is overwritten.
        qw, qs = attn.wq_b.weight, attn.wq_b.scale
        self.query_weight = qw.view(self.heads, 32, 16, self.q_rank).transpose(0, 1).reshape(qw.shape)
        self.query_scale = qs.view(self.heads, 32, 16, 1).transpose(0, 1).reshape(qs.shape)
        self.query_gemm = attn.wq_b._gemm
        self._sources = [qw, qs, attn.wo_a.weight, freqs_cis, self.sink]
        if type(attn.wo_a) is PPUPTPCLinear:
            weight, scale = attn.wo_a.weight, attn.wo_a.scale
            self._sources.append(scale)
        else:
            # This is the same per-output-channel conversion used by the PPU
            # reference. It occurs once, before capture, never per forward.
            weight, scale = quantize_ptpc_activation(attn.wo_a.weight)
        self.output_weight = weight.view(self.groups, self.rank, 8, 16, 32).transpose(-3, -2).reshape(
            self.groups, self.rank, GROUP_WIDTH
        )
        self.output_scale = scale.view(self.groups, self.rank, 1)
        self._source_keys = tuple(_key(value) for value in self._sources)

    def _current(self):
        tensors = [self.owner.wq_b.weight, self.owner.wq_b.scale, self.owner.wo_a.weight,
                   self.freqs_cis, self.owner.attn_sink]
        if len(self._sources) == 6:
            tensors.append(self.owner.wo_a.scale)
        return tuple(_key(value) for value in tensors) == self._source_keys

    def query_reason(self, q, positions, *, phase, valid_mask=None):
        """Check before selecting the fused branch or omitting ordinary RoPE."""
        if not enabled():
            return "disabled"
        if phase not in _PHASES:
            return "unsupported_phase"
        if not self._current():
            return "weights_changed"
        if torch.is_grad_enabled():
            return "requires_inference"
        if not (
            _normal(q, torch.bfloat16) and q.device == self.device
            and q.ndim == 3 and q.shape[1:] == (self.heads, HEAD_DIM)
            and q.is_contiguous()
        ):
            return "unsupported_query"
        if not (
            _normal(positions) and positions.device == self.device
            and positions.shape == (q.shape[0],)
            and positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
        ):
            return "unsupported_positions"
        if valid_mask is not None and not (
            _normal(valid_mask, torch.bool) and valid_mask.device == self.device
            and valid_mask.shape == positions.shape and valid_mask.is_contiguous()
        ):
            return "unsupported_valid_mask"
        return None

    def _indices_ok(self, indices, lengths, rows):
        return (
            _normal(indices, torch.int32) and indices.device == self.device
            and indices.ndim in (2, 3) and indices.shape[0] == rows
            and (indices.ndim == 2 or indices.shape[1] == 1)
            and indices.shape[-1] > 0 and indices.shape[-1] % 64 == 0
            and indices.is_contiguous()
            and _normal(lengths, torch.int32) and lengths.device == self.device
            and lengths.shape == (rows,) and lengths.is_contiguous()
        )

    def prefill_reason(self, q, kv, indices, positions, topk_lengths, *, valid_mask=None):
        reason = self.query_reason(q, positions, phase="prefill", valid_mask=valid_mask)
        if reason is not None:
            return reason
        if not (
            _normal(kv, torch.bfloat16) and kv.device == self.device
            and kv.ndim == 3 and kv.shape[1:] == (1, HEAD_DIM) and kv.is_contiguous()
        ):
            return "unsupported_prefill_kv"
        if not self._indices_ok(indices, topk_lengths, q.shape[0]):
            return "unsupported_prefill_indices"
        return None

    def _cache_ok(self, cache, widths):
        return (
            _normal(cache) and cache.device == self.device
            and cache.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn)
            and cache.ndim == 4 and cache.shape[1] > 0 and cache.shape[2] == 1
            and cache.shape[3] in widths and cache.stride(3) == 1
            and cache.stride(1) == cache.shape[3]
            and cache.stride(0) >= cache.shape[1] * cache.shape[3]
        )

    def decode_reason(self, q, plan, positions, *, phase="decode"):
        reason = self.query_reason(q, positions, phase=phase, valid_mask=plan.valid_mask)
        if reason is not None:
            return reason
        if not self._cache_ok(plan.swa_cache, (528,)):
            return "unsupported_swa_cache"
        if not self._indices_ok(plan.swa_indices, plan.swa_lengths, q.shape[0]):
            return "unsupported_swa_indices"
        if plan.global_cache is not None:
            if not self._cache_ok(plan.global_cache, (288, 528)):
                return "unsupported_global_cache"
            if not self._indices_ok(plan.global_indices, plan.global_lengths, q.shape[0]):
                return "unsupported_global_indices"
        elif plan.global_indices is not None or plan.global_lengths is not None:
            return "inconsistent_global_metadata"
        return None

    def project_q(self, qr, quantized_input=None):
        """Wq_b with private row permutation: returned Q is packed, unrotated."""
        if not enabled() or torch.is_grad_enabled():
            raise ValueError("PPU MegaAttention query projection requires opt-in inference")
        if not self._current():
            raise ValueError("PPU MegaAttention projection weights changed")
        if not (
            _normal(qr, torch.bfloat16) and qr.device == self.device
            and qr.ndim == 2 and qr.shape[1] == self.q_rank and qr.is_contiguous()
        ):
            raise ValueError("PPU MegaAttention needs contiguous BF16 query latents")
        pair = self.owner.wq_b.quantize_input(qr) if quantized_input is None else quantized_input
        payload, scale = pair
        if not (
            _normal(payload, torch.float8_e4m3fn) and payload.shape == qr.shape
            and payload.device == self.device and payload.is_contiguous()
            and _normal(scale, torch.float32) and scale.shape == (qr.shape[0], 1)
            and scale.device == self.device and scale.is_contiguous()
        ):
            raise ValueError("PPU MegaAttention received invalid PTPC query input")
        q = torch.empty((qr.shape[0], self.heads * HEAD_DIM), device=self.device, dtype=torch.bfloat16)
        if qr.shape[0]:
            self.query_gemm((payload, scale), (self.query_weight, self.query_scale), q)
        return q.view(-1, self.heads, HEAD_DIM)

    def _query(self, q, packed_q):
        if packed_q:
            return q
        return q.view(-1, self.heads, 32, 16).transpose(1, 2).contiguous().view_as(q)

    def unpack_query(self, q):
        """Restore ordinary head order before a pre-dispatch fallback."""
        return q.view(-1, 32, self.heads, 16).transpose(1, 2).contiguous().view_as(q)

    def _common(self, q, positions, valid_mask):
        # Padded Graph rows may carry -1; native RoPE requires an addressable
        # row even when the key prefix is empty. Live positions are caller-owned.
        if valid_mask is not None:
            positions = torch.where(valid_mask, positions, 0)
        return dict(
            q=q, token_positions=positions.to(torch.int32), cos_sin_cache=self.cos_sin_cache,
            enable_q_norm=False, rms_norm_eps=0.0, is_rope_neox_style=False, rope_dim=ROPE_DIM,
            n_wv_group=self.groups, num_per_channels=GROUP_WIDTH,
            use_tma_aligned_col_major_sf=False, round_sf=False, use_packed_ue8m0=False,
            sm_scale=self.softmax_scale, d_v=HEAD_DIM, attn_sink=self.sink,
        )

    def _project_output(self, data, scale, rows, valid_mask):
        if not (
            _normal(data, torch.float8_e4m3fn) and data.device == self.device
            and data.shape == (rows, self.groups, GROUP_WIDTH) and data.stride(-1) == 1
            and _normal(scale, torch.float32) and scale.device == self.device
            and scale.shape == (rows, self.groups, 1)
        ):
            raise RuntimeError("FlashMLA did not return PPU channelwise FP8 MegaAttention output")
        out = torch.empty((rows, self.groups, self.rank), device=self.device, dtype=torch.bfloat16)
        self.backend.einsum(
            "bhr,hdr->bhd", (data, scale), (self.output_weight, self.output_scale),
            out, recipe=(1, 1, GROUP_WIDTH),
        )
        if valid_mask is not None:
            if (
                optimization_env_value(LAUNCH_REUSE_ENV, "0") == "1"
                and rows in (1, 8, 24, 32) and _normal(out, torch.bfloat16) and out.is_contiguous()
                and valid_mask.dtype == torch.bool and valid_mask.is_contiguous()
                and valid_mask.device == out.device
            ):
                # out is a fresh private destination, never a caller-owned
                # view. Leave live bytes untouched (including signed zero)
                # and write +0 to idle rows, even if their GEMM produced NaN.
                _mask_owned_output(out, valid_mask)
            else:
                out = torch.where(valid_mask[:, None, None], out, 0)
        return out

    def prefill(self, q, kv, indices, positions, topk_lengths, *, packed_q=False, valid_mask=None):
        reason = self.prefill_reason(q, kv, indices, positions, topk_lengths, valid_mask=valid_mask)
        if reason is not None:
            raise ValueError("PPU MegaAttention prefill: " + reason)
        if q.shape[0] == 0:
            return q.new_empty((0, self.groups, self.rank))
        common = self._common(self._query(q, packed_q), positions, valid_mask)
        data, scale, _, _ = self.backend.prefill(
            **common, kv=kv, indices=indices.view(q.shape[0], 1, -1),
            topk_length=topk_lengths, **self.backend.prefill_kwargs,
        )
        return self._project_output(data, scale, q.shape[0], valid_mask)

    def decode(self, q, plan, positions, *, packed_q=False, mask_output=True, phase="decode", shared=None):
        """Consume the same interleaved-pool plan as ordinary packed_decode.

        A Verify caller must enter its existing journal staging context before
        this call. An explicit phase retains that distinction in admission.
        """
        reason = self.decode_reason(q, plan, positions, phase=phase)
        if reason is not None:
            raise ValueError("PPU MegaAttention paged decode: " + reason)
        if q.shape[0] == 0:
            return q.new_empty((0, self.groups, self.rank))
        if (shared is not None and plan.valid_mask is not None
                and optimization_env_value(LAUNCH_REUSE_ENV, "0") == "1"
                and readonly_decode_positions(self)):
            # The dictionary belongs to this one capture/forward. Retain
            # sources as well as the result, so object IDs cannot be reused.
            key = ("mega_decode_positions", id(positions), id(plan.valid_mask))
            if key not in shared:
                converted = torch.where(plan.valid_mask, positions, 0).to(torch.int32)
                shared[key] = positions, plan.valid_mask, converted
            source, valid, converted = shared[key]
            if source is not positions or valid is not plan.valid_mask:
                raise RuntimeError("MegaAttention position reuse belongs to another replay")
            common = self._common(self._query(q, packed_q), converted, None)
        else:
            common = self._common(self._query(q, packed_q), positions, plan.valid_mask)
        extra = plan.global_indices
        data, scale, _ = self.backend.decode(
            **common, k_cache=plan.swa_cache,
            indices_in_kvcache=plan.swa_indices.view(q.shape[0], -1),
            topk_length=plan.swa_lengths, extra_k_cache=plan.global_cache,
            extra_indices_in_kvcache=None if extra is None else extra.view(q.shape[0], -1),
            extra_topk_length=plan.global_lengths, **self.backend.decode_kwargs,
        )
        return self._project_output(data, scale, q.shape[0], plan.valid_mask if mask_output else None)


_READONLY_PREFILL_METHODS = tuple(
    (name, vars(PPUMegaAttention)[name])
    for name in ("prefill", "prefill_reason", "query_reason", "_common",
                 "_query", "_current", "_indices_ok")
)
_READONLY_PREFILL_NORMAL = _normal


@lru_cache(maxsize=1)
def _owned_output_mask_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask_owned_output(output, valid, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        if not tl.load(valid + row):
            column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            tl.store(output + row * WIDTH + column, 0, column < WIDTH)

    return triton, mask_owned_output


def _mask_owned_output(output, valid):
    triton, kernel = _owned_output_mask_kernel()
    width = output.numel() // output.shape[0]
    kernel[(output.shape[0], triton.cdiv(width, 1024))](output, valid, width, 1024, num_warps=4)
    return output


_READONLY_DECODE_METHODS = tuple(
    (name, vars(PPUMegaAttention)[name])
    for name in ("decode", "decode_reason", "query_reason", "_common", "_query",
                 "_current", "_indices_ok", "_cache_ok")
)


def readonly_decode_positions(mega):
    """Only the authenticated original const-position consumer may reuse."""
    if type(mega) is not PPUMegaAttention or _normal is not _READONLY_PREFILL_NORMAL:
        return False
    if any(name in vars(mega) or vars(PPUMegaAttention).get(name) is not original
           for name, original in _READONLY_DECODE_METHODS):
        return False
    prepared = getattr(mega, "_readonly_decode", None)
    if prepared is None:
        return False
    backend = mega.backend
    owner, binding = prepared
    if backend is not owner or getattr(backend, "readonly_decode", None) is not binding:
        return False
    module, operation, code, native, entry = binding
    return (backend.decode is operation and vars(module).get("decode") is operation
            and operation.__code__ is code and operation.__globals__.get("_backend") is native
            and getattr(native, "fused_norm_rope_attn_rope_cast_decode", None) is entry)


def readonly_prefill_positions(mega):
    """Only the original prepared consumer may share a forward's cast."""
    if type(mega) is not PPUMegaAttention or _normal is not _READONLY_PREFILL_NORMAL:
        return False
    if any(
        name in vars(mega) or vars(PPUMegaAttention).get(name) is not original
        for name, original in _READONLY_PREFILL_METHODS
    ):
        return False
    backend = mega.backend
    prepared = getattr(mega, "_readonly_prefill", None)
    if prepared is None:
        return False
    owner, binding = prepared
    if backend is not owner or getattr(backend, "readonly_prefill", None) is not binding:
        return False
    module, operation, code, native, entry = binding
    return (
        backend.prefill is operation
        and vars(module).get("prefill") is operation
        and operation.__code__ is code
        and operation.__globals__.get("_backend") is native
        and getattr(native, "fused_norm_rope_attn_rope_cast_fwd", None) is entry
    )
