"""Independent native sparse-attention units for the shared-input benchmark.

Boundary: common pre-RoPE BF16 Q/K, Wkc and logical indices -> attn_out512.
No Wvc/Wo, no cache transcoding and no imports from earlier benchmark helpers.
RTP uses its production writer/copy/index kernels. TRT follows SGLang
v0.5.15.post1's FlashInfer RoPE/FP8 call and small-batch byte-store/token-map
algorithms. History creation is outside the measured forward boundary.
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

PAGE = 64
Q_NOPE = 192
LATENT = 512
ROPE = 64
ABSORBED = LATENT + ROPE
TOPK = 2048
ATTENTION_SCALE = (Q_NOPE + ROPE) ** -0.5
FP8 = torch.float8_e4m3fn


@lru_cache(maxsize=None)
def production_module(relative_name):
    """Load just the requested kernel source, avoiding service registrations."""
    for root in Path(__file__).resolve().parents:
        source = root / "rtp_llm" / "models_py" / "triton_kernels" / relative_name
        if source.is_file():
            key = "_glm53_attention_unit_" + source.stem
            spec = importlib.util.spec_from_file_location(key, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Production kernel source not found: {relative_name}")


@triton.jit
def _copy_native_trt_token_bytes(
    dst,
    src_nope,
    src_rope,
    token_slots,
    stride_nope: tl.constexpr,
    stride_rope: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    # SGLang mem_cache/triton_ops/mla_buffer.py small-batch fallback, DCP=1.
    # Inputs are uint8 views of already-quantized FP8, so this is bitwise copy.
    row = tl.program_id(0).to(tl.int64)
    byte = tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    slot = tl.load(token_slots + row).to(tl.int64)
    a = tl.load(src_nope + row * stride_nope + byte, byte < 512, other=0)
    b = tl.load(
        src_rope + row * stride_rope + byte - 512,
        (byte >= 512) & (byte < 576),
        other=0,
    )
    value = tl.where(byte < 512, a, b)
    # Extra safety for an explicitly invalid slot. Normal benchmark inputs use
    # only allocated positive pages, exactly as the native path does.
    tl.store(
        dst + tl.maximum(slot, 0) * 576 + byte,
        value,
        (byte < 576) & (slot >= 0),
    )
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _lookup_native_trt_tokens(
    token_table,
    logical_topk,
    physical_topk,
    row_stride: tl.constexpr,
    K: tl.constexpr,
):
    # SGLang attention/dsa/transform_index.py decode fast path. One token-map
    # row per query, containing physical token IDs (not physical page IDs).
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, K)
    logical = tl.load(logical_topk + row * K + col)
    physical = tl.load(
        token_table + row * row_stride + logical,
        logical >= 0,
        other=-1,
    )
    tl.store(physical_topk + row * K + col, physical)


class _ConsumableInputs:
    def __init__(self, common):
        self.common = common
        self.q_work = torch.empty_like(common.q)
        self.krope_work = torch.empty_like(common.k_rope)

    def stage(self):
        """Supply equal consumable BF16 inputs; caller places this before timing."""
        self.q_work.copy_(self.common.q)
        self.krope_work.copy_(self.common.k_rope)

    def assert_staged_equal(self):
        # This explicit validation is outside forward/timing. Byte comparison
        # catches signed zero and all exact bit patterns, unlike numeric close.
        for actual, original in (
            (self.q_work, self.common.q),
            (self.krope_work, self.common.k_rope),
        ):
            if not torch.equal(
                actual.contiguous().view(torch.uint8),
                original.contiguous().view(torch.uint8),
            ):
                raise AssertionError("Staged input bytes differ from common BF16 input")


class FlashUnit(_ConsumableInputs):
    """RTP first-layer complete unit, including per-step scheduler rebuilding."""

    def __init__(self, common):
        super().__init__(common)
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        self.flash = flash_mla_with_kvcache
        self.metadata, _ = get_mla_metadata()
        self.write_cache = production_module(
            "sparse_mla/fused_qk_rope_cat_cache_mla.py"
        ).fused_qk_rope_cat_cache_mla
        self.copy_rope = production_module(
            "common/strided_slice_copy.py"
        ).strided_slice_copy_
        self.map_tokens = production_module(
            "sparse_mla/block_index_to_global.py"
        ).triton_convert_req_index_to_global_index
        self.cache = torch.empty(
            (common.pages, PAGE, 656), dtype=torch.uint8, device=common.device
        )
        self.cache[0].zero_()
        self.q_absorbed = torch.empty(
            (common.tokens, common.heads, ABSORBED),
            dtype=torch.bfloat16,
            device=common.device,
        )
        # Native FlashMLA cannot take H8/H16. Their explicit padding adapter is
        # inside the unit timing, not presented as a native low-head kernel.
        self.kernel_heads = 64
        self.q_kernel = (
            self.q_absorbed
            if common.heads == self.kernel_heads
            else torch.empty(
                (common.tokens, self.kernel_heads, ABSORBED),
                dtype=torch.bfloat16,
                device=common.device,
            )
        )

    def append_history(self, k, krope, positions, slots):
        # The production writer rotates its q/k_rope arguments in place. Never
        # consume the source chunk shared with the other backend or golden.
        disposable_q = torch.zeros(
            (k.shape[0], 1, Q_NOPE + ROPE),
            dtype=torch.bfloat16,
            device=k.device,
        )
        self.write_cache(
            disposable_q,
            k,
            krope.clone(),
            self.cache,
            slots,
            positions,
            self.common.cos_sin,
            LATENT,
            ROPE,
            True,
            "fp8_ds_mla",
        )

    def forward(self):
        c = self.common
        self.write_cache(
            self.q_work,
            c.k,
            self.krope_work,
            self.cache,
            c.slots,
            c.positions,
            c.cos_sin,
            LATENT,
            ROPE,
            True,
            "fp8_ds_mla",
        )
        self.copy_rope(self.q_absorbed, self.q_work[..., Q_NOPE:], LATENT)
        torch.bmm(
            self.q_work[..., :Q_NOPE].transpose(0, 1),
            c.wkc,
            out=self.q_absorbed[..., :LATENT].transpose(0, 1),
        )
        if c.heads != self.kernel_heads:
            self.q_kernel.zero_()
            self.q_kernel[:, : c.heads].copy_(self.q_absorbed)
        indices = self.map_tokens(
            c.request_ids, c.page_table, c.logical_topk, PAGE, PAGE, TOPK
        )
        # Match SparseMlaFp8Op._forward_with_kvcache(layer_id=0). During CUDA
        # Graph capture this records scheduler kernels, which rerun on replay.
        self.metadata.tile_scheduler_metadata = None
        self.metadata.num_splits = None
        output, _ = self.flash(
            q=self.q_kernel.view(1, c.tokens, self.kernel_heads, ABSORBED),
            k_cache=self.cache.unsqueeze(2),
            block_table=c.page_table,
            cache_seqlens=None,
            head_dim_v=LATENT,
            tile_scheduler_metadata=self.metadata,
            num_splits=None,
            is_fp8_kvcache=True,
            indices=indices.view(1, c.tokens, TOPK),
            softmax_scale=ATTENTION_SCALE,
        )
        return output.view(c.tokens, self.kernel_heads, LATENT)[:, : c.heads]


class TrtUnit(_ConsumableInputs):
    """SGLang native FP8 unit; Q and KV scale factors are both exactly one."""

    def __init__(self, common):
        super().__init__(common)
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        from flashinfer.rope import mla_rope_quantize_fp8

        if common.tokens >= 768:
            raise ValueError(
                "This unit reproduces SGLang's small-batch Triton cache store; "
                "T >= 768 requires its distinct TMA store path"
            )
        self.decode = trtllm_batch_decode_with_kv_cache_mla
        self.rope_quantize = mla_rope_quantize_fp8
        self.pdl = torch.cuda.get_device_capability(common.device)[0] >= 9
        self.cache = torch.empty(
            (common.pages, 1, PAGE, ABSORBED), dtype=FP8, device=common.device
        )
        self.cache[0].view(torch.uint8).zero_()
        self.workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=common.device)
        self.q_nope = torch.empty(
            (common.tokens, common.heads, LATENT),
            dtype=torch.bfloat16,
            device=common.device,
        )
        self.q_fp8 = torch.empty(
            (common.tokens, common.heads, ABSORBED), dtype=FP8, device=common.device
        )
        self.k_fp8 = torch.empty(
            (common.tokens, LATENT), dtype=FP8, device=common.device
        )
        self.rope_fp8 = torch.empty(
            (common.tokens, ROPE), dtype=FP8, device=common.device
        )
        logical_tokens = torch.arange(
            common.context, dtype=torch.int64, device=common.device
        )
        # In SGLang this expanded token table is runtime metadata, ready before
        # the attention unit. The per-forward sparse lookup remains timed.
        per_request = (
            common.page_table[:, logical_tokens // PAGE] * PAGE
            + (logical_tokens % PAGE).int()
        )
        self.token_table = per_request.repeat_interleave(common.queries, dim=0)
        self.physical_indices = torch.empty_like(common.logical_topk)

    def _quantize_and_rotate(
        self, q_nope, q_rope, k, krope, positions, q_out, k_out, r_out
    ):
        self.rope_quantize(
            q_rope=q_rope,
            k_rope=krope,
            q_nope=q_nope,
            k_nope=k,
            cos_sin_cache=self.common.cos_sin,
            pos_ids=positions,
            is_neox=True,
            quantize_dtype=FP8,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            q_nope_out=q_out[..., :LATENT],
            q_rope_out=q_out[..., LATENT:],
            k_nope_out=k_out,
            k_rope_out=r_out,
            enable_pdl=self.pdl,
        )

    def _store(self, k, rope, slots):
        _copy_native_trt_token_bytes[(slots.numel(),)](
            self.cache.view(torch.uint8),
            k.view(torch.uint8),
            rope.view(torch.uint8),
            slots,
            k.stride(0),
            rope.stride(0),
            BLOCK=1024,
            PDL=self.pdl,
            launch_pdl=self.pdl,
        )

    def append_history(self, k, krope, positions, slots):
        # FlashInfer writes separate FP8 outputs; BF16 history sources remain
        # unchanged. Dummy Q exists solely to use the real fused writer API.
        count = k.shape[0]
        q_nope = torch.zeros((count, 1, LATENT), dtype=torch.bfloat16, device=k.device)
        q_rope = torch.zeros((count, 1, ROPE), dtype=torch.bfloat16, device=k.device)
        q_out = torch.empty((count, 1, ABSORBED), dtype=FP8, device=k.device)
        k_out = torch.empty_like(k, dtype=FP8)
        rope_out = torch.empty_like(krope, dtype=FP8)
        self._quantize_and_rotate(
            q_nope, q_rope, k, krope, positions, q_out, k_out, rope_out
        )
        # Chunked history setup need only match storage semantics, not TMA
        # prefill performance; it is excluded from every timed unit forward.
        self._store(k_out, rope_out, slots)

    def forward(self):
        c = self.common
        torch.bmm(
            self.q_work[..., :Q_NOPE].transpose(0, 1),
            c.wkc,
            out=self.q_nope.transpose(0, 1),
        )
        self._quantize_and_rotate(
            self.q_nope,
            self.q_work[..., Q_NOPE:],
            c.k,
            self.krope_work,
            c.positions,
            self.q_fp8,
            self.k_fp8,
            self.rope_fp8,
        )
        self._store(self.k_fp8, self.rope_fp8, c.slots)
        _lookup_native_trt_tokens[(c.tokens,)](
            self.token_table,
            c.logical_topk,
            self.physical_indices,
            self.token_table.stride(0),
            K=TOPK,
        )
        result = self.decode(
            query=self.q_fp8.view(c.tokens, 1, c.heads, ABSORBED),
            kv_cache=self.cache,
            workspace_buffer=self.workspace,
            qk_nope_head_dim=Q_NOPE,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=self.physical_indices.view(c.tokens, 1, TOPK),
            seq_lens=c.seq_lens,
            max_seq_len=c.context,
            sparse_mla_top_k=TOPK,
            bmm1_scale=ATTENTION_SCALE,
            bmm2_scale=1.0,
            backend="trtllm-gen",
            enable_pdl=self.pdl,
        )
        return result.view(c.tokens, c.heads, LATENT)
