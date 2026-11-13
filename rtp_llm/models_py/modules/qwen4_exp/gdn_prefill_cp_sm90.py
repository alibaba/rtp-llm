# SPDX-License-Identifier: Apache-2.0
# Reuses FlashInfer 0.7.3 SM90 CP kernels. Only checkpoint addressing changes.
"""FlashInfer 0.7.3 H20 Prefill with direct RTP checkpoint writes.

Reuses the upstream SM90 precompute, fixup and prefill kernels. Only checkpoint
addressing changes; Q/K and beta rounding follow the native RTP contract.
"""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cuda.bindings import driver as cuda_driver
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm90 as upstream
from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import choose_cp_chunk_len_host
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.triton_kernels.common.scatter_qkv import (
    try_scatter_qkv_l2norm_sm90,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import _gdn_gate_values


class _PagedCP(upstream.CPDeltaRulePrefillSm90):
    @cute.jit
    def maybe_store_checkpoint(
        self,
        state,
        pool,
        page_ids,
        page_size,
        tiled_mma,
        thread,
        sequence,
        head,
        heads,
        count,
        block_end,
        sequence_length,
        ckpt_key=None,
        row_pref=None,
    ):
        end = block_end
        if block_end > sequence_length:
            end = sequence_length
        if end == sequence_length or end % page_size == cutlass.Int32(0):
            position = (end - cutlass.Int32(1)) // page_size
            page = cutlass.Int32(page_ids[position])
            if page > cutlass.Int32(0):
                row_stride = cute.assume(pool.stride[0], divby=4)
                layout = cute.make_layout(
                    (self.D, self.D, heads, pool.shape[0]),
                    stride=(1, self.D, self.D * self.D, row_stride),
                )
                destination = cute.make_tensor(pool.iterator, layout)
                self.kv_store(
                    state, destination[None, None, head, page], tiled_mma, thread
                )


@lru_cache(maxsize=2)
def _kernel(cumulative_dtype):
    return _PagedCP(
        dtype=cutlass.BFloat16,
        needs_initial_state=False,
        store_final_state=False,
        initial_state_dtype=cutlass.Float32,
        checkpoint_state_dtype=cutlass.Float32,
        needs_checkpointing=True,
        cu_seqlens_dtype=cumulative_dtype,
        checkpoint_cu_starts_dtype=cumulative_dtype,
    )


def _run(q, k, v, alpha, beta, pool, table, cumulative, page, cp_chunk_len=None):
    rows, query_heads, dimension = q.shape
    heads = v.shape[1]
    assert q.shape == k.shape == (rows, 8, 128) and v.shape == (rows, 24, 128)
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert alpha.dtype == beta.dtype == pool.dtype == torch.float32
    assert pool.shape[1:] == (24, 128, 128) and pool.stride()[1:] == (16384, 128, 1)
    assert pool.stride(0) % 4 == 0 and table.shape[0] == 1 and cumulative.numel() == 2
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert page in (64, 128)
    page_ids = table[0]
    assert page_ids.is_contiguous() and page_ids.numel() >= triton.cdiv(rows, page)
    device = q.device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    cp_chunk_len = cp_chunk_len or choose_cp_chunk_len_host(
        rows, heads, sms, device_capability=(9, 0), device_name="NVIDIA H20"
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    common = dict(
        cp_chunk_len=cp_chunk_len, _skip_check=True, _device=device, _stream=stream
    )
    t = upstream.cp_delta_rule_t_precompute_dsl_sm90(
        k,
        beta,
        cumulative,
        rows,
        max_seqlen=rows,
        _skip_check=True,
        _device=device,
        _stream=stream,
    )
    local_transfer, local_state = upstream.cp_delta_rule_mn_precompute_dsl_sm90(
        k,
        v,
        t,
        alpha,
        cumulative,
        rows,
        max_seqlen=rows,
        **common,
    )
    fixed = upstream.cp_delta_rule_fixup_dsl_sm90(
        local_transfer,
        local_state,
        cumulative,
        rows,
        initial_state=None,
        **common,
    )
    output = torch.empty_like(v)
    # The upstream signature requires a four-dimensional state argument even
    # when final-state storage is disabled. This view is never written there.
    state_argument = pool[:1]
    q_tma = q.as_strided(
        (rows, dimension, query_heads), (query_heads * dimension, 1, dimension)
    )
    k_tma = k.as_strided(
        (dimension, rows, query_heads), (1, query_heads * dimension, dimension)
    )
    v_tma = v.as_strided((dimension, rows, heads), (1, heads * dimension, dimension))
    o_tma = output.as_strided(
        (dimension, rows, heads), (1, heads * dimension, dimension)
    )
    chunks = t.shape[0]
    t_tma = t.as_strided((64, 64, heads, chunks), (64, 1, 4096, heads * 4096))
    descriptors = torch.empty(sms * 128, device=device, dtype=torch.uint8)
    total_cp_chunks = fixed.shape[0]
    dimensions = (
        query_heads,
        query_heads,
        heads,
        heads,
        cp_chunk_len,
        total_cp_chunks,
        triton.cdiv(rows, cp_chunk_len),
        1,
        pool.shape[0],
        page,
    )
    runtime = (
        q_tma,
        k_tma,
        v_tma,
        t_tma,
        o_tma,
        alpha.reshape(-1),
        state_argument,
        fixed.reshape(-1),
        None,
        None,
        pool,
        page_ids,
        descriptors,
        cumulative,
        128**-0.5,
        *dimensions,
        stream,
    )
    kernel = _kernel(cumulative.dtype)
    compiled = upstream.get_cached_compile(kernel, upstream._SM90_COMPILE_OPTIONS)
    if compiled is None:

        def tensor(value, leading_dim=None, align=16):
            converted = cute.runtime.from_dlpack(
                value, assumed_align=align, enable_tvm_ffi=True
            )
            if leading_dim is not None:
                return converted.mark_layout_dynamic(leading_dim=leading_dim)
            return converted.mark_layout_dynamic()

        arguments = (
            tensor(q_tma, 1),
            tensor(k_tma, 0),
            tensor(v_tma, 0),
            tensor(t_tma, 1),
            tensor(o_tma, 0),
            tensor(alpha.reshape(-1)),
            tensor(state_argument),
            tensor(fixed.reshape(-1)),
            None,
            None,
            tensor(pool),
            tensor(page_ids, align=4),
            tensor(descriptors, align=128),
            tensor(cumulative, align=4),
            cutlass.Float32(128**-0.5),
            *(cutlass.Int32(value) for value in dimensions),
            stream,
        )
        compiled = upstream.cached_compile(
            kernel, *arguments, compile_options=upstream._SM90_COMPILE_OPTIONS
        )
    compiled(*runtime)
    return output


@triton.jit
def _copy_value_alpha_beta(
    packed,
    a,
    b,
    alog,
    bias,
    v,
    alpha,
    beta,
    rows,
    STRIDE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    TOKENS: tl.constexpr = 16,
    HEADS: tl.constexpr = 1,
):
    token = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    first_head = tl.program_id(1) * HEADS
    head = first_head + tl.arange(0, HEADS)
    feature = tl.arange(0, HEADS * 128)
    valid = token < rows
    source = (
        token[:, None].to(tl.int64) * ((2 * H + HV) * 128)
        + (2 * H + first_head) * 128
        + feature[None, :]
    )
    value = tl.load(packed + source, valid[:, None], other=0)
    target = (
        token[:, None].to(tl.int64) * (HV * 128) + first_head * 128 + feature[None, :]
    )
    tl.store(v + target, value, valid[:, None])
    offset = token[:, None].to(tl.int64) * STRIDE + head[None, :]
    gate, update = _gdn_gate_values(
        tl.load(alog + head)[None, :],
        tl.load(a + offset, valid[:, None], other=0),
        tl.load(b + offset, valid[:, None], other=0),
        tl.load(bias + head)[None, :],
        1.0,
        20.0,
    )
    # Match torch.exp(g) and the original beta BF16 -> FP32 materialization.
    output = token[:, None] * HV + head[None, :]
    tl.store(alpha + output, libdevice.exp(gate), valid[:, None])
    tl.store(
        beta + output,
        update.to(b.dtype.element_ty).to(tl.float32),
        valid[:, None],
    )


def _prepare_cp(packed, a, b, alog, bias, heads, value_heads, *, group_heads=False):
    result = try_scatter_qkv_l2norm_sm90(
        packed, heads, value_heads, 128, 128, copy_v=False
    )
    if result is None:
        raise ValueError("CP preparation requires the qualified native SM90 layout")
    q, k, v = result
    rows = packed.shape[0]
    alpha = torch.empty((rows, value_heads), device=packed.device, dtype=torch.float32)
    beta = torch.empty_like(alpha)
    head_group = 8 if group_heads and rows == 32768 and value_heads == 24 else 1
    tokens = 4 if head_group == 8 else 16
    _copy_value_alpha_beta[(triton.cdiv(rows, tokens), value_heads // head_group)](
        packed,
        a,
        b,
        alog,
        bias,
        v,
        alpha,
        beta,
        rows,
        a.stride(0),
        heads,
        value_heads,
        TOKENS=tokens,
        HEADS=head_group,
        num_warps=4,
    )
    return q.squeeze(0), k.squeeze(0), v.squeeze(0), alpha, beta


def try_sm90_prefill(self, packed, b, a, cache, page, inputs, metadata):
    """Cold, single-request H20 Prefill; None keeps native dispatch."""
    if not (
        cache is not None
        and page == 64
        and packed.ndim == 2
        and packed.shape in ((8192, 5120), (32768, 5120))
        and packed.dtype == a.dtype == b.dtype == torch.bfloat16
        and packed.is_contiguous()
        and self.local_num_k_heads == 8
        and self.local_num_v_heads == 24
        and self.head_k_dim == self.head_v_dim == 128
        and a.shape == b.shape == (packed.shape[0], 24)
        and a.stride() == b.stride()
        and a.stride(1) == 1
        and not inputs.is_cuda_graph
        and inputs.input_lengths.device.type == "cpu"
        and inputs.prefix_lengths.device.type == "cpu"
        and inputs.input_lengths.numel() == inputs.prefix_lengths.numel() == 1
        and int(inputs.input_lengths[0]) == packed.shape[0]
        and int(inputs.prefix_lengths[0]) == 0
        and not any(value.requires_grad for value in (packed, a, b))
    ):
        return None
    pool = self._get_ssm_states(cache)
    if not (
        pool.dtype == torch.float32
        and pool.shape[1:] == (24, 128, 128)
        and pool.stride()[1:] == (16384, 128, 1)
        and pool.stride(0) % 4 == 0
    ):
        return None
    q, k, v, alpha, beta = _prepare_cp(
        packed,
        a,
        b,
        self.alog,
        self.dt_bias,
        8,
        24,
        group_heads=getattr(self, "sm90_grouped_prepare", False),
    )
    return _run(
        q,
        k,
        v,
        alpha,
        beta,
        pool,
        inputs.kv_cache_kernel_block_id_device,
        inputs.cu_seqlens_device,
        page,
        cp_chunk_len=2048 if packed.shape[0] == 32768 else None,
    )
