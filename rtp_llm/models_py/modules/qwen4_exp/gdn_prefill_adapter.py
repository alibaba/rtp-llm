"""Qwen4-only SM120 prefill candidate retaining every resumable cache page.

The FlashInfer0.7 CP implementation is isolated under gdn_sm120. It does
not replace installed FlashInfer or the speculative Decode state owner.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNetPrefill
from rtp_llm.models_py.triton_kernels.fla.block import load_initial_state_from_block_map


@triton.jit(do_not_specialize=["TABLE_STRIDE", "CACHE_STRIDE"])
def _store_page_checkpoints(
    Checkpoints,
    Final,
    Starts,
    Cu,
    Prefix,
    Table,
    Cache,
    TABLE_STRIDE,
    CACHE_STRIDE,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch, page, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Cu + batch + 1) - tl.load(Cu + batch)
    prefix = tl.load(Prefix + batch)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = HEADS * 128 * 128
    valid_page = page * PAGE < length
    block = tl.load(
        Table + batch * TABLE_STRIDE + prefix // PAGE + page, valid_page, other=0
    ).to(tl.int64)
    final_page = (page + 1) * PAGE >= length
    checkpoint = tl.load(Starts + batch) + page
    a = tl.load(
        Checkpoints + checkpoint * width + col,
        valid_page & ~final_page & (col < width),
        other=0,
    )
    b = tl.load(
        Final + batch * width + col, valid_page & final_page & (col < width), other=0
    )
    tl.store(
        Cache + block * CACHE_STRIDE + col,
        tl.where(final_page, b, a),
        valid_page & (block > 0) & (col < width),
    )


def sm120_prefill_prepared(
    q, k, v, alpha, beta_fp32, initial, cu_seqlens, lengths, page_size=128
):
    from .gdn_prefill_prepare import checkpoint_starts as make_checkpoint_starts

    heads = max(q.shape[1], v.shape[1])
    if page_size != 128:
        raise ValueError("SM120 prefill checkpoint pages must contain 128 tokens")
    checkpoint_starts = make_checkpoint_starts(cu_seqlens, len(lengths))
    checkpoints = torch.empty(
        (sum(n // page_size for n in lengths), heads, 128, 128),
        device=q.device,
        dtype=torch.float32,
    )
    final = torch.empty(
        (len(lengths), heads, 128, 128), device=q.device, dtype=torch.float32
    )
    output = torch.empty((q.shape[0], heads, 128), device=q.device, dtype=q.dtype)
    initial_fp32 = initial.float().contiguous() if initial is not None else None
    from .gdn_sm120.delta_rule_cp_sm120 import cp_delta_rule_dsl_sm120

    cp_delta_rule_dsl_sm120(
        output,
        final,
        q,
        k,
        v,
        alpha,
        beta_fp32,
        cu_seqlens,
        128**-0.5,
        initial_state=initial_fp32,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=checkpoint_starts,
        checkpoint_every_n_tokens=page_size,
        max_seqlen=max(lengths),
    )
    return output.unsqueeze(0), checkpoints, final, checkpoint_starts


class Qwen4Sm120GdnPrefill(Qwen3NextGatedDeltaNetPrefill):
    def _fla(
        self,
        mixed_qkv,
        b,
        a,
        kv_cache_tensor,
        seq_size_per_block,
        attn_inputs,
        attn_meta,
    ):
        host_lengths = attn_inputs.input_lengths
        host_prefixes = attn_inputs.prefix_lengths
        supported = (
            mixed_qkv.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(mixed_qkv.device) == (12, 0)
            and mixed_qkv.dtype == torch.bfloat16
            and 2048 <= mixed_qkv.shape[0] <= 32768
            and self.head_k_dim == self.head_v_dim == 128
            and seq_size_per_block == 128
            and not attn_inputs.is_cuda_graph
            and host_lengths.device.type == "cpu"
            and host_prefixes.device.type == "cpu"
        )
        if not supported:
            return super()._fla(
                mixed_qkv,
                b,
                a,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta,
            )
        lengths = tuple(int(n) for n in host_lengths.tolist())
        prefixes = tuple(int(n) for n in host_prefixes.tolist())
        if any(n < 0 or n % 128 for n in prefixes) or any(n <= 0 for n in lengths):
            return super()._fla(
                mixed_qkv,
                b,
                a,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta,
            )
        from .gdn_prefill_prepare import is_supported as preparation_supported

        if not preparation_supported(
            mixed_qkv,
            a,
            b,
            self.alog,
            self.dt_bias,
            self.local_num_k_heads,
            self.local_num_v_heads,
        ):
            return super()._fla(
                mixed_qkv,
                b,
                a,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta,
            )
        states = (
            self._get_ssm_states(kv_cache_tensor)
            if kv_cache_tensor is not None
            else None
        )
        initial = None
        if states is not None:
            initial = torch.empty(
                (len(lengths), self.local_num_v_heads, 128, 128),
                device=mixed_qkv.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                states,
                initial,
                128,
            )
        from .gdn_prefill_prepare import prepare_sm120_prefill

        q, k, v, alpha, beta = prepare_sm120_prefill(
            mixed_qkv,
            a,
            b,
            self.alog,
            self.dt_bias,
            self.local_num_k_heads,
            self.local_num_v_heads,
        )
        output, checkpoints, final, starts = sm120_prefill_prepared(
            q, k, v, alpha, beta, initial, attn_inputs.cu_seqlens_device, lengths
        )
        if states is not None:
            _store_page_checkpoints[
                (
                    len(lengths),
                    triton.cdiv(max(lengths), 128),
                    triton.cdiv(self.local_num_v_heads * 128 * 128, 1024),
                )
            ](
                checkpoints,
                final,
                starts,
                attn_inputs.cu_seqlens_device,
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                states,
                TABLE_STRIDE=attn_inputs.kv_cache_kernel_block_id_device.stride(0),
                CACHE_STRIDE=states.stride(0),
                HEADS=self.local_num_v_heads,
                PAGE=128,
                BLOCK=1024,
            )
        return output.squeeze(0)
