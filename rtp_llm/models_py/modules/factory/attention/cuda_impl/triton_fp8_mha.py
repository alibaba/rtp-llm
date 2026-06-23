import logging
import os
from typing import Optional

import torch
import triton

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.triton_fp8_mha_kernels import (
    _triton_fp8_paged_mha_kernel,
    _triton_fp8_paged_mha_split_combine_kernel,
    _triton_fp8_paged_mha_split_kernel,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops


def _debug_sync(stage: str) -> None:
    if os.environ.get("RTP_LLM_DEBUG_FP8_PER_TOKEN_HEAD") == "1":
        torch.cuda.synchronize()
        logging.warning("Triton FP8 per-token-head stage done: %s", stage)


def _is_fp8_per_token_head(attn_configs: AttentionConfigs) -> bool:
    return (
        attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        and getattr(attn_configs, "fp8_kv_cache_scale_mode", "per_tensor")
        == "per_token_head"
    )


def _is_cuda_graph(attn_inputs: PyAttentionInputs) -> bool:
    return bool(getattr(attn_inputs, "is_cuda_graph", False))


def _get_mha_scale_views(
    kv_cache: LayerKVCache,
    num_kv_heads: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = kv_cache.kv_scale_base
    num_blocks = scale.size(0)
    scale_flat = scale.reshape(num_blocks, -1)
    scale_per_kv = num_kv_heads * page_size
    k_flat = scale_flat[:, :scale_per_kv]
    v_flat = scale_flat[:, scale_per_kv : 2 * scale_per_kv]
    k_scale = torch.as_strided(
        k_flat,
        (num_blocks, page_size, num_kv_heads),
        (scale_flat.stride(0), 1, page_size),
    )
    v_scale = torch.as_strided(
        v_flat,
        (num_blocks, page_size, num_kv_heads),
        (scale_flat.stride(0), 1, page_size),
    )
    return k_scale, v_scale


class TritonFp8PagedMHAOp:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_size = attn_configs.size_per_head
        self.block_size = attn_configs.kernel_tokens_per_block
        self.softmax_scale = self.head_size**-0.5

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: LayerKVCache,
        attn_inputs: PyAttentionInputs,
    ) -> torch.Tensor:
        q = q.reshape(q.shape[0], self.num_heads, self.head_size)
        paged_kv_cache = common.reshape_paged_kv_cache(
            kv_cache.kv_cache_base, self.num_kv_heads, self.block_size, self.head_size
        )
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]
        k_scale_cache, v_scale_cache = _get_mha_scale_views(
            kv_cache, self.num_kv_heads, self.block_size
        )

        batch_size = attn_inputs.input_lengths.size(0)
        is_cuda_graph = _is_cuda_graph(attn_inputs)
        input_lengths = attn_inputs.input_lengths.to(q.device, non_blocking=True)
        if not attn_inputs.is_prefill:
            if is_cuda_graph:
                input_lengths = attn_inputs.input_lengths_d
            else:
                q_len = q.shape[0] // batch_size
                input_lengths = torch.full(
                    (batch_size,),
                    q_len,
                    dtype=input_lengths.dtype,
                    device=q.device,
                )
        cu_q = attn_inputs.cu_seqlens
        if (
            is_cuda_graph
            and not attn_inputs.is_prefill
            and attn_inputs.decode_cu_seqlens_d.numel() >= batch_size + 1
        ):
            cu_q = attn_inputs.decode_cu_seqlens_d[: batch_size + 1]
        elif not attn_inputs.is_prefill or cu_q.numel() != batch_size + 1:
            cu_q = torch.empty(
                batch_size + 1, dtype=input_lengths.dtype, device=q.device
            )
            cu_q[0] = 0
            cu_q[1:] = torch.cumsum(input_lengths, dim=0)
        seq_lens = attn_inputs.sequence_lengths
        if (
            is_cuda_graph
            and not attn_inputs.is_prefill
            and attn_inputs.sequence_lengths_plus_1_d.numel() >= batch_size
        ):
            seq_lens = attn_inputs.sequence_lengths_plus_1_d[:batch_size]
        elif seq_lens.numel() != batch_size:
            seq_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths
        if not seq_lens.is_cuda:
            seq_lens = seq_lens.to(q.device, non_blocking=True)
        if not attn_inputs.is_prefill and not is_cuda_graph:
            seq_lens = seq_lens + input_lengths
        block_table = attn_inputs.kv_cache_kernel_block_id_device
        out = torch.empty_like(q)
        head_size_padded = triton.next_power_of_2(self.head_size)
        block_n = 64
        split_size = int(os.environ.get("RTP_LLM_FP8_PTH_SPLIT_SIZE", "256"))
        use_split_decode = (
            not attn_inputs.is_prefill
            and split_size > 0
            and block_table.shape[1] * self.block_size > split_size
        )
        if os.environ.get("RTP_LLM_DEBUG_FP8_PER_TOKEN_HEAD") == "1":
            block_table_cpu = block_table.detach().cpu()
            seq_lens_cpu = seq_lens.detach().cpu()
            cu_q_cpu = cu_q.detach().cpu()
            logging.warning(
                "Triton FP8 per-token-head input shapes: q=%s k_cache=%s v_cache=%s "
                "k_scale=%s block_table=%s cu_q=%s seq_lens=%s block_table_min=%s "
                "block_table_max=%s seq_lens_values=%s cu_q_values=%s",
                tuple(q.shape),
                tuple(k_cache.shape),
                tuple(v_cache.shape),
                tuple(k_scale_cache.shape),
                tuple(block_table.shape),
                tuple(cu_q.shape),
                tuple(seq_lens.shape),
                block_table_cpu.min().item() if block_table_cpu.numel() > 0 else None,
                block_table_cpu.max().item() if block_table_cpu.numel() > 0 else None,
                seq_lens_cpu.tolist(),
                cu_q_cpu.tolist(),
            )

        if use_split_decode:
            max_kv_tokens = block_table.shape[1] * self.block_size
            num_splits = triton.cdiv(max_kv_tokens, split_size)
            partial_acc = torch.empty(
                (q.shape[0], self.num_heads, num_splits, head_size_padded),
                device=q.device,
                dtype=torch.float32,
            )
            partial_m = torch.empty(
                (q.shape[0], self.num_heads, num_splits),
                device=q.device,
                dtype=torch.float32,
            )
            partial_l = torch.empty_like(partial_m)
            _triton_fp8_paged_mha_split_kernel[
                (q.shape[0], self.num_heads, num_splits)
            ](
                q,
                k_cache,
                v_cache,
                k_scale_cache,
                v_scale_cache,
                block_table,
                cu_q,
                seq_lens,
                partial_acc,
                partial_m,
                partial_l,
                total_tokens=q.shape[0],
                batch_size=batch_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                block_size=self.block_size,
                block_table_stride=block_table.stride(0),
                q_stride_t=q.stride(0),
                q_stride_h=q.stride(1),
                q_stride_d=q.stride(2),
                k_stride_b=k_cache.stride(0),
                k_stride_h=k_cache.stride(1),
                k_stride_s=k_cache.stride(2),
                k_stride_d=k_cache.stride(3),
                v_stride_b=v_cache.stride(0),
                v_stride_h=v_cache.stride(1),
                v_stride_s=v_cache.stride(2),
                v_stride_d=v_cache.stride(3),
                ks_stride_b=k_scale_cache.stride(0),
                ks_stride_s=k_scale_cache.stride(1),
                ks_stride_h=k_scale_cache.stride(2),
                vs_stride_b=v_scale_cache.stride(0),
                vs_stride_s=v_scale_cache.stride(1),
                vs_stride_h=v_scale_cache.stride(2),
                partial_stride_t=partial_acc.stride(0),
                partial_stride_h=partial_acc.stride(1),
                partial_stride_s=partial_acc.stride(2),
                partial_stride_d=partial_acc.stride(3),
                stats_stride_t=partial_m.stride(0),
                stats_stride_h=partial_m.stride(1),
                stats_stride_s=partial_m.stride(2),
                softmax_scale=self.softmax_scale,
                BLOCK_N=block_n,
                SPLIT_SIZE=split_size,
                HEAD_SIZE_PADDED=head_size_padded,
            )
            _triton_fp8_paged_mha_split_combine_kernel[(q.shape[0], self.num_heads)](
                partial_acc,
                partial_m,
                partial_l,
                out,
                num_splits=num_splits,
                head_size=self.head_size,
                partial_stride_t=partial_acc.stride(0),
                partial_stride_h=partial_acc.stride(1),
                partial_stride_s=partial_acc.stride(2),
                partial_stride_d=partial_acc.stride(3),
                stats_stride_t=partial_m.stride(0),
                stats_stride_h=partial_m.stride(1),
                stats_stride_s=partial_m.stride(2),
                out_stride_t=out.stride(0),
                out_stride_h=out.stride(1),
                out_stride_d=out.stride(2),
                HEAD_SIZE_PADDED=head_size_padded,
            )
            return out

        _triton_fp8_paged_mha_kernel[(q.shape[0], self.num_heads)](
            q,
            k_cache,
            v_cache,
            k_scale_cache,
            v_scale_cache,
            block_table,
            cu_q,
            seq_lens,
            out,
            total_tokens=q.shape[0],
            batch_size=batch_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            block_size=self.block_size,
            block_table_stride=block_table.stride(0),
            q_stride_t=q.stride(0),
            q_stride_h=q.stride(1),
            q_stride_d=q.stride(2),
            k_stride_b=k_cache.stride(0),
            k_stride_h=k_cache.stride(1),
            k_stride_s=k_cache.stride(2),
            k_stride_d=k_cache.stride(3),
            v_stride_b=v_cache.stride(0),
            v_stride_h=v_cache.stride(1),
            v_stride_s=v_cache.stride(2),
            v_stride_d=v_cache.stride(3),
            ks_stride_b=k_scale_cache.stride(0),
            ks_stride_s=k_scale_cache.stride(1),
            ks_stride_h=k_scale_cache.stride(2),
            vs_stride_b=v_scale_cache.stride(0),
            vs_stride_s=v_scale_cache.stride(1),
            vs_stride_h=v_scale_cache.stride(2),
            out_stride_t=out.stride(0),
            out_stride_h=out.stride(1),
            out_stride_d=out.stride(2),
            softmax_scale=self.softmax_scale,
            BLOCK_N=block_n,
            HEAD_SIZE_PADDED=head_size_padded,
        )
        return out


class TritonFp8PagedPrefillImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_impl = TritonFp8PagedMHAOp(attn_configs)
        self.rope_impl = (
            None
            if attn_configs.rope_config.style == RopeStyle.No
            else MhaRotaryEmbeddingOp(attn_configs)
        )
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
            fp8_kv_cache_scale_mode="per_token_head",
            kv_cache_dtype=attn_configs.kv_cache_dtype,
        )
        self.params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_params = self.params
        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            attn_configs.kernel_tokens_per_block,
        )
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.params)
        self.kv_cache_write_op.set_params(self.params)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_dim = self.attn_configs.size_per_head
        q, k, v = torch.split(
            qkv.reshape(qkv.shape[0], -1),
            [
                head_dim * self.attn_configs.head_num,
                head_dim * self.attn_configs.kv_head_num,
                head_dim * self.attn_configs.kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.attn_configs.head_num, head_dim)
        k = k.reshape(k.shape[0], self.attn_configs.kv_head_num, head_dim)
        v = v.reshape(q.shape[0], self.attn_configs.kv_head_num, head_dim)
        return q, k, v

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            attn_inputs.is_prefill
            and _is_fp8_per_token_head(attn_configs)
            and not attn_configs.use_mla
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.attn_configs.kernel_tokens_per_block,
            True,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        assert (
            kv_cache is not None
        ), "kv_cache is required for Triton FP8 paged attention"
        if self.need_rope_kv_cache:
            if self.rope_impl is None:
                q, k, v = self._split_qkv(qkv)
            else:
                q, k, v = self.rope_impl.forward(qkv)
                _debug_sync("prefill_rope")
            self.kv_cache_write_op.forward(k, v, kv_cache)
            _debug_sync("prefill_kv_write")
        else:
            q = qkv.reshape(
                qkv.shape[0],
                self.attn_configs.head_num,
                self.attn_configs.size_per_head,
            )

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        output = self.fmha_impl.forward(q, kv_cache, self.attn_inputs)
        _debug_sync("prefill_triton_mha")
        return output


class TritonFp8PagedDecodeImpl(FMHAImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_impl = TritonFp8PagedMHAOp(attn_configs)
        self.rope_impl = (
            None
            if attn_configs.rope_config.style == RopeStyle.No
            else MhaRotaryEmbeddingOp(attn_configs)
        )
        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
            fp8_kv_cache_scale_mode="per_token_head",
            kv_cache_dtype=attn_configs.kv_cache_dtype,
        )
        self.params = rtp_llm_ops.FlashInferMlaAttnParams()
        self.fmha_params = self.params
        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            attn_configs.kernel_tokens_per_block,
        )
        if self.rope_impl is not None:
            self.rope_impl.set_params(self.params)
        self.kv_cache_write_op.set_params(self.params)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    def _split_qkv(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_dim = self.attn_configs.size_per_head
        q, k, v = torch.split(
            qkv.reshape(qkv.shape[0], -1),
            [
                head_dim * self.attn_configs.head_num,
                head_dim * self.attn_configs.kv_head_num,
                head_dim * self.attn_configs.kv_head_num,
            ],
            dim=-1,
        )
        q = q.reshape(q.shape[0], self.attn_configs.head_num, head_dim)
        k = k.reshape(q.shape[0], self.attn_configs.kv_head_num, head_dim)
        v = v.reshape(q.shape[0], self.attn_configs.kv_head_num, head_dim)
        return q, k, v

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            not attn_inputs.is_prefill
            and _is_fp8_per_token_head(attn_configs)
            and not attn_configs.use_mla
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.attn_configs.kernel_tokens_per_block,
            True,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        assert (
            kv_cache is not None
        ), "kv_cache is required for Triton FP8 paged attention"
        if self.need_rope_kv_cache:
            if self.rope_impl is None:
                q, k, v = self._split_qkv(qkv)
            else:
                q, k, v = self.rope_impl.forward(qkv)
                _debug_sync("decode_rope")
            self.kv_cache_write_op.forward(k, v, kv_cache)
            _debug_sync("decode_kv_write")
        else:
            q = qkv.reshape(
                qkv.shape[0],
                self.attn_configs.head_num,
                self.attn_configs.size_per_head,
            )
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        output = self.fmha_impl.forward(q, kv_cache, self.attn_inputs)
        _debug_sync("decode_triton_mha")
        return output
