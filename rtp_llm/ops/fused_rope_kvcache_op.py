import inspect
from dataclasses import dataclass, replace
from functools import cache
from typing import Callable, Optional

import torch

from librtp_compute_ops import LayerKVCache, PyAttentionInputs, get_scalar_type
from rtp_llm.ops.attention_input_utils import select_prefill_position_ids
from libth_transformer_config import (
    AttentionConfigs,
    RopeStyle,
    check_rope_cache,
    get_rope_cache_once,
)


@cache
def _get_fused_rope_kvcache():
    # Lazy: keeps import free of JIT builds; warm-up still hits this pre-readiness.
    from rtp_kernel import fused_rope_kvcache

    return fused_rope_kvcache


@cache
def _get_prefill_position_ids_keyword(
    prefill_op: Callable[..., torch.Tensor],
) -> str:
    parameters = inspect.signature(prefill_op).parameters
    if "position_ids" in parameters:
        return "position_ids"
    if "cp_position_ids" in parameters:
        return "cp_position_ids"
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return "position_ids"
    raise RuntimeError(
        "rtp_kernel prefill_fused_rope_kvcache supports neither "
        "position_ids nor cp_position_ids"
    )


@cache
def _decode_uses_sequence_lengths(
    decode_op: Callable[..., torch.Tensor],
) -> bool:
    parameters = inspect.signature(decode_op).parameters
    if "sequence_lengths" in parameters:
        return True
    if "batch_size" in parameters:
        return False
    raise RuntimeError(
        "rtp_kernel decode_fused_rope_kvcache has an unsupported signature: "
        "batch_size is missing"
    )


@dataclass
class FusedRopeAttnParams:
    kv_cache_offset: Optional[torch.Tensor]
    kv_cache_offset_h: Optional[torch.Tensor]
    padding_offset: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    cu_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    input_lengths: torch.Tensor
    prefix_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    max_seq_len: int
    max_prefix_length: int
    context_total_kv_length: int
    decode_plan: bool
    attn_type: torch.dtype


class FusedRopeKVCachePrefillOpBase:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs

    def prepare(self, attn_inputs: PyAttentionInputs) -> FusedRopeAttnParams:
        if (
            attn_inputs.kv_cache_kernel_block_id_device is not None
            and attn_inputs.kv_cache_kernel_block_id_device.numel() > 0
        ):
            kv_cache_offset = _get_fused_rope_kvcache().convert_offset_to_block_array(
                attn_inputs.kv_cache_kernel_block_id_device
            )
        else:
            kv_cache_offset = None
        kv_cache_offset_h = None  # not used

        # CP remaps explicit position IDs alongside the local token shard. Keep
        # them for models such as Qwen3-VL whose mRoPE positions have three axes;
        # shuffle indices are only a fallback for models without position IDs.
        position_ids = select_prefill_position_ids(attn_inputs)

        return FusedRopeAttnParams(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            position_ids,
            attn_inputs.cu_seqlens_device,
            attn_inputs.cu_kv_seqlens_device,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths.max().item(),
            attn_inputs.prefix_lengths.max().item(),
            attn_inputs.context_total_kv_length,
            False,
            get_scalar_type(attn_inputs.dtype),
        )

    def _forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: FusedRopeAttnParams,
        store_q_no_transpose: bool,
        store_q: bool,
        store_kv: bool,
        store_qkv: bool,
        store_qkv_fp8: bool,
        use_paged_fmha: bool,
    ) -> torch.Tensor:
        store_cache = kv_cache is not None
        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache_once(rope_config, self.attn_configs.max_seq_len)

        prefill_op = _get_fused_rope_kvcache().prefill_fused_rope_kvcache
        position_ids_keyword = _get_prefill_position_ids_keyword(prefill_op)
        legacy_mrope = (
            position_ids_keyword == "cp_position_ids"
            and rope_config.style == RopeStyle.Mrope
        )
        position_ids = params.position_ids
        if legacy_mrope:
            from rtp_llm.models_py.triton_kernels.common.legacy_mrope import (
                apply_mrope_qk_inplace,
            )

            if position_ids is None:
                # Text-only/warmup requests may omit explicit positions. Derive
                # their absolute token positions without a host synchronization.
                tokens = torch.arange(qkv.size(0), dtype=torch.int32, device=qkv.device)
                batch = torch.bucketize(tokens, params.cu_seqlens[1:], right=True)
                positions = tokens - params.cu_seqlens[batch]
                if params.max_prefix_length > 0:
                    positions = positions + params.prefix_lengths.to(qkv.device)[batch]
                position_ids = positions[:, None].expand(-1, 3)
            qkv = apply_mrope_qk_inplace(
                qkv,
                position_ids,
                self.attn_configs.head_num,
                self.attn_configs.kv_head_num,
                self.attn_configs.size_per_head,
                rope_config,
            )
            # The legacy kernel now only packs Q/K/V and writes physical KV
            # slots. Never let its old mRoPE math rotate the tensor again.
            position_ids = None

        return prefill_op(
            qkv,
            params.cu_seqlens,
            params.cu_seqlens.size(0) - 1,
            params.max_seq_len,
            self.attn_configs.head_num,
            self.attn_configs.kv_head_num,
            self.attn_configs.size_per_head,
            tokens_per_block=self.attn_configs.kernel_tokens_per_block,
            store_q_no_transpose=store_q_no_transpose,
            store_q=store_q,
            store_kv=store_kv,
            store_qkv=store_qkv,
            store_qkv_fp8=store_qkv_fp8,
            store_cache=store_cache,
            use_paged_fmha=use_paged_fmha,
            kv_cache=None if kv_cache is None else kv_cache.kv_cache_base,
            kv_cache_scale=None if kv_cache is None else kv_cache.kv_scale_base,
            kv_cache_offset=params.kv_cache_offset,
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=(
                rope_cache.data
                if not legacy_mrope and check_rope_cache(rope_config, rope_cache)
                else None
            ),
            padding_offset=params.padding_offset,
            use_logn_attn=self.attn_configs.use_logn_attn,
            rope_style=RopeStyle.No if legacy_mrope else rope_config.style,
            rope_dim=0 if legacy_mrope else rope_config.dim,
            rope_base=rope_config.base,
            rope_scale=rope_config.scale,
            rope_beta_slow=rope_config.factor1,
            rope_beta_fast=rope_config.factor2,
            rope_original_max_position_embeddings=rope_config.max_pos,
            rope_extrapolation_factor=rope_config.extrapolation_factor,
            rope_mscale=rope_config.mscale,
            rope_offset=rope_config.offset,
            rope_index_factor=1 if legacy_mrope else rope_config.index_factor,
            rope_mrope_dim1=rope_config.mrope_dim1,
            rope_mrope_dim2=rope_config.mrope_dim2,
            rope_mrope_dim3=rope_config.mrope_dim3,
            prefix_prompt_lengths=params.prefix_lengths,
            max_prefix_length=params.max_prefix_length,
            count_length=params.max_prefix_length > 0,
            **{position_ids_keyword: position_ids},
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        raise NotImplementedError()


class FusedRopeKVCachePrefillOpQKVOut(FusedRopeKVCachePrefillOpBase):
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        return self._forward(
            qkv, kv_cache, params, False, False, False, True, False, False
        )


class FusedRopeKVCachePrefillOpQOut(FusedRopeKVCachePrefillOpBase):
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        use_paged_fmha = kv_cache is not None and params.max_prefix_length > 0

        return self._forward(
            qkv, kv_cache, params, True, False, False, False, False, use_paged_fmha
        )


class FusedRopeKVCacheDecodeOp:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs
        self._dummy_scale: Optional[torch.Tensor] = None

    def _get_kv_scale(self, kv_cache: LayerKVCache) -> Optional[torch.Tensor]:
        # FP8 KV cache uses direct cast (no dynamic scaling), so the kernel always writes
        # scale = 1.0. The buffer here is an output target for the kernel, not a real scale.
        #
        # `is not None` is sufficient here: pybind11 maps an undefined C++ torch::Tensor to
        # Python None, and the cache allocator only stores defined tensors with numel > 0
        # (see SingleTypeKVCacheAllocator::allLayerCacheBase). MHAKVCacheSpec guarantees
        # FP8 dtype always has a scale buffer, so the dummy-scale branch below is purely
        # defensive and unreachable under normal operation.
        if kv_cache.kv_scale_base is not None:
            return kv_cache.kv_scale_base
        if kv_cache.kv_cache_base.dtype == torch.float8_e4m3fn:
            num_pages = kv_cache.kv_cache_base.shape[0]
            # convert_offset_to_block_array encodes K offset = page*2, V offset = page*2+1,
            # so max offset is 2*num_pages-1 and scale buffer needs 2*num_pages blocks.
            needed = 2 * num_pages * self.attn_configs.kernel_tokens_per_block * self.attn_configs.kv_head_num
            if self._dummy_scale is None or self._dummy_scale.numel() < needed:
                self._dummy_scale = torch.ones(needed, dtype=torch.float32, device=kv_cache.kv_cache_base.device)
            return self._dummy_scale
        return None

    def _legacy_mrope_decode(
        self,
        decode_op: Callable[..., torch.Tensor],
        qkv: torch.Tensor,
        kv_cache: LayerKVCache,
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        # Legacy decode uses its sole position_ids input both for RoPE and
        # the KV token offset. Image/video mRoPE coordinates cannot stand in
        # for the physical sequence length. Apply RoPE without a cache write,
        # then use the old decode kernel only to append already-rotated KV.
        batch_size = params.sequence_lengths.size(0)
        if qkv.size(0) != batch_size or params.position_ids is None:
            raise ValueError("legacy mRoPE decode requires one token and position IDs per sequence")
        prefill_params = replace(
            params,
            cu_seqlens=torch.arange(
                batch_size + 1, dtype=torch.int32, device=qkv.device
            ),
            padding_offset=None,
            prefix_lengths=torch.zeros(
                batch_size, dtype=torch.int32, device=qkv.device
            ),
            max_seq_len=1,
            max_prefix_length=0,
            decode_plan=False,
        )
        rotated_qkv = FusedRopeKVCachePrefillOpQKVOut(self.attn_configs).forward(
            qkv, None, prefill_params
        )
        return decode_op(
            rotated_qkv,
            params.sequence_lengths.to(device=qkv.device, non_blocking=True),
            batch_size=batch_size,
            head_num=self.attn_configs.head_num,
            kv_head_num=self.attn_configs.kv_head_num,
            size_per_head=self.attn_configs.size_per_head,
            kv_cache=kv_cache.kv_cache_base,
            kv_cache_offset=params.kv_cache_offset,
            tokens_per_block=self.attn_configs.kernel_tokens_per_block,
            store_kv=False,
            kv_cache_scale=self._get_kv_scale(kv_cache),
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=None,
            use_logn_attn=False,
            rope_style=RopeStyle.No,
            rope_dim=0,
            rope_index_factor=1,
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: LayerKVCache,
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache_once(rope_config, self.attn_configs.max_seq_len)
        assert params.kv_cache_offset is not None
        assert params.sequence_lengths.is_cuda or params.sequence_lengths.is_pinned(), (
            "sequence_lengths must be CUDA or pinned host memory"
        )
        decode_op = _get_fused_rope_kvcache().decode_fused_rope_kvcache
        sequence_lengths_kwargs = {}
        if _decode_uses_sequence_lengths(decode_op):
            sequence_lengths_kwargs["sequence_lengths"] = params.sequence_lengths
        elif rope_config.style == RopeStyle.Mrope:
            return self._legacy_mrope_decode(decode_op, qkv, kv_cache, params)

        return decode_op(
            qkv,
            params.position_ids,
            batch_size=params.sequence_lengths.size(0),
            head_num=self.attn_configs.head_num,
            kv_head_num=self.attn_configs.kv_head_num,
            size_per_head=self.attn_configs.size_per_head,
            kv_cache=kv_cache.kv_cache_base,
            kv_cache_offset=params.kv_cache_offset,
            tokens_per_block=self.attn_configs.kernel_tokens_per_block,
            store_kv=False,
            kv_cache_scale=self._get_kv_scale(kv_cache),
            kv_cache_offset_h=params.kv_cache_offset_h,
            rope_cache=(
                rope_cache.data if check_rope_cache(rope_config, rope_cache) else None
            ),
            use_logn_attn=self.attn_configs.use_logn_attn,
            rope_style=rope_config.style,
            rope_dim=rope_config.dim,
            rope_base=rope_config.base,
            rope_scale=rope_config.scale,
            rope_beta_slow=rope_config.factor1,
            rope_beta_fast=rope_config.factor2,
            rope_original_max_position_embeddings=rope_config.max_pos,
            rope_extrapolation_factor=rope_config.extrapolation_factor,
            rope_mscale=rope_config.mscale,
            rope_offset=rope_config.offset,
            rope_index_factor=rope_config.index_factor,
            rope_mrope_dim1=rope_config.mrope_dim1,
            rope_mrope_dim2=rope_config.mrope_dim2,
            rope_mrope_dim3=rope_config.mrope_dim3,
            **sequence_lengths_kwargs,
        )

    def prepare(self, attn_inputs: PyAttentionInputs) -> FusedRopeAttnParams:
        assert (
            attn_inputs.kv_cache_kernel_block_id_device is not None
            and attn_inputs.kv_cache_kernel_block_id_device.numel() > 0
        )
        kv_cache_offset = _get_fused_rope_kvcache().convert_offset_to_block_array(
            attn_inputs.kv_cache_kernel_block_id_device
        )
        kv_cache_offset_h = None  # not used
        return FusedRopeAttnParams(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            attn_inputs.combo_position_ids,
            attn_inputs.cu_seqlens_device,
            attn_inputs.cu_kv_seqlens_device,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            0,
            0,
            attn_inputs.context_total_kv_length,
            True,
            get_scalar_type(attn_inputs.dtype),
        )
