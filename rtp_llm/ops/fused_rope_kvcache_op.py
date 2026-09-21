import logging
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


class DecodeRopeContractError(RuntimeError):
    """Decode RoPE input/state invariant violation that must not fall back."""


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


@cache
def _decode_uses_cu_seqlens(decode_op: Callable[..., torch.Tensor]) -> bool:
    return "cu_seqlens" in inspect.signature(decode_op).parameters


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
    prefix_lengths_device: Optional[torch.Tensor] = None


class FusedRopeKVCachePrefillOpBase:
    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs

    def prepare_kv_cache_offset(
        self, attn_inputs: PyAttentionInputs
    ) -> Optional[torch.Tensor]:
        """Build the device KV-offset table without preparing host scalars.

        CUDA-graph replay only needs to refresh the page-table-derived offset.
        Re-running ``prepare`` would also read length maxima back to the host,
        although the captured RoPE launch geometry is unchanged.
        """
        block_ids = attn_inputs.kv_cache_kernel_block_id_device
        if block_ids is None or block_ids.numel() == 0:
            return None
        return _get_fused_rope_kvcache().convert_offset_to_block_array(block_ids)

    def prepare(self, attn_inputs: PyAttentionInputs) -> FusedRopeAttnParams:
        kv_cache_offset = self.prepare_kv_cache_offset(attn_inputs)
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
            prefix_lengths_device=getattr(attn_inputs, "prefix_lengths_device", None),
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
        # Some kernels renamed cp_position_ids without adding interleaved
        # mRoPE support. Apply Qwen3.5's interleaved rotation explicitly;
        # the keyword name alone does not establish the rotary layout.
        legacy_mrope = rope_config.style == RopeStyle.Mrope and (
            position_ids_keyword == "cp_position_ids"
            or rope_config.mrope_interleaved
        )
        position_ids = params.position_ids
        if legacy_mrope:
            from rtp_llm.models_py.triton_kernels.common.prefill_mrope_cache import (
                maybe_prefill_mrope_cache,
            )

            if not (store_q or store_kv or store_qkv_fp8):
                fused = maybe_prefill_mrope_cache(
                    qkv,
                    kv_cache,
                    params,
                    self.attn_configs,
                    qout=store_q_no_transpose,
                    qkvout=store_qkv,
                )
                if fused is not None:
                    return fused
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
        self._sequence_lengths_device: Optional[torch.Tensor] = None

    def _get_kv_scale(self, kv_cache: LayerKVCache) -> Optional[torch.Tensor]:
        # FP8 KV cache uses direct cast (no dynamic scaling), so the kernel always writes
        # scale = 1.0. The buffer here is an output target for the kernel, not a real scale.
        #
        # `is not None` is sufficient here: pybind11 maps an undefined C++ torch::Tensor to
        # Python None, and the cache allocator only stores defined tensors with numel > 0
        # (see CoordinatorCacheManager::allLayerCacheBase). MHAKVCacheSpec guarantees
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

    def refresh_sequence_lengths(
        self,
        attn_inputs: PyAttentionInputs,
        device: Optional[torch.device] = None,
        *,
        forbid_reallocation: bool = False,
    ) -> torch.Tensor:
        """Refresh the stable CUDA buffer consumed by decode RoPE.

        CUDA graphs capture ``data_ptr()`` from the first prepared params, so
        the buffer is allocated once and updated in place for every replay. The
        authoritative ``sequence_lengths`` mirror is used instead of
        ``sequence_lengths_plus_1_device`` because RoPE needs the current token
        position rather than the plus-one KV length, and the graph runner zeros
        padded rows only in the former. CUDA graph callers must pass
        ``forbid_reallocation=True`` and enqueue this refresh on the same stream
        as the subsequent graph replay.
        """
        sequence_lengths = attn_inputs.sequence_lengths
        if sequence_lengths is None or sequence_lengths.numel() == 0:
            raise DecodeRopeContractError(
                "decode sequence_lengths must be a non-empty tensor"
            )
        if sequence_lengths.dtype != torch.int32:
            raise DecodeRopeContractError(
                f"decode sequence_lengths must use torch.int32, got {sequence_lengths.dtype}"
            )
        source_supports_async_copy = (
            sequence_lengths.is_cuda or sequence_lengths.is_pinned()
        )
        if not source_supports_async_copy and (
            forbid_reallocation or attn_inputs.is_cuda_graph
        ):
            raise DecodeRopeContractError(
                "CUDA graph decode sequence_lengths must be CUDA or pinned CPU memory"
            )

        if device is None:
            if self._sequence_lengths_device is not None:
                device = self._sequence_lengths_device.device
            elif sequence_lengths.is_cuda:
                device = sequence_lengths.device
            else:
                sequence_lengths_plus_1 = (
                    attn_inputs.sequence_lengths_plus_1_device
                )
                if (
                    sequence_lengths_plus_1 is None
                    or not sequence_lengths_plus_1.is_cuda
                ):
                    raise DecodeRopeContractError(
                        "a CUDA target device is required when sequence_lengths is on CPU"
                    )
                device = sequence_lengths_plus_1.device

        device = torch.device(device)
        if device.type != "cuda":
            raise DecodeRopeContractError(
                f"decode sequence_lengths target must be CUDA, got {device}"
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        needs_allocation = (
            self._sequence_lengths_device is None
            or self._sequence_lengths_device.shape != sequence_lengths.shape
            or self._sequence_lengths_device.device != device
        )
        if needs_allocation:
            if (
                self._sequence_lengths_device is not None
                and (forbid_reallocation or attn_inputs.is_cuda_graph)
            ):
                raise DecodeRopeContractError(
                    "CUDA graph replay cannot resize or move the captured sequence_lengths buffer"
                )
            if self._sequence_lengths_device is not None:
                logging.warning(
                    "Reallocating decode RoPE sequence_lengths buffer outside CUDA graph: "
                    "shape %s on %s -> shape %s on %s",
                    tuple(self._sequence_lengths_device.shape),
                    self._sequence_lengths_device.device,
                    tuple(sequence_lengths.shape),
                    device,
                )
            self._sequence_lengths_device = torch.empty(
                sequence_lengths.shape,
                dtype=torch.int32,
                device=device,
            )

        self._sequence_lengths_device.copy_(
            sequence_lengths,
            non_blocking=source_supports_async_copy,
        )
        return self._sequence_lengths_device

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
        cache_positions = params.sequence_lengths.to(
            device=qkv.device, non_blocking=True
        )
        return decode_op(
            rotated_qkv,
            cache_positions,
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
            # This historical ABI calls per-sequence cache positions
            # cu_seqlens; these are not cumulative query-token offsets.
            **({"cu_seqlens": cache_positions}
               if _decode_uses_cu_seqlens(decode_op) else {}),
        )

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: LayerKVCache,
        params: FusedRopeAttnParams,
    ) -> torch.Tensor:
        if params.kv_cache_offset is None:
            raise DecodeRopeContractError("decode RoPE requires kv_cache_offset")
        if not params.sequence_lengths.is_cuda:
            raise DecodeRopeContractError(
                "decode RoPE sequence_lengths must be on CUDA"
            )
        if params.sequence_lengths is not self._sequence_lengths_device:
            raise DecodeRopeContractError(
                "decode RoPE params lost the op-owned sequence_lengths buffer"
            )
        rope_config = self.attn_configs.rope_config
        rope_cache = get_rope_cache_once(rope_config, self.attn_configs.max_seq_len)
        decode_op = _get_fused_rope_kvcache().decode_fused_rope_kvcache
        sequence_lengths_kwargs = {}
        if _decode_uses_cu_seqlens(decode_op):
            sequence_lengths_kwargs["cu_seqlens"] = (
                params.cu_seqlens
                if _decode_uses_sequence_lengths(decode_op)
                else params.sequence_lengths.to(device=qkv.device, non_blocking=True)
            )
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

    def prepare(
        self,
        attn_inputs: PyAttentionInputs,
        *,
        forbid_reallocation: bool = False,
    ) -> FusedRopeAttnParams:
        assert (
            attn_inputs.kv_cache_kernel_block_id_device is not None
            and attn_inputs.kv_cache_kernel_block_id_device.numel() > 0
        )
        kv_cache_offset = _get_fused_rope_kvcache().convert_offset_to_block_array(
            attn_inputs.kv_cache_kernel_block_id_device
        )
        kv_cache_offset_h = None  # not used
        if self.attn_configs.need_rope_kv_cache:
            sequence_lengths_device = self.refresh_sequence_lengths(
                attn_inputs,
                kv_cache_offset.device,
                forbid_reallocation=forbid_reallocation,
            )
        else:
            # The params object is retained by attention implementations even
            # when RoPE is disabled, but its sequence lengths are never consumed.
            sequence_lengths_device = attn_inputs.sequence_lengths
        return FusedRopeAttnParams(
            kv_cache_offset,
            kv_cache_offset_h,
            attn_inputs.padding_offset,
            attn_inputs.combo_position_ids,
            attn_inputs.cu_seqlens_device,
            attn_inputs.cu_kv_seqlens_device,
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            sequence_lengths_device,
            0,
            0,
            attn_inputs.context_total_kv_length,
            True,
            get_scalar_type(attn_inputs.dtype),
        )
