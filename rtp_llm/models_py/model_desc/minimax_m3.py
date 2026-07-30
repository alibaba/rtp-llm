import functools
import logging
import os
from typing import Any, Dict, Optional

import torch
from rtp_kernel.fused_rope_kvcache import convert_offset_to_block_array
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeModel,
)
from rtp_llm.models_py.model_desc.multimodal_generic import MultimodalGenericModel
from rtp_llm.models_py.modules import CausalAttention
from rtp_llm.models_py.modules.factory.attention.common import copy_kv_cache_offset
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention
from rtp_llm.ops import HWKernelConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs
from rtp_llm.utils.model_weight import W


class MiniMaxM3DecoderLayer(GenericMoeDecoderLayer):
    def _create_attention(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: Any,
        hw_kernel_config: Optional[HWKernelConfig],
    ) -> nn.Module:
        if config.attn_config.use_mla:
            return super()._create_attention(
                config,
                parallelism_config,
                weights,
                global_weights,
                layer_idx,
                quant_config,
                hw_kernel_config,
            )

        # MiniMax-M3 attention weights are not tensor-parallel sharded.
        attn_configs = config.getAttentionConfigs(1)
        msa_config = config.msa_sparse_config
        is_sparse_layer = (
            msa_config is not None
            and layer_idx in set(msa_config.get("sparse_layer_ids", []))
            and W.msa_idx_q_w in weights
        )
        if is_sparse_layer:
            return MSAAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                msa_config,
                layer_idx,
                quant_config,
                hw_kernel_config,
            )
        return CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )

    def _input_quant_projection(self) -> Optional[nn.Module]:
        if isinstance(self.self_attn, MSAAttention):
            return getattr(self.self_attn, "qkv_proj", None)
        return super()._input_quant_projection()

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        prev_topk_indices: Optional[torch.Tensor],
        force_reuse_topk_indices: bool,
        attn_inputs: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(self.self_attn, MSAAttention):
            return super()._forward_attention(
                hidden_states,
                fmha_impl,
                kv_cache,
                prev_topk_indices,
                force_reuse_topk_indices,
                attn_inputs,
                x_fp8,
                x_scale,
            )

        quantized_inputs = {}
        if x_fp8 is not None:
            quantized_inputs = {"x_fp8": x_fp8, "x_scale": x_scale}
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attn_inputs=attn_inputs,
            kv_cache=kv_cache,
            **quantized_inputs,
        )
        return hidden_states, None


def _target_verify_width(attn_inputs) -> int:
    prefix_lengths = attn_inputs.prefix_lengths
    if prefix_lengths is None or prefix_lengths.numel() == 0:
        raise RuntimeError("MiniMax-M3 target verify requires prefix lengths")
    request_rows = int(prefix_lengths.numel())
    total_tokens = int(attn_inputs.total_tokens)
    if total_tokens == 0:
        input_lengths = attn_inputs.input_lengths
        if input_lengths is None or input_lengths.numel() != request_rows:
            raise RuntimeError(
                "MiniMax-M3 target verify capture metadata is incomplete"
            )
        verify_tokens = int(input_lengths[0].item())
        if verify_tokens <= 0 or not bool(torch.all(input_lengths == verify_tokens)):
            raise RuntimeError(
                "MiniMax-M3 target verify capture rows must have one fixed width"
            )
        return verify_tokens
    if total_tokens < 0 or total_tokens % request_rows != 0:
        raise RuntimeError(
            "MiniMax-M3 target verify token rows must be divisible by request rows: "
            f"tokens={total_tokens}, requests={request_rows}"
        )
    return total_tokens // request_rows


def _validate_target_verify_replay_shape(attn_inputs, verify_tokens: int) -> None:
    request_capacity = int(attn_inputs.prefix_lengths.numel())
    total_tokens = int(attn_inputs.total_tokens)
    token_capacity = request_capacity * verify_tokens
    if total_tokens < 0 or total_tokens > token_capacity:
        raise RuntimeError(
            "MiniMax-M3 target verify replay exceeds the captured token capacity: "
            f"tokens={total_tokens}, capacity={token_capacity}"
        )
    if total_tokens > 0 and total_tokens % verify_tokens != 0:
        raise RuntimeError(
            "MiniMax-M3 target verify replay contains an incomplete request window: "
            f"tokens={total_tokens}, verify_tokens={verify_tokens}"
        )


def _expand_target_verify_rows(
    prefix_lengths: torch.Tensor,
    block_table: torch.Tensor,
    verify_tokens: int,
    valid_requests: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prefix_lengths.numel() == 0:
        raise RuntimeError("MiniMax-M3 target verify requires prefix lengths")
    if block_table.dim() != 2:
        raise RuntimeError("MiniMax-M3 target verify requires a 2-D block table")
    if block_table.shape[0] != prefix_lengths.numel():
        raise RuntimeError("MiniMax-M3 target verify block-table batch mismatch")
    if verify_tokens <= 0:
        raise RuntimeError("MiniMax-M3 target verify token count must be positive")

    positions = torch.arange(
        1, verify_tokens + 1, dtype=torch.int32, device=prefix_lengths.device
    )
    sequence_lengths_plus_1 = prefix_lengths.to(dtype=torch.int32).unsqueeze(
        1
    ) + positions.unsqueeze(0)
    if valid_requests is not None:
        if valid_requests.numel() != prefix_lengths.numel():
            raise RuntimeError(
                "MiniMax-M3 target verify valid-request mask batch mismatch"
            )
        sequence_lengths_plus_1.masked_fill_(
            ~valid_requests.to(dtype=torch.bool).unsqueeze(1), 0
        )
    sequence_lengths_plus_1 = sequence_lengths_plus_1.reshape(-1)
    token_block_table = block_table.repeat_interleave(verify_tokens, dim=0).contiguous()
    return sequence_lengths_plus_1, token_block_table


def _update_target_verify_rope_kv_offset(rope_params, block_table) -> None:
    """Refresh the graph-owned RoPE KV offset from the current block table."""
    if block_table is None or block_table.numel() == 0:
        raise RuntimeError("MiniMax-M3 target verify requires a KV block table")
    if rope_params is None or rope_params.kv_cache_offset is None:
        raise RuntimeError("MiniMax-M3 target verify RoPE parameters are incomplete")

    new_offset = convert_offset_to_block_array(block_table)
    copy_kv_cache_offset(rope_params.kv_cache_offset, new_offset)


def _fill_target_verify_compact_lengths(
    prefix_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    verify_tokens: int,
    output: torch.Tensor,
    clear_padded_requests: bool,
) -> None:
    if (
        prefix_lengths.numel() != input_lengths.numel()
        or output.numel() != prefix_lengths.numel()
    ):
        raise RuntimeError("MiniMax-M3 target verify compact metadata batch mismatch")
    torch.add(prefix_lengths, verify_tokens, out=output)
    if clear_padded_requests:
        output.masked_fill_(input_lengths <= 0, 0)


def _require_target_verify_physical_block_table(attn_inputs) -> torch.Tensor:
    physical_block_table = getattr(attn_inputs, "kv_cache_block_id_device", None)
    if (
        not isinstance(physical_block_table, torch.Tensor)
        or physical_block_table.numel() == 0
    ):
        raise RuntimeError(
            "MiniMax-M3 FA4 target verify requires a physical KV block table"
        )
    return physical_block_table


def _requested_target_verify_backend() -> str:
    requested_backend = (
        os.environ.get("RTP_LLM_M3_TARGET_VERIFY_BACKEND", "flashinfer").strip().lower()
    )
    if requested_backend not in ("auto", "fa4", "flashinfer"):
        raise ValueError(
            "RTP_LLM_M3_TARGET_VERIFY_BACKEND must be one of "
            f"auto, fa4, flashinfer; got {requested_backend!r}"
        )
    return requested_backend


def _target_verify_query_dtype(
    kv_cache_dtype: KvCacheDataType,
    model_dtype: torch.dtype,
    backend: str,
) -> torch.dtype:
    # FA4 requires Q/K/V to use the same dtype, while FlashInfer supports the
    # model activation dtype for Q with an independently quantized KV cache.
    # Keep FlashInfer on the historical BF16-Q path; planning it as FP8-Q is
    # unsupported on this runtime and changes target-model numerics.
    if backend == "fa4" and kv_cache_dtype == KvCacheDataType.FP8:
        return torch.float8_e4m3fn
    return model_dtype


class _TargetVerifyFA4Metadata:
    def __init__(
        self,
        cu_seqlens_q: torch.Tensor,
        kv_sequence_lengths: torch.Tensor,
        physical_block_table: torch.Tensor,
    ) -> None:
        self.cu_seqlens_q = cu_seqlens_q
        self.kv_sequence_lengths = kv_sequence_lengths
        self.physical_block_table = physical_block_table


@functools.lru_cache(maxsize=1)
def _target_verify_impl_class():
    fa4_import_error = None
    try:
        from flash_attn.cute import flash_attn_varlen_func as fa4_varlen_func

        fa4_available = True
    except (ImportError, OSError) as error:
        fa4_varlen_func = None
        fa4_available = False
        fa4_import_error = error

    from rtp_llm.models_py.modules.factory.attention import common
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeAttnOp,
        PyFlashinferDecodeImpl,
    )
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCachePrefillOpQNoTransposeOut,
        get_scalar_type,
    )

    class MiniMaxM3TargetVerifyAttnOp(PyFlashinferDecodeAttnOp):
        def __init__(self, attn_configs):
            super().__init__(attn_configs)
            self._verify_tokens = None
            requested_backend = _requested_target_verify_backend()
            runs_on_blackwell = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 10
            )
            fa4_target_verify_supported = (
                fa4_available
                and runs_on_blackwell
                and self.kv_cache_dtype == KvCacheDataType.FP8
            )
            if requested_backend == "fa4" and not fa4_target_verify_supported:
                raise RuntimeError(
                    "MiniMax-M3 target verify FA4 requires Blackwell, "
                    "flash-attn-4, and FP8 KV cache"
                ) from fa4_import_error
            if requested_backend == "fa4" or (
                requested_backend == "auto" and fa4_target_verify_supported
            ):
                self._backend = "fa4"
            else:
                self._backend = "flashinfer"
            self._fa4_metadata_by_request_capacity = {}
            self._active_fa4_metadata = None
            self._fa4_max_kv_sequence_length = attn_configs.max_seq_len
            logging.info(
                "MiniMax-M3 target verify attention backend: %s "
                "(requested=%s, kv_cache_dtype=%s)",
                self._backend,
                requested_backend,
                self.kv_cache_dtype,
            )

        def _resolve_verify_tokens(
            self, attn_inputs, is_cuda_graph_replay: bool
        ) -> int:
            if self._verify_tokens is None:
                self._verify_tokens = _target_verify_width(attn_inputs)
            elif is_cuda_graph_replay:
                _validate_target_verify_replay_shape(attn_inputs, self._verify_tokens)
            return self._verify_tokens

        def _build_flashinfer_token_rows(
            self, attn_inputs, is_cuda_graph_replay: bool = False
        ):
            prefix_lengths = attn_inputs.prefix_lengths
            block_table = attn_inputs.kv_cache_kernel_block_id_device
            input_lengths = attn_inputs.input_lengths
            if prefix_lengths is None or block_table is None or input_lengths is None:
                raise RuntimeError("MiniMax-M3 target verify metadata is incomplete")
            verify_tokens = self._resolve_verify_tokens(
                attn_inputs, is_cuda_graph_replay
            )
            valid_requests = input_lengths > 0 if is_cuda_graph_replay else None
            return _expand_target_verify_rows(
                prefix_lengths,
                block_table,
                verify_tokens,
                valid_requests,
            )

        def _kv_dtype(self, attn_inputs):
            if self.kv_cache_dtype == KvCacheDataType.INT8:
                return torch.int8
            if self.kv_cache_dtype == KvCacheDataType.FP8:
                return torch.float8_e4m3fn
            return get_scalar_type(attn_inputs.dtype)

        def _plan_decode_wrapper(self, attn_inputs) -> None:
            self.decode_wrapper.plan(
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.page_indice_d,
                self.fmha_params.paged_kv_last_page_len_d,
                self.local_head_num,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.seq_size_per_block,
                q_data_type=_target_verify_query_dtype(
                    self.kv_cache_dtype,
                    get_scalar_type(attn_inputs.dtype),
                    self._backend,
                ),
                kv_data_type=self._kv_dtype(attn_inputs),
                o_data_type=get_scalar_type(attn_inputs.dtype),
            )

        def _prepare_fa4_paged_attention_metadata(
            self, attn_inputs, is_cuda_graph_replay: bool = False
        ) -> None:
            verify_tokens = self._resolve_verify_tokens(
                attn_inputs, is_cuda_graph_replay
            )
            prefix_lengths = attn_inputs.prefix_lengths
            input_lengths = attn_inputs.input_lengths
            if prefix_lengths is None or input_lengths is None:
                raise RuntimeError("MiniMax-M3 target verify metadata is incomplete")
            physical_block_table = _require_target_verify_physical_block_table(
                attn_inputs
            )
            request_capacity = int(prefix_lengths.numel())
            metadata = self._fa4_metadata_by_request_capacity.get(request_capacity)
            if metadata is None:
                metadata = _TargetVerifyFA4Metadata(
                    cu_seqlens_q=torch.arange(
                        0,
                        (request_capacity + 1) * verify_tokens,
                        verify_tokens,
                        dtype=torch.int32,
                        device=prefix_lengths.device,
                    ),
                    kv_sequence_lengths=torch.empty_like(
                        prefix_lengths, dtype=torch.int32
                    ),
                    physical_block_table=physical_block_table,
                )
                self._fa4_metadata_by_request_capacity[request_capacity] = metadata
            elif (
                metadata.physical_block_table.data_ptr()
                != physical_block_table.data_ptr()
            ):
                raise RuntimeError(
                    "MiniMax-M3 target verify FA4 block-table storage changed "
                    f"for request capacity {request_capacity}"
                )

            _fill_target_verify_compact_lengths(
                prefix_lengths,
                input_lengths,
                verify_tokens,
                metadata.kv_sequence_lengths,
                is_cuda_graph_replay,
            )
            self._active_fa4_metadata = metadata

        def prepare(self, attn_inputs):
            if self._backend == "fa4":
                self._prepare_fa4_paged_attention_metadata(attn_inputs)
                return self.fmha_params

            sequence_lengths_plus_1, block_table = self._build_flashinfer_token_rows(
                attn_inputs
            )
            self.fmha_params.fill_params_mha_device(
                torch.empty(
                    0, dtype=torch.int32, device=sequence_lengths_plus_1.device
                ),
                sequence_lengths_plus_1 - 1,
                torch.ones_like(sequence_lengths_plus_1),
                block_table,
                self.seq_size_per_block,
            )
            self._enable_cuda_graph_wrapper(attn_inputs)
            self._plan_decode_wrapper(attn_inputs)
            return self.fmha_params

        def prepare_for_cuda_graph_replay(self, attn_inputs):
            if self._backend == "fa4":
                self._prepare_fa4_paged_attention_metadata(
                    attn_inputs, is_cuda_graph_replay=True
                )
                return

            fill_decode = getattr(
                self.fmha_params, "fill_decode_cuda_graph_params", None
            )
            if not callable(fill_decode):
                raise RuntimeError(
                    "MiniMax-M3 target verify CUDA Graph requires "
                    "fill_decode_cuda_graph_params"
                )
            sequence_lengths_plus_1, block_table = self._build_flashinfer_token_rows(
                attn_inputs, is_cuda_graph_replay=True
            )
            fill_decode(
                sequence_lengths_plus_1,
                block_table,
                self.seq_size_per_block,
            )
            # Capture uses MAX_SEQ_LEN, but FlashInfer's split schedule is not
            # a reusable upper bound: chunk mappings, merge indptr, and the
            # valid-CTA mask describe the exact planned KV lengths. Refresh it
            # after updating the stable paged-KV metadata for this replay.
            self._plan_decode_wrapper(attn_inputs)

        def forward(self, query, kv_cache, params):
            query = query.reshape(query.shape[0], self.local_head_num, self.head_dim_qk)
            query_dtype = _target_verify_query_dtype(
                self.kv_cache_dtype, query.dtype, self._backend
            )
            if query.dtype != query_dtype:
                query = query.to(query_dtype)
            if self._backend == "flashinfer":
                return super().forward(query, kv_cache, params)

            assert kv_cache is not None, "kv_cache is required"
            paged_kv_cache = kv_cache.kv_cache_base
            if paged_kv_cache is not None and paged_kv_cache.dim() == 2:
                paged_kv_cache = common.reshape_paged_kv_cache(
                    paged_kv_cache,
                    self.local_kv_head_num,
                    self.seq_size_per_block,
                    self.head_dim_qk,
                )
            if paged_kv_cache is None:
                raise RuntimeError("MiniMax-M3 target verify requires paged KV cache")
            metadata = self._active_fa4_metadata
            if metadata is None:
                raise RuntimeError(
                    "MiniMax-M3 FA4 target verify metadata was not prepared"
                )
            key_cache = paged_kv_cache[:, 0].transpose(1, 2)
            value_cache = paged_kv_cache[:, 1].transpose(1, 2)
            attention_output = fa4_varlen_func(
                query,
                key_cache,
                value_cache,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_q=self._verify_tokens,
                max_seqlen_k=self._fa4_max_kv_sequence_length,
                seqused_k=metadata.kv_sequence_lengths,
                page_table=metadata.physical_block_table,
                causal=True,
                softmax_scale=self.head_dim_qk**-0.5,
                num_splits=0,
            )
            return (
                attention_output[0]
                if isinstance(attention_output, tuple)
                else attention_output
            )

    class MiniMaxM3TargetVerifyImpl(PyFlashinferDecodeImpl):
        def _create_fmha_impl(self, attn_configs):
            return MiniMaxM3TargetVerifyAttnOp(attn_configs)

        def __init__(self, attn_configs, attn_inputs, parallelism_config=None):
            super().__init__(attn_configs, attn_inputs, parallelism_config)
            if self.need_rope_kv_cache:
                self.rope_kvcache_impl = FusedRopeKVCachePrefillOpQNoTransposeOut(
                    attn_configs
                )
                self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

        def _refresh_rope_kv_offset(self, attn_inputs):
            if not self.need_rope_kv_cache or self.rope_kvcache_impl is None:
                return
            # Other RoPE fields already reference the graph-owned attention
            # buffers updated by CudaGraphRunner. KV offset is different: it is
            # produced by a conversion kernel, so its captured storage must be
            # refreshed explicitly without rebuilding host scalar metadata.
            _update_target_verify_rope_kv_offset(
                self.rope_params,
                attn_inputs.kv_cache_kernel_block_id_device,
            )

        def prepare_cuda_graph(self, attn_inputs):
            self.fmha_impl.prepare_for_cuda_graph_replay(attn_inputs)
            self._refresh_rope_kv_offset(attn_inputs)

    return MiniMaxM3TargetVerifyImpl


class _MiniMaxM3ModelMixin:
    def __init__(self, model_config: ModelConfig, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        self._mtp_target_hidden_layer_ids = tuple(
            getattr(
                model_config,
                "_minimax_m3_eagle3_aux_hidden_state_layer_ids",
                (),
            )
        )
        self._mtp_target_hidden_layer_slots = {
            layer_id: slot
            for slot, layer_id in enumerate(self._mtp_target_hidden_layer_ids)
        }
        self._mtp_target_hidden_states: Optional[torch.Tensor] = None
        if self._mtp_target_hidden_layer_ids and (
            any(
                layer_id < 0 or layer_id > self.layer_num
                for layer_id in self._mtp_target_hidden_layer_ids
            )
            or int(model_config.hc_mult) != len(self._mtp_target_hidden_layer_ids)
        ):
            raise ValueError(
                "invalid MiniMax-M3 EAGLE3 target hidden-state contract: "
                f"layers={self._mtp_target_hidden_layer_ids}, "
                f"hc_mult={model_config.hc_mult}, model_layers={self.layer_num}"
            )

    def _begin_mtp_target_hidden_capture(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self._mtp_target_hidden_layer_ids:
            return None
        capture = hidden_states.new_empty(
            hidden_states.size(0),
            hidden_states.size(1) * len(self._mtp_target_hidden_layer_ids),
        )
        initial_slot = self._mtp_target_hidden_layer_slots.get(0)
        if initial_slot is not None:
            capture.narrow(
                1,
                initial_slot * hidden_states.size(1),
                hidden_states.size(1),
            ).copy_(hidden_states)
        return capture

    def _capture_mtp_target_hidden(
        self,
        capture: torch.Tensor,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        slot = self._mtp_target_hidden_layer_slots.get(layer_id)
        if slot is None:
            return
        torch.add(
            hidden_states,
            residual,
            out=capture.narrow(
                1,
                slot * hidden_states.size(1),
                hidden_states.size(1),
            ),
        )

    def _finish_mtp_target_hidden_capture(self, capture: torch.Tensor) -> None:
        self._mtp_target_hidden_states = capture

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        hidden_states = self._mtp_target_hidden_states
        if hidden_states is None or num_tokens < 0:
            return hidden_states
        if num_tokens > hidden_states.size(0):
            raise RuntimeError(
                "requested more MiniMax-M3 EAGLE3 hidden rows than produced: "
                f"requested={num_tokens}, available={hidden_states.size(0)}"
            )
        return hidden_states.narrow(0, 0, num_tokens)

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        attn_inputs = inputs.attention_inputs
        if attn_inputs is None or not bool(
            getattr(attn_inputs, "is_target_verify", False)
        ):
            return super().prepare_fmha_impl(inputs, is_cuda_graph)

        target_verify_impl = _target_verify_impl_class()
        attn_inputs.is_cuda_graph = is_cuda_graph
        attn_configs = self.config.getAttentionConfigs(1)
        return target_verify_impl(attn_configs, attn_inputs, self.parallelism_config)


class MiniMaxM3Model(_MiniMaxM3ModelMixin, GenericMoeModel):
    decoder_layer_cls = MiniMaxM3DecoderLayer


class MiniMaxM3MultimodalModel(_MiniMaxM3ModelMixin, MultimodalGenericModel):
    decoder_layer_cls = MiniMaxM3DecoderLayer
