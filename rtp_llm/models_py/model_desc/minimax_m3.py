import functools
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
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
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


@functools.lru_cache(maxsize=1)
def _target_verify_impl_class():
    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
        PyFlashinferDecodeAttnOp,
        PyFlashinferDecodeImpl,
    )
    from rtp_llm.ops import KvCacheDataType
    from rtp_llm.ops.compute_ops import (
        FusedRopeKVCachePrefillOpQNoTransposeOut,
        get_scalar_type,
    )

    class MiniMaxM3TargetVerifyAttnOp(PyFlashinferDecodeAttnOp):
        def __init__(self, attn_configs):
            super().__init__(attn_configs)
            self._verify_tokens = None

        def _token_rows(self, attn_inputs, mask_padding=False):
            prefix_lengths = attn_inputs.prefix_lengths
            block_table = attn_inputs.kv_cache_kernel_block_id_device
            input_lengths = attn_inputs.input_lengths
            if prefix_lengths is None or block_table is None or input_lengths is None:
                raise RuntimeError("MiniMax-M3 target verify metadata is incomplete")
            if self._verify_tokens is None:
                self._verify_tokens = _target_verify_width(attn_inputs)
            elif mask_padding:
                # Replay keeps request-row tensors at the captured bucket size,
                # while total_tokens describes only live requests. Verify width
                # is a graph invariant and must not be recomputed from both.
                _validate_target_verify_replay_shape(attn_inputs, self._verify_tokens)
            valid_requests = input_lengths > 0 if mask_padding else None
            return _expand_target_verify_rows(
                prefix_lengths,
                block_table,
                self._verify_tokens,
                valid_requests,
            )

        def _kv_dtype(self, attn_inputs):
            if self.kv_cache_dtype == KvCacheDataType.INT8:
                return torch.int8
            if self.kv_cache_dtype == KvCacheDataType.FP8:
                return torch.float8_e4m3fn
            return get_scalar_type(attn_inputs.dtype)

        def prepare(self, attn_inputs):
            sequence_lengths_plus_1, block_table = self._token_rows(attn_inputs)
            self.fmha_params.fill_params_mha_device(
                torch.empty(
                    0, dtype=torch.int32, device=sequence_lengths_plus_1.device
                ),
                sequence_lengths_plus_1 - 1,
                torch.ones_like(sequence_lengths_plus_1),
                block_table,
                self.seq_size_per_block,
            )
            self.decode_wrapper.plan(
                self.fmha_params.decode_page_indptr_d,
                self.fmha_params.page_indice_d,
                self.fmha_params.paged_kv_last_page_len_d,
                self.local_head_num,
                self.local_kv_head_num,
                self.head_dim_qk,
                self.seq_size_per_block,
                q_data_type=get_scalar_type(attn_inputs.dtype),
                kv_data_type=self._kv_dtype(attn_inputs),
            )
            return self.fmha_params

        def prepare_for_cuda_graph_replay(self, attn_inputs):
            fill_decode = getattr(
                self.fmha_params, "fill_decode_cuda_graph_params", None
            )
            if not callable(fill_decode):
                raise RuntimeError(
                    "MiniMax-M3 target verify CUDA Graph requires "
                    "fill_decode_cuda_graph_params"
                )
            sequence_lengths_plus_1, block_table = self._token_rows(
                attn_inputs, mask_padding=True
            )
            fill_decode(
                sequence_lengths_plus_1,
                block_table,
                self.seq_size_per_block,
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
