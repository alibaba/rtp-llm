import logging
from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    get_primary_attention_inputs,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttentionImpl
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()
        self.nvfp4_lm_head = None
        quant_config = getattr(config, "quant_config", None)
        if quant_config is not None and quant_config.get_method() == "modelopt_mixed":
            from rtp_llm.models_py.modules.factory.linear import LinearFactory

            head = weight.get_global_weight_or_none(W.lm_head)
            if head is not None and head.dtype == torch.uint8:
                self.nvfp4_lm_head = LinearFactory.create_linear(
                    head,
                    None,
                    weight.get_global_weight(W.lm_head_s),
                    quant_config,
                    py_hw_kernel_config,
                    weight_scale_2=weight.get_global_weight(W.lm_head_s2),
                    input_scale=weight.get_global_weight(W.lm_head_i_s),
                )

    def compute_quantized_logits(self, hidden):
        return self._compute_quantized_logits_native(hidden).float()

    def _compute_quantized_logits_native(self, hidden):
        if self.nvfp4_lm_head is None:
            raise ValueError("No quantized LM head was initialized")
        # Preserve the GEMM's BF16 output through TP communication. Promotion
        # to FP32 belongs after gathering the vocabulary shards.
        return self.nvfp4_lm_head(hidden.to(torch.bfloat16))

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = self.kv_cache.layer_count
            layer0_caches = (
                self.kv_cache.get_layer_cache_groups(0) if num_layers > 0 else []
            )
            layer0_shapes = [cache.kv_cache_base.shape for cache in layer0_caches]
            layer0_scale_count = sum(
                cache.kv_scale_base is not None and cache.kv_scale_base.numel() > 0
                for cache in layer0_caches
            )
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shapes={layer0_shapes}, "
                f"layer0_scale_groups={layer0_scale_count}, "
            )
        return True

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> AttentionImpl | dict[str, AttentionImpl]:
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            fmha_group_tags = self._get_fmha_group_tags()
            selected_group_inputs = (
                attention_inputs.items()
                if fmha_group_tags is None
                else (
                    (tag, select_attention_inputs_for_tag(attention_inputs, tag))
                    for tag in fmha_group_tags
                )
            )
            return {
                tag: AttnImplFactory.get_fmha_impl(
                    self.config,
                    self.parallelism_config,
                    self.weight,
                    group_inputs,
                    self.fmha_config,
                    is_cuda_graph,
                )
                for tag, group_inputs in selected_group_inputs
            }
        return AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """Model hook: None means every attention-input tag requires FMHA."""
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")

    def _forward_with_quantized_lm_head(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        outputs = self._forward_quantized_outputs(inputs, fmha_impl)
        return self._postprocess_quantized_outputs(inputs, outputs)

    def _forward_quantized_for_cuda_graph(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        # Capture only the fixed-size computation. Row selection and vocabulary
        # communication use the real request size after replay.
        outputs = self._forward_without_lm_head(inputs, fmha_impl)
        outputs.local_logits = self._compute_quantized_logits_native(
            outputs.hidden_states
        )
        return outputs

    def _forward_quantized_outputs(
        self, inputs: PyModelInputs, fmha_impl: Any = None
    ) -> PyModelOutputs:
        outputs = self._forward_without_lm_head(inputs, fmha_impl)
        hidden = outputs.hidden_states
        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        cp_info = attention_inputs.context_parallel_info
        indexes = inputs.lm_output_indexes
        selected = False
        if cp_info is not None:
            from rtp_llm.models_py.distributed.collective_torch import (
                Group,
                all_gather,
                all_reduce,
            )

            if inputs.need_all_logits or inputs.need_all_hidden_states:
                gathered = all_gather(hidden, Group.TP)
                valid = torch.nonzero(cp_info.prefill_qkv_padding_mask, as_tuple=True)[
                    0
                ]
                restore = cp_info.prefill_qkv_restore_indice.index_select(0, valid)
                hidden = gathered.index_select(0, restore.long())
            else:
                # Reconstruct only requested rows before the vocabulary projection.
                # Every row has exactly one owner, so summing is lossless in BF16.
                if indexes is None or indexes.numel() == 0:
                    raise ValueError("CP LM head requires output token indexes")
                indexes = indexes.to(
                    device=hidden.device, dtype=torch.long, non_blocking=True
                )
                valid = torch.searchsorted(
                    cp_info.prefill_qkv_padding_mask.cumsum(0), indexes + 1
                )
                positions = cp_info.prefill_qkv_restore_indice.index_select(0, valid)
                owners = positions // hidden.shape[0]
                local = hidden.index_select(0, (positions % hidden.shape[0]).long())
                local = torch.where(
                    (owners == self.parallelism_config.tp_rank).unsqueeze(-1),
                    local,
                    torch.zeros_like(local),
                )
                hidden = all_reduce(local, Group.TP)
                selected = True

        last_hidden = hidden
        if (
            not selected
            and not inputs.need_all_logits
            and indexes is not None
            and indexes.numel() > 0
        ):
            last_hidden = hidden.index_select(
                0,
                indexes.to(device=hidden.device, dtype=torch.long, non_blocking=True),
            )
        outputs.hidden_states = hidden
        outputs.last_hidden_states = last_hidden
        outputs.local_logits = self._compute_quantized_logits_native(last_hidden)
        return outputs

    def _postprocess_quantized_outputs(
        self, inputs: PyModelInputs, outputs: PyModelOutputs
    ) -> PyModelOutputs:
        hidden = outputs.hidden_states
        last_hidden = outputs.last_hidden_states
        logits = outputs.local_logits
        indexes = inputs.lm_output_indexes

        if last_hidden is None:
            # Replay returns views into reusable graph buffers. Own the hidden
            # output before the next replay can overwrite those buffers.
            hidden = hidden.clone()
            last_hidden = hidden
            if not inputs.need_all_logits and indexes is not None and indexes.numel():
                indexes = indexes.to(
                    device=hidden.device, dtype=torch.long, non_blocking=True
                )
                last_hidden = hidden.index_select(0, indexes)
                logits = logits.index_select(0, indexes)

        if logits.ndim != 2 or logits.shape[0] != last_hidden.shape[0]:
            raise ValueError("LM head logits must match the selected hidden rows")
        outputs.local_logits = logits
        tp_size = self.parallelism_config.tp_size
        vocab_ids = self.config.output_vocab_ids
        if vocab_ids:
            padded_size = self.config.output_vocab_padded_size
            if (
                padded_size < len(vocab_ids)
                or padded_size % tp_size
                or logits.shape[1] * tp_size != padded_size
            ):
                raise ValueError("Invalid output vocabulary LM head layout")
        if tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

            rows, local_vocab = logits.shape
            # all_gather concatenates ranks along dim 0; the LM head partitions
            # vocabulary columns. Keep BF16 until rank-major shards are joined.
            logits = (
                all_gather(logits.contiguous(), Group.TP)
                .view(tp_size, rows, local_vocab)
                .permute(1, 0, 2)
                .reshape(rows, tp_size * local_vocab)
            )
        if vocab_ids:
            logits = logits[:, : len(vocab_ids)]
        logits = logits.float()

        outputs.hidden_states = hidden
        outputs.last_hidden_states = last_hidden
        if inputs.need_all_logits:
            if indexes is None:
                raise ValueError("All-logits output requires output token indexes")
            outputs.all_logits = logits
            outputs.logits = logits.index_select(
                0,
                indexes.to(device=logits.device, dtype=torch.long, non_blocking=True),
            )
        else:
            outputs.logits = logits
        return outputs


def wrap_quantized_lm_head(model: Optional[nn.Module]) -> Optional[nn.Module]:
    """Install the quantized output contract without replacing the model object."""
    if isinstance(model, GptModelBase) and model.nvfp4_lm_head is not None:
        if model.forward != model._forward_with_quantized_lm_head:
            model._forward_without_lm_head = model.forward
            model.forward = model._forward_with_quantized_lm_head
            model.forward_for_cuda_graph = model._forward_quantized_for_cuda_graph
            model.postprocess_outputs = model._postprocess_quantized_outputs
    return model
