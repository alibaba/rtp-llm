import logging
from typing import Any, Optional

import torch
from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules import AttnImplFactory
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

        ## (batch_size -> fmha_params)
        self.params_dict: dict[int, Any] = {}

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = len(self.kv_cache.kv_cache_base_by_layer)
            layer0_shape = (
                self.kv_cache.kv_cache_base_by_layer[0].shape
                if num_layers > 0
                and self.kv_cache.kv_cache_base_by_layer[0] is not None
                else None
            )
            num_scale_layers = len(self.kv_cache.kv_scale_base_by_layer)
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shape={layer0_shape}, "
                f"num_scale_layers={num_scale_layers}, "
            )
        return True

    ## for cuda graph attn kernel params' fill
    def fill_params(
        self,
        sequence_lengths: Tensor,
        input_lengths: Tensor,
        kv_cache_block_id_host: Tensor,
        replay_batch_size: int,
        capture_batch_size: int,
        seq_size_per_block: int,
    ):
        assert capture_batch_size in self.params_dict
        params_ptr = self.params_dict[capture_batch_size]
        assert params_ptr is not None
        params_ptr.fillParams(
            sequence_lengths,
            input_lengths,
            kv_cache_block_id_host,
            replay_batch_size,
            seq_size_per_block,
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )
        return fmha_impl

    def get_inputs_embeds(self, input_ids: Tensor, inputs: PyModelInputs) -> Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        return self.apply_input_embeddings(inputs_embeds, inputs)

    def apply_input_embeddings(
        self, inputs_embeds: Tensor, inputs: PyModelInputs
    ) -> Tensor:
        if inputs.input_embeddings is not None and len(inputs.input_embeddings) > 0:
            locs = inputs.input_embeddings_locs
            if locs is None:
                raise ValueError("input_embeddings_locs must be set")
            if inputs_embeds.dim() != 2:
                raise ValueError(
                    "inputs_embeds must be a 2D tensor of shape [tokens, hidden_size]"
                )
            if locs.device.type == "cpu" and locs.dtype in (
                torch.int32,
                torch.int64,
            ):
                loc_values = locs.view(-1).tolist()
            else:
                loc_values = locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            if len(inputs.input_embeddings) != len(loc_values):
                raise ValueError(
                    f"input_embeddings count ({len(inputs.input_embeddings)}) "
                    f"!= input_embeddings_locs count ({len(loc_values)})"
                )
            token_num = inputs_embeds.size(0)
            hidden_size = inputs_embeds.size(1)
            normalized_embeddings = []
            previous_end = 0
            for i, (emb, loc) in enumerate(zip(inputs.input_embeddings, loc_values)):
                if loc < 0:
                    raise ValueError(f"input_embeddings_locs[{i}]={loc} must be >= 0")
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if emb.dim() != 2:
                    raise ValueError(
                        f"input_embeddings[{i}] must be 1D or 2D, got dim={emb.dim()}"
                    )
                if not emb.is_floating_point():
                    raise ValueError(
                        f"input_embeddings[{i}] must be floating point, got dtype={emb.dtype}"
                    )
                if emb.size(0) <= 0:
                    raise ValueError(f"input_embeddings[{i}] must not be empty")
                if emb.size(1) != hidden_size:
                    raise ValueError(
                        f"input_embeddings[{i}] hidden size {emb.size(1)} "
                        f"!= model hidden size {hidden_size}"
                    )
                emb_len = emb.size(0)
                if loc + emb_len > token_num:
                    raise ValueError(
                        f"input_embeddings[{i}] at loc {loc} with length {emb_len} "
                        f"exceeds token count {token_num}"
                    )
                if loc < previous_end:
                    raise ValueError(
                        f"input_embeddings_locs[{i}]={loc} overlaps or is out of order; "
                        f"previous interval ends at {previous_end}"
                    )
                normalized_embeddings.append((loc, emb))
                previous_end = loc + emb_len
            for loc, emb in normalized_embeddings:
                emb_len = emb.size(0)
                target = inputs_embeds.narrow(0, loc, emb_len)
                if emb.device == target.device:
                    target.copy_(emb)
                else:
                    target.copy_(
                        emb.to(
                            device=target.device,
                            dtype=target.dtype,
                        )
                    )
        return inputs_embeds

    @staticmethod
    def _has_input_embeddings(inputs: PyModelInputs) -> bool:
        return inputs.input_embeddings is not None and len(inputs.input_embeddings) > 0

    def _reject_input_embeddings(self, inputs: PyModelInputs) -> None:
        if self._has_input_embeddings(inputs):
            raise RuntimeError(
                f"{type(self).__name__} does not support input_embeddings."
            )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
