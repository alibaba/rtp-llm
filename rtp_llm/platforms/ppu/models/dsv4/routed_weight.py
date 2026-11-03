"""Rank-local loading for PPU DSV4's separate packed FP4 expert tensors."""

import torch

from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, PreShardedTensor
from rtp_llm.platforms.ppu.models.dsv4.resources import tp_split_strategies
from rtp_llm.utils.model_weight import identity, stack_


class PpuRoutedWeight(MoeAtomicWeight):
    @classmethod
    def support(cls, quant_config, src_weight_info):
        # Selected explicitly by the PPU model, never by global quant dispatch.
        return False

    def _get_split_func(self):
        return tp_split_strategies()[self.name]

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        if (
            not load_config.moe_pure_tp_mode
            or load_config.merge_lora
            or self.stacked_ckpt_keys
        ):
            return super()._load_raw_tensor(tensor_source, layer_id, device, load_config)
        if (len(self.weights) != 1 or self.weights[0].merge_fun is not identity
                or self.process_fun is not stack_):
            raise ValueError("PPU routed TP loading requires separate expert tensors")

        experts = list(load_config.get_selected_experts(layer_id, self.config.expert_num))
        if not experts:
            raise ValueError("PPU routed TP loading requires at least one expert")
        output = None
        expert_shape = None
        dtype = self.data_type or load_config.compute_dtype
        for index, expert in enumerate(experts):
            name = self._ckpt_name(self.weights[0], layer_id, expert)
            tensor = identity(tensor_source.load_tensor(name, dtype))
            if tensor.ndim != 2 or (expert_shape is not None and tensor.shape != expert_shape):
                raise ValueError(f"PPU routed expert shape mismatch: {name}")
            expert_shape = tensor.shape
            # Slice before device transfer; only rank-local storage lives on PPU.
            shard = super()._split(tensor.unsqueeze(0), load_config)[self.name][0]
            if output is None:
                output = torch.empty((len(experts), *shard.shape), dtype=dtype, device=device)
            output[index].view(torch.uint8).copy_(shard.view(torch.uint8))
            del tensor, shard
        return {self.name: PreShardedTensor(output)}
