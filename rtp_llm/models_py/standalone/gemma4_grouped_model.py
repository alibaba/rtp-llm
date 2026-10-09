from collections.abc import Sequence

import torch
from rtp_llm.models_py.standalone.auto_model import AutoModel
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


class GroupedKVCacheAdapter:
    def __init__(
        self,
        layer_tensors: Sequence[torch.Tensor],
        layer_tags: Sequence[str],
        seq_size_per_block: int,
    ) -> None:
        if len(layer_tensors) != len(layer_tags):
            raise ValueError("layer tensors and tags must align")
        self._layer_tensors = tuple(layer_tensors)
        self._layer_tags = tuple(str(tag) for tag in layer_tags)
        self._seq_size_per_block = int(seq_size_per_block)

    def get_layer_cache(self, layer_id: int) -> LayerKVCache:
        return LayerKVCache(
            self._layer_tensors[layer_id],
            self._seq_size_per_block,
            layer_id=layer_id,
            tag=self._layer_tags[layer_id],
        )

    def get_layer_cache_groups(self, layer_id: int) -> list[LayerKVCache]:
        return [self.get_layer_cache(layer_id)]

    def get_seq_size_per_block(self, tag: str) -> int:
        return self._seq_size_per_block

    def get_kernel_seq_size_per_block(self, tag: str) -> int:
        return self._seq_size_per_block


class Gemma4GroupedAutoModel(AutoModel):
    def _init_kv_cache(self) -> None:
        self.layer_num = self.model_config.num_layers
        self.tokens_per_block = self.model_config.attn_config.tokens_per_block
        self.group_block_nums = {}
        self.group_active_tail_blocks = {}
        self.group_kv_bytes = {}
        self.bounded_group_tags = set()
        layer_tensors = []
        layer_tags = []
        for layer_id, descs in enumerate(self.model_config.kv_cache_spec_descs):
            if len(descs) != 1:
                raise ValueError(f"Gemma4 layer {layer_id} expects exactly one KV spec")
            desc = descs[0]
            kv_head_num = desc.kv_head_num
            size_per_head = desc.size_per_head
            if kv_head_num is None or size_per_head is None:
                raise ValueError(
                    f"Gemma4 layer {layer_id} requires explicit KV geometry"
                )
            block_nums = self.block_nums
            capacity = getattr(desc, "capacity", None)
            if capacity is not None and bool(capacity.bounded_by_active_tail):
                tail = getattr(desc, "tail", None)
                active_tail = getattr(tail, "active_tail_blocks", None)
                active_tail_blocks = int(active_tail) if active_tail is not None else 0
                if active_tail_blocks <= 0:
                    raise ValueError(
                        f"bounded Gemma4 group {desc.tag} requires an active tail"
                    )
                growth_blocks = max(
                    int(self.py_env_configs.runtime_config.max_block_size_per_item),
                    0,
                )
                block_nums = 1 + active_tail_blocks + 2 * growth_blocks
                self.group_active_tail_blocks[desc.tag] = active_tail_blocks
                self.bounded_group_tags.add(desc.tag)
            previous = self.group_block_nums.setdefault(desc.tag, block_nums)
            if previous != block_nums:
                raise ValueError(
                    f"Gemma4 group {desc.tag} has inconsistent block counts"
                )
            layer_tensor = torch.zeros(
                block_nums,
                2,
                kv_head_num,
                self.tokens_per_block,
                size_per_head,
                dtype=self.compute_dtype,
                device=self.device,
            )
            layer_tensors.append(layer_tensor)
            self.group_kv_bytes[desc.tag] = self.group_kv_bytes.get(desc.tag, 0) + (
                layer_tensor.numel() * layer_tensor.element_size()
            )
            layer_tags.append(desc.tag)
        self.kv_cache = GroupedKVCacheAdapter(
            layer_tensors, layer_tags, self.tokens_per_block
        )

    def _prepare_group_attention_inputs(
        self, template: PyAttentionInputs
    ) -> dict[str, PyAttentionInputs]:
        def clone_inputs(source: PyAttentionInputs) -> PyAttentionInputs:
            result = PyAttentionInputs()
            result.is_prefill = source.is_prefill
            result.dtype = source.dtype
            for field in (
                "input_lengths",
                "prefix_lengths",
                "sequence_lengths",
                "cu_seqlens_device",
                "cu_kv_seqlens_device",
                "padding_offset",
                "kv_cache_block_id_device",
                "kv_cache_kernel_block_id_device",
                "kv_cache_block_id",
                "kv_cache_kernel_block_id",
            ):
                value = getattr(source, field, None)
                if value is not None:
                    setattr(result, field, value)
            result.context_total_kv_length = source.context_total_kv_length
            return result

        tags = {
            desc.tag
            for descs in self.model_config.kv_cache_spec_descs
            for desc in descs
        }
        grouped_inputs = {tag: clone_inputs(template) for tag in sorted(tags)}
        for tag in self.bounded_group_tags:
            source_table = template.kv_cache_block_id
            if source_table.size(0) != 1:
                raise ValueError("bounded standalone Gemma4 supports batch size 1")
            logical_blocks = source_table.size(1)
            resident_blocks = self.group_active_tail_blocks[tag]
            usable_blocks = self.group_block_nums[tag] - 1
            resident_start = max(0, logical_blocks - resident_blocks)
            block_ids = [-1] * logical_blocks
            for logical_block in range(resident_start, logical_blocks):
                block_ids[logical_block] = logical_block % usable_blocks + 1
            block_table = torch.tensor([block_ids], dtype=torch.int32)
            grouped_inputs[tag].kv_cache_block_id = block_table
            grouped_inputs[tag].kv_cache_kernel_block_id = block_table
            grouped_inputs[tag].kv_cache_block_id_device = block_table.to(self.device)
            grouped_inputs[tag].kv_cache_kernel_block_id_device = grouped_inputs[
                tag
            ].kv_cache_block_id_device
        return grouped_inputs

    def forward_with_logits(self, model_inputs) -> tuple[torch.Tensor, torch.Tensor]:
        model_outputs = self.model.forward(model_inputs)
        hidden_states = model_outputs.hidden_states
        last_hidden = hidden_states[-1:, :]
        logits = torch.matmul(
            last_hidden.to(self.lm_head_weight.dtype), self.lm_head_weight.t()
        ).to(torch.float32)
        cap = float(self.model_config.final_logit_softcapping)
        if cap > 0:
            logits = cap * torch.tanh(logits / cap)
        return logits, hidden_states

    def _sample_next_token(self, model_outputs, sampling_params=None) -> torch.Tensor:
        logits, _ = self.forward_with_logits_from_outputs(model_outputs)
        return torch.argmax(logits, dim=-1)

    def forward_with_logits_from_outputs(
        self, model_outputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = model_outputs.hidden_states
        last_hidden = hidden_states[-1:, :]
        logits = torch.matmul(
            last_hidden.to(self.lm_head_weight.dtype), self.lm_head_weight.t()
        ).to(torch.float32)
        cap = float(self.model_config.final_logit_softcapping)
        if cap > 0:
            logits = cap * torch.tanh(logits / cap)
        return logits, hidden_states
