"""Meta-only expert subdivision before FSDP2 wrapping.

The fixed chunk order and explicit zero-gradient dependency are important:
every distributed rank calls every chunk even if its local tokens hit none of
that chunk's experts. GPU collective ordering still requires a real FSDP test.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ExpertChunk(nn.Module):
    def __init__(self, gate_up, down, expert_begin, *, copy_values=False):
        super().__init__()
        if gate_up.ndim != 3 or down.ndim != 3:
            raise ValueError('Packed expert parameters must be three-dimensional')
        if (gate_up.shape[0] != down.shape[0]
                or gate_up.shape[1] != 2 * down.shape[2]
                or gate_up.shape[2] != down.shape[1]):
            raise ValueError('Packed expert parameter shapes disagree')
        self.expert_begin = expert_begin
        self.expert_count = gate_up.shape[0]
        self.expert_end = expert_begin + self.expert_count
        self.hidden_dim = gate_up.shape[2]
        self.intermediate_dim = down.shape[2]
        # Actual Torch2.13 only permits evenly divisible Shard(i>0) during
        # backward. Flatten expert×output_feature into ordinary axis0 instead;
        # uneven axis0 is supported and balances 72 ranks without padding.
        gate_up = gate_up.reshape(-1, self.hidden_dim)
        down = down.reshape(-1, self.intermediate_dim)
        self.gate_up_proj = nn.Parameter(gate_up.detach().clone() if copy_values else gate_up)
        self.down_proj = nn.Parameter(down.detach().clone() if copy_values else down)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Return individual contributions so the parent preserves the upstream
        # increasing-global-expert index_add order, rather than regrouping sums.
        outputs = []
        gate_up = self.gate_up_proj.view(self.expert_count, 2 * self.intermediate_dim, self.hidden_dim)
        down = self.down_proj.view(self.expert_count, self.hidden_dim, self.intermediate_dim)
        for expert in range(self.expert_begin, self.expert_end):
            token_idx, top_k_pos = torch.where(top_k_index == expert)
            if token_idx.numel() == 0:
                continue
            value = hidden_states[token_idx]
            local = expert - self.expert_begin
            gate, up = F.linear(value, gate_up[local]).chunk(2, dim=-1)
            value = F.linear(F.silu(gate) * up, down[local])
            value = value * top_k_weights[token_idx, top_k_pos, None]
            outputs.append((token_idx, value.to(hidden_states.dtype)))
        # This empty contribution connects both parameters even with no local
        # routing hits, without changing any real output coordinate.
        zero = (self.gate_up_proj[0, 0] + self.down_proj[0, 0]) * 0
        empty = hidden_states.new_empty((0, hidden_states.shape[-1])) + zero
        outputs.append((top_k_index.new_empty((0,), dtype=torch.long), empty))
        return tuple(outputs)


class ChunkedExperts(nn.Module):
    def __init__(self, packed, *, chunk_experts=16, require_meta=True):
        super().__init__()
        gate_up = packed.gate_up_proj; down = packed.down_proj
        if require_meta and (gate_up.device.type != 'meta' or down.device.type != 'meta'):
            raise RuntimeError('Subdivision is permitted only before materializing real model weights')
        experts, double_intermediate, hidden = gate_up.shape
        if type(chunk_experts) is not int or chunk_experts < 1 or experts % chunk_experts:
            raise ValueError('Expert chunks must exactly divide the source expert dimension')
        self.num_experts = experts; self.hidden_dim = hidden
        self.intermediate_dim = double_intermediate // 2
        self.chunk_experts = chunk_experts
        self.chunks = nn.ModuleList([
            ExpertChunk(gate_up[begin:begin + chunk_experts],
                        down[begin:begin + chunk_experts], begin,
                        copy_values=not require_meta)
            for begin in range(0, experts, chunk_experts)])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.ndim != 2 or top_k_index.ndim != 2:
            raise ValueError('Expected flattened tokens and a two-dimensional top-k expert index')
        if top_k_index.shape != top_k_weights.shape or top_k_index.shape[0] != hidden_states.shape[0]:
            raise ValueError('Routing indices, routing probabilities and tokens must align')
        output = torch.zeros_like(hidden_states)
        for chunk in self.chunks:
            for token_idx, contribution in chunk(hidden_states, top_k_index, top_k_weights):
                output.index_add_(0, token_idx, contribution)
        return output


def install_meta_chunks(model, *, chunk_experts=16):
    """Return a source mapping for the streamed loader and the RTP exporter."""
    from qwen38_training_contract import validate_config
    # HF may record the explicit FP32 master construction dtype in the runtime
    # config. The original BF16 checkpoint identity is verified separately.
    validate_config(model.config.to_dict(), allow_fp32_master=True)
    sources = {}
    candidates = [(name, module) for name, module in model.named_modules()
                  if type(module).__name__ == 'Qwen3_5MoeExperts']
    if len(candidates) != 92:
        raise ValueError('Expected exactly the 92 main-policy packed expert modules')
    for prefix, packed in candidates:
        parent_name, attribute = prefix.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        replacement = ChunkedExperts(packed, chunk_experts=chunk_experts, require_meta=True)
        setattr(parent, attribute, replacement)
        for index, chunk in enumerate(replacement.chunks):
            for field in ('gate_up_proj', 'down_proj'):
                name = f'{prefix}.chunks.{index}.{field}'
                sources[name] = {'canonical_name': f'{prefix}.{field}',
                                 'expert_begin': chunk.expert_begin, 'expert_end': chunk.expert_end,
                                 'runtime_shape': list(getattr(chunk, field).shape),
                                 'source_shape': list(getattr(packed, field).shape),
                                 'flattened_expert_feature_axis0': True}
    model._qwen38_expert_sources = sources
    model._qwen38_expert_chunk_size = chunk_experts
    return sources


def fsdp_shard_placement(parameter):
    """Use the runtime-tested ordinary axis0, including uneven row extents."""
    from torch.distributed.tensor import Shard
    if parameter.ndim != 2:
        raise ValueError('Expert chunks must use flattened two-dimensional parameters')
    return Shard(0)


def shard_expert_chunks_before_decoder_wrap(model, fsdp_kwargs):
    from torch.distributed.fsdp import fully_shard
    chunks = [module for module in model.modules() if isinstance(module, ExpertChunk)]
    expected = 92 * (512 // model._qwen38_expert_chunk_size)
    if len(chunks) != expected:
        raise ValueError('Missing or extra expert chunk modules before FSDP wrapping')
    kwargs = dict(fsdp_kwargs, shard_placement_fn=fsdp_shard_placement)
    for chunk in chunks:
        fully_shard(chunk, **kwargs)
        # Empty lists disable explicit/default backward lookahead on supported
        # Torch versions. Validate the actual runtime API before using the path.
        if not hasattr(chunk, 'set_modules_to_backward_prefetch'):
            raise RuntimeError('Torch FSDP2 backward-prefetch control is required')
        chunk.set_modules_to_backward_prefetch([])
        if hasattr(chunk, 'set_modules_to_forward_prefetch'):
            chunk.set_modules_to_forward_prefetch([])
    return len(chunks)
