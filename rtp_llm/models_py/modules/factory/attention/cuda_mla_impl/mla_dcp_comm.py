"""Head-gather / LSE-weighted head-scatter for Decode Page-RR MLA."""

import logging

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather_into,
    all_to_all_single,
    get_process_group,
)


@triton.jit
def _pack_a2a(
    partial, lse, lengths, packed_o, packed_lse, tokens,
    heads: tl.constexpr, local_heads: tl.constexpr, dim: tl.constexpr, block: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    destination = head // local_heads
    output_row = (destination * tokens + token) * local_heads + head % local_heads
    # Each wire row is dim BF16 values followed by one bit-exact FP32 LSE.
    offset = output_row * (dim + 2)
    nonempty = tl.load(lengths + token) > 0
    logsumexp = tl.load(lse + token * heads + head)
    tl.store(packed_lse + offset // 2 + dim // 2, tl.where(nonempty, logsumexp, -float("inf")))
    d = tl.arange(0, block)
    value = tl.load(partial + (token * heads + head) * dim + d, d < dim, other=0)
    tl.store(packed_o + offset + d, tl.where(nonempty, value, 0.0), d < dim)


@triton.jit
def _combine_a2a(
    received_o, received_lse, output, tokens,
    local_heads: tl.constexpr, dim: tl.constexpr, cp_size: tl.constexpr, block: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    offset = (token * local_heads + head) * (dim + 2)
    rank_stride = tokens * local_heads * (dim + 2)
    maximum = tl.full((), -float("inf"), tl.float32)
    for rank in tl.static_range(cp_size):
        lse = tl.load(received_lse + (rank * rank_stride + offset + dim) // 2)
        maximum = tl.maximum(maximum, lse, propagate_nan=tl.PropagateNan.ALL)
    has_keys = maximum != -float("inf")
    denominator = tl.full((), 0.0, tl.float32)
    value_sum = tl.full((block,), 0.0, tl.float32)
    d = tl.arange(0, block)
    for rank in tl.static_range(cp_size):
        lse = tl.load(received_lse + (rank * rank_stride + offset + dim) // 2)
        weight = tl.where(has_keys, tl.exp2(lse - maximum), 0.0)
        value = tl.load(received_o + rank * rank_stride + offset + d, d < dim, other=0).to(tl.float32)
        value = tl.where(lse == -float("inf"), 0.0, value)
        denominator += weight
        value_sum += value * weight
    result = tl.where(has_keys, value_sum / denominator, 0.0)
    tl.store(output + (head * tokens + token) * dim + d, result, d < dim)


class MlaDcpCommunicator:
    """Communication state shared by serial MLA layers in one TP group.

    Q is contiguous [local_heads,T,D]; partial O/LSE are [T,all_heads,L]
    and [T,all_heads]. combine returns BF16 [local_heads,T,L]. LSE is base-2.
    The local MLA adapter supplies O=0/LSE=-inf for empty shards.
    """

    def __init__(self, local_heads, q_dim, latent_dim, dtype, device):
        self.group = get_process_group(Group.TP)
        self.size = torch.distributed.get_world_size(self.group)
        self.rank = torch.distributed.get_rank(self.group)
        self.local_heads = local_heads
        self.heads = local_heads * self.size
        self.q_dim = q_dim
        self.latent_dim = latent_dim
        self.dtype = dtype
        self.device = torch.device(device)
        contract = (local_heads, q_dim, latent_dim, str(dtype))
        contracts = [None] * self.size
        torch.distributed.all_gather_object(contracts, contract, group=self.group)
        if any(value != contract for value in contracts):
            raise ValueError(f"MLA DCP initialization differs across TP ranks: {contracts}")
        if (
            dtype not in (torch.bfloat16, torch.float8_e4m3fn) or latent_dim % 2
            or min(local_heads, q_dim, latent_dim) <= 0
        ):
            raise ValueError("MLA DCP requires BF16/E4M3 Q and an even latent dimension")
        self.backend = "a2a"
        logging.info(
            "[MLA_DCP] backend=a2a tp=%d rank=%d world_rank=%d dp_rank=%d",
            self.size, self.rank, torch.distributed.get_rank(),
            torch.distributed.get_rank() // self.size,
        )

    def query_gather(self, local_q):
        _, tokens, _ = local_q.shape
        if (
            local_q.shape != (self.local_heads, tokens, self.q_dim)
            or not local_q.is_contiguous() or local_q.dtype != self.dtype
            or local_q.device != self.device
        ):
            raise ValueError("MLA DCP Q must be contiguous [local_heads,T,D] on the configured device")
        output = local_q.new_empty(self.heads, tokens, self.q_dim)
        # NCCL copies FP8 codes without needing FP8 reduction support.
        wire_dtype = torch.uint8 if self.dtype == torch.float8_e4m3fn else self.dtype
        all_gather_into(local_q.view(wire_dtype), output.view(wire_dtype), Group.TP)
        return output

    def combine(self, partial_o, partial_lse, local_seq_lens):
        tokens, heads, dim = partial_o.shape
        if (
            heads != self.heads or dim != self.latent_dim
            or partial_o.dtype != torch.bfloat16 or not partial_o.is_contiguous()
            or partial_lse.shape != (tokens, heads) or partial_lse.dtype != torch.float32
            or not partial_lse.is_contiguous() or local_seq_lens.numel() != tokens
            or not local_seq_lens.is_contiguous() or local_seq_lens.dtype != torch.int32
            or not (partial_o.device == partial_lse.device == local_seq_lens.device == self.device)
        ):
            raise ValueError("MLA DCP combine requires BF16 O[T,H,L], FP32 LSE[T,H] and int32 lengths[T]")
        shape = (self.size, tokens, self.local_heads, dim + 2)
        packed = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        _pack_a2a[(tokens, heads)](
            partial_o, partial_lse, local_seq_lens, packed, packed.view(torch.float32),
            tokens, heads=heads, local_heads=self.local_heads, dim=dim,
            block=triton.next_power_of_2(dim),
        )
        received = all_to_all_single(packed, Group.TP)
        output = partial_o.new_empty(self.local_heads, tokens, dim)
        _combine_a2a[(tokens, self.local_heads)](
            received, received.view(torch.float32), output, tokens,
            local_heads=self.local_heads, dim=dim, cp_size=self.size,
            block=triton.next_power_of_2(dim),
        )
        return output


_communicators = {}


def _communicator_key(config, device, dtype):
    return (get_process_group(Group.TP), device, config.head_num,
            config.kv_lora_rank, config.rope_head_dim, dtype)


def get_mla_dcp(config, device, dtype):
    key = _communicator_key(config, device, dtype)
    if key not in _communicators:
        _communicators[key] = MlaDcpCommunicator(
            config.head_num, config.kv_lora_rank + config.rope_head_dim,
            config.kv_lora_rank, dtype, device,
        )
    return _communicators[key]
