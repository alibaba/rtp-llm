"""Shared synthetic input fixtures for graph runner regression tests."""

import copy

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta

GROUP_TAGS = ["full", "aux"]
HIDDEN_SIZE = 4
TOKENS_PER_BLOCK = 8
KERNEL_BLOCK_TABLE_WIDTH = 8


class EmbeddingGraphTestModel:
    def prepare_embedding_graph_attention(self, inputs, is_cuda_graph=True):
        return self.prepare_fmha_impl(inputs, is_cuda_graph)

    def prepare_embedding_graph_replay(self, attention, graph_inputs, request_inputs):
        pass  # These signature-only models have no attention plan.

    def validate_embedding_graph_inputs(self, inputs):
        mm = inputs.multimodal_inputs
        mask = inputs.embedding_inputs.text_tokens_mask
        return (
            not mm.multimodal_features
            and (mm.mm_features_locs is None or mm.mm_features_locs.numel() == 0)
            and (mask is None or mask.numel() == 0)
        )

    def prepare_transformer_inputs(self, inputs):
        return inputs.input_hiddens


def _tag_attention_inputs(
    common: PyAttentionInputs, tags: list[str], values: dict[str, int]
) -> dict[str, PyAttentionInputs]:
    tagged = {}
    for tag in tags:
        tag_inputs = copy.copy(common)
        host_blocks = torch.full_like(
            common.kv_cache_kernel_block_id, values[tag], device="cpu"
        ).pin_memory()
        device_blocks = host_blocks.cuda()
        tag_inputs.kv_cache_kernel_block_id = host_blocks
        tag_inputs.kv_cache_kernel_block_id_device = device_blocks
        tag_inputs.kv_cache_block_id = host_blocks
        tag_inputs.kv_cache_block_id_device = device_blocks
        tagged[tag] = tag_inputs
    return tagged


def _build_common_inputs(
    attention_inputs: PyAttentionInputs,
    tags: list[str],
    values: dict[str, int],
    batch_size: int,
    token_count: int,
    block_count: int,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )

    attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attention_inputs.padding_offset = torch.zeros(
        token_count, dtype=torch.int32, device="cuda"
    )
    attention_inputs.total_tokens = token_count
    attention_inputs.kv_cache_kernel_block_id = torch.zeros(
        (batch_size, block_count), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_kernel_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id.cuda()
    )
    attention_inputs.kv_cache_block_id = attention_inputs.kv_cache_kernel_block_id
    attention_inputs.kv_cache_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id_device
    )
    inputs.attention_inputs = _tag_attention_inputs(attention_inputs, tags, values)
    return inputs


def _build_prefill_inputs(
    tags: list[str], values: dict[str, int], seq_len: int | list[int] = 4
) -> PyModelInputs:
    seq_lens = [seq_len] if isinstance(seq_len, int) else seq_len
    token_count = sum(seq_lens)
    cu_seqlens = [0]
    for length in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = True
    attention_inputs.is_target_verify = False
    attention_inputs.input_lengths = torch.tensor(
        seq_lens, dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.zeros(
        len(seq_lens), dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens = torch.tensor(
        cu_seqlens, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device.clone()
    attention_inputs.context_total_kv_length = token_count
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=len(seq_lens),
        token_count=token_count,
        block_count=max(1, (max(seq_lens) + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK),
    )
