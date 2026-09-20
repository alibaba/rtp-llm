from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
    build_flashmla_device_params,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute,
)
from rtp_llm.ops.compute_ops import LayerKVCache

PAGE_SIZE = 128
NUM_HEADS = 12
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
PACKED_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM + V_HEAD_DIM
EXPANDED_KV_BYTES_PER_TOKEN = NUM_HEADS * PACKED_HEAD_DIM * 2


class DeterministicPackedProjection:
    """Cheap packed projection used to isolate FlashMLA executor behavior."""

    @staticmethod
    def supports_skip_head_mid(
        input_tensor: torch.Tensor, head_splits: Sequence[int]
    ) -> bool:
        return (
            input_tensor.ndim == 2
            and input_tensor.shape[1] == KV_LORA_RANK
            and tuple(head_splits) == (QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM, V_HEAD_DIM)
        )

    def forward_skip_head_mid(
        self,
        input_tensor: torch.Tensor,
        head_splits: Sequence[int],
        *,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.supports_skip_head_mid(input_tensor, head_splits):
            raise ValueError("unexpected packed projection geometry")
        tokens = input_tensor.shape[0]
        if output is None:
            output = input_tensor.new_empty((tokens, NUM_HEADS * PACKED_HEAD_DIM))
        if tuple(output.shape) != (tokens, NUM_HEADS * PACKED_HEAD_DIM):
            raise ValueError("unexpected packed projection output shape")

        packed = output.view(tokens, NUM_HEADS, PACKED_HEAD_DIM)
        packed[..., :QK_NOPE_HEAD_DIM].copy_(input_tensor[:, None, :QK_NOPE_HEAD_DIM])
        packed[..., -V_HEAD_DIM:].copy_(
            input_tensor[:, None, QK_NOPE_HEAD_DIM : QK_NOPE_HEAD_DIM + V_HEAD_DIM]
        )
        return output


@dataclass(frozen=True)
class CaseInputs:
    params: SimpleNamespace
    q: torch.Tensor
    compressed_kv: torch.Tensor
    k_pe: torch.Tensor
    kv_cache: LayerKVCache | None


def _indptr(lengths: Sequence[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + int(length))
    return result


def make_case_inputs(q_lens: Sequence[int], prefix_lens: Sequence[int]) -> CaseInputs:
    if not q_lens or len(q_lens) != len(prefix_lens):
        raise ValueError("q_lens and prefix_lens must have equal nonzero size")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260831 + sum(q_lens) * 17 + sum(prefix_lens) * 31)

    q_tokens = sum(q_lens)
    q = torch.randn(
        (q_tokens, NUM_HEADS, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(0.125)
    compressed_kv = torch.randn(
        (q_tokens, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(0.125)
    k_pe = torch.randn(
        (q_tokens, QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(0.125)

    page_counts = [
        (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE for prefix_len in prefix_lens
    ]
    total_pages = sum(page_counts)
    cache = torch.randn(
        (max(total_pages, 1), PAGE_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).mul_(0.125)
    batch_reuse_info = []
    page_cursor = 0
    for owner, (prefix_len, page_count) in enumerate(
        zip(prefix_lens, page_counts, strict=True)
    ):
        batch_reuse_info.append([owner, int(prefix_len), page_cursor, page_count])
        page_cursor += page_count

    qo_indptr = _indptr(q_lens)
    kv_indptr = _indptr(
        [
            q_len + prefix_len
            for q_len, prefix_len in zip(q_lens, prefix_lens, strict=True)
        ]
    )
    params = SimpleNamespace(
        qo_indptr_h=torch.tensor(qo_indptr, dtype=torch.int32),
        prefill_ragged_kv_len_indptr_h=torch.tensor(kv_indptr, dtype=torch.int32),
        qo_indptr_d=torch.tensor(qo_indptr, dtype=torch.int32, device="cuda"),
        prefill_ragged_kv_len_indptr_d=torch.tensor(
            kv_indptr, dtype=torch.int32, device="cuda"
        ),
        batch_reuse_info_vec_h=torch.tensor(batch_reuse_info, dtype=torch.int32),
        batch_reuse_info_vec_d=torch.tensor(
            batch_reuse_info, dtype=torch.int32, device="cuda"
        ),
    )
    kv_cache = None
    if total_pages:
        kv_cache = LayerKVCache()
        kv_cache.kv_cache_base = cache
    return CaseInputs(params, q, compressed_kv, k_pe, kv_cache)


def make_direct_case_inputs(
    q_lens: Sequence[int],
    prefix_lens: Sequence[int],
    *,
    strided_k_pe: bool = False,
) -> CaseInputs:
    """Build the direct-attention metadata used by the production planner."""

    inputs = make_case_inputs(q_lens, prefix_lens)
    q_lens = tuple(q_lens)
    prefix_lens = tuple(prefix_lens)
    max_blocks = max(
        (q_len + prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
        for q_len, prefix_len in zip(q_lens, prefix_lens, strict=True)
    )
    block_table = torch.zeros(
        (len(q_lens), max_blocks), dtype=torch.int32, device="cuda"
    )
    page_cursor = 0
    for owner, prefix_len in enumerate(prefix_lens):
        page_count = (prefix_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_table[owner, :page_count] = torch.arange(
            page_cursor,
            page_cursor + page_count,
            dtype=torch.int32,
            device="cuda",
        )
        page_cursor += page_count

    max_q_len = max(q_lens)
    padding_offset = [
        owner * max_q_len - sum(q_lens[:owner])
        for owner, q_len in enumerate(q_lens)
        for _ in range(q_len)
    ]
    attn_inputs = SimpleNamespace(
        is_prefill=True,
        total_tokens=sum(q_lens),
        input_lengths_host=torch.tensor(q_lens, dtype=torch.int32),
        prefix_lengths_host=torch.tensor(prefix_lens, dtype=torch.int32),
        input_lengths=torch.tensor(q_lens, dtype=torch.int32, device="cuda"),
        prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
        cu_seqlens=inputs.params.qo_indptr_d,
        cu_kv_seqlens=inputs.params.prefill_ragged_kv_len_indptr_d,
        padding_offset=torch.tensor(padding_offset, dtype=torch.int32, device="cuda"),
        kv_cache_kernel_block_id_device_by_group=[block_table],
        kv_cache_kernel_block_id_device=block_table,
    )
    params = build_flashmla_device_params(attn_inputs, PAGE_SIZE)
    k_pe = inputs.k_pe
    if strided_k_pe:
        k_pe_storage = torch.empty(
            (sum(q_lens), k_pe.shape[1] + 17),
            dtype=k_pe.dtype,
            device=k_pe.device,
        )
        strided_k_pe_view = k_pe_storage.narrow(1, 11, k_pe.shape[1])
        strided_k_pe_view.copy_(k_pe)
        k_pe = strided_k_pe_view
    return CaseInputs(
        params,
        inputs.q,
        inputs.compressed_kv,
        k_pe,
        inputs.kv_cache,
    )


def make_page_rr_rank_inputs(
    inputs: CaseInputs,
    *,
    shard_size: int,
    shard_rank: int,
) -> CaseInputs:
    """Shard a replicated physical-page cache for one Page-RR TP rank."""

    if inputs.kv_cache is None:
        raise ValueError("Page-RR test inputs require a prefix cache")
    prefix_lens = tuple(inputs.params.prefix_lens_host)
    source = inputs.kv_cache.kv_cache_base
    canonical_table = inputs.params.attn_inputs.kv_cache_kernel_block_id_device
    width = max(
        1,
        max(
            (length + PAGE_SIZE * shard_size - 1) // (PAGE_SIZE * shard_size)
            for length in prefix_lens
        ),
    )
    local_cache_storage = source.new_full(
        (len(prefix_lens) * width + 1, PAGE_SIZE, source.shape[-1]), -91
    )
    local_table = canonical_table.new_full((len(prefix_lens), width), -1)
    for request, length in enumerate(prefix_lens):
        for local_page, global_page in enumerate(
            range(
                shard_rank,
                (length + PAGE_SIZE - 1) // PAGE_SIZE,
                shard_size,
            )
        ):
            block = local_cache_storage.shape[0] - 1 - request * width - local_page
            local_table[request, local_page] = block
            local_cache_storage[block].copy_(
                source[canonical_table[request, global_page]]
            )

    local_attn_inputs = SimpleNamespace(**vars(inputs.params.attn_inputs))
    local_attn_inputs.kv_cache_kernel_block_id_device = local_table
    local_attn_inputs.kv_cache_kernel_block_id_device_by_group = [local_table]
    local_cache = LayerKVCache()
    local_cache.kv_cache_base = local_cache_storage
    return CaseInputs(
        build_flashmla_device_params(local_attn_inputs, PAGE_SIZE),
        inputs.q,
        inputs.compressed_kv,
        inputs.k_pe,
        local_cache,
    )


def make_op(
    *,
    expanded_kv_capacity_tokens: int,
    page_rr_cache_adapter=None,
) -> MlaFlashMLAPrefillOp:
    return MlaFlashMLAPrefillOp(
        num_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kernel_page_tokens=PAGE_SIZE,
        prefix_chunk_alignment_tokens=PAGE_SIZE,
        softmax_extra_scale=1.0,
        use_mla=True,
        weights=[{}],
        page_rr_cache_adapter=page_rr_cache_adapter,
        expanded_kv_budget_gib=(
            expanded_kv_capacity_tokens * EXPANDED_KV_BYTES_PER_TOKEN / 1024**3
        ),
    )


def call_op(
    op: MlaFlashMLAPrefillOp,
    inputs: Any,
) -> torch.Tensor:
    return op.forward(
        inputs.q,
        inputs.compressed_kv,
        inputs.k_pe,
        inputs.kv_cache,
        0,
    )


def output_and_lse(
    op: MlaFlashMLAPrefillOp,
    inputs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if op._forward_plan.route is FlashMLAForwardRoute.HYBRID:
        output = call_op(op, inputs)
        torch.cuda.synchronize()
        return output.clone(), op._forward_workspace.canonical_lse.clone()

    original = op._run_dense_attention
    captured: list[torch.Tensor] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        # Production FULL forwards intentionally skip LSE. This numerical
        # helper opts back in so FULL and HYBRID states can still be compared.
        kwargs["return_lse"] = True
        result = original(*args, **kwargs)
        captured.append(result[1])
        return result

    op._run_dense_attention = capture
    try:
        output = call_op(op, inputs)
        torch.cuda.synchronize()
    finally:
        op._run_dense_attention = original
    return output.clone(), captured[-1].clone()
