from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
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


def make_op(
    *, expanded_kv_capacity_tokens: int, external_prefix_cache: bool = False
) -> MlaFlashMLAPrefillOp:
    return MlaFlashMLAPrefillOp(
        num_heads=NUM_HEADS,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        page_size=PAGE_SIZE,
        softmax_extra_scale=1.0,
        use_mla=True,
        weights=[{}],
        external_prefix_cache=external_prefix_cache,
        expanded_kv_budget_bytes=(
            expanded_kv_capacity_tokens * EXPANDED_KV_BYTES_PER_TOKEN
        ),
    )


def call_op(
    op: MlaFlashMLAPrefillOp,
    inputs: Any,
    canonical_prefix_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    return op.forward(
        inputs.q,
        inputs.compressed_kv,
        inputs.k_pe,
        inputs.kv_cache,
        0,
        canonical_prefix_kv=canonical_prefix_kv,
    )


def output_and_lse(
    op: MlaFlashMLAPrefillOp,
    inputs: Any,
    canonical_prefix_kv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if op._forward_plan.route is FlashMLAForwardRoute.HYBRID:
        output = call_op(op, inputs, canonical_prefix_kv)
        torch.cuda.synchronize()
        return output.clone(), op._forward_workspace.canonical_lse.clone()

    original = op._run_dense_attention
    captured: list[torch.Tensor] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        captured.append(result[1])
        return result

    op._run_dense_attention = capture
    try:
        output = call_op(op, inputs, canonical_prefix_kv)
        torch.cuda.synchronize()
    finally:
        op._run_dense_attention = original
    return output.clone(), captured[-1].clone()
