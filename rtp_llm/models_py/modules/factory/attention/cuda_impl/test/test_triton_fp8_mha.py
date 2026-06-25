import importlib.util
import math
import unittest
from pathlib import Path

import torch
import triton

_KERNEL_PATH = Path(__file__).resolve().parents[1] / "triton_fp8_mha_kernels.py"
_KERNEL_SPEC = importlib.util.spec_from_file_location(
    "triton_fp8_mha_kernels_for_test", _KERNEL_PATH
)
_KERNEL_MODULE = importlib.util.module_from_spec(_KERNEL_SPEC)
assert _KERNEL_SPEC is not None and _KERNEL_SPEC.loader is not None
_KERNEL_SPEC.loader.exec_module(_KERNEL_MODULE)

_triton_fp8_paged_mha_kernel = _KERNEL_MODULE._triton_fp8_paged_mha_kernel
_triton_fp8_paged_mha_split_kernel = _KERNEL_MODULE._triton_fp8_paged_mha_split_kernel
_triton_fp8_paged_xqa_split_kernel = _KERNEL_MODULE._triton_fp8_paged_xqa_split_kernel
_triton_fp8_paged_xqa_dot_split_kernel = (
    _KERNEL_MODULE._triton_fp8_paged_xqa_dot_split_kernel
)
_triton_fp8_paged_mha_split_combine_kernel = (
    _KERNEL_MODULE._triton_fp8_paged_mha_split_combine_kernel
)

_FLASHINFER_AVAILABLE = importlib.util.find_spec("flashinfer") is not None


def _make_block_table(sequence_lengths: list[int], block_size: int) -> torch.Tensor:
    max_blocks = max(math.ceil(seq_len / block_size) for seq_len in sequence_lengths)
    block_table = torch.zeros((len(sequence_lengths), max_blocks), dtype=torch.int32)
    block_id = 0
    for batch_idx, seq_len in enumerate(sequence_lengths):
        num_blocks = math.ceil(seq_len / block_size)
        block_table[batch_idx, :num_blocks] = torch.arange(
            block_id, block_id + num_blocks, dtype=torch.int32
        )
        block_id += num_blocks
    return block_table


def _make_cache(
    sequence_lengths: list[int],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_table_cpu = _make_block_table(sequence_lengths, block_size)
    num_blocks = int(block_table_cpu.max().item()) + 1
    k_cache = torch.randn(
        (num_blocks, num_kv_heads, block_size, head_size),
        device="cuda",
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        k_cache = k_cache.to(dtype)
        v_cache = v_cache.to(dtype)
    else:
        k_cache = k_cache.to(dtype)
        v_cache = v_cache.to(dtype)
    scale = (
        torch.rand(
            (num_blocks, 2 * num_kv_heads * block_size),
            device="cuda",
            dtype=torch.float32,
        )
        * 0.03
        + 0.01
    )
    k_flat = scale[:, : num_kv_heads * block_size]
    v_flat = scale[:, num_kv_heads * block_size :]
    k_scale = torch.as_strided(
        k_flat,
        (num_blocks, block_size, num_kv_heads),
        (scale.stride(0), 1, block_size),
    )
    v_scale = torch.as_strided(
        v_flat,
        (num_blocks, block_size, num_kv_heads),
        (scale.stride(0), 1, block_size),
    )
    return block_table_cpu.cuda(), k_cache, v_cache, k_scale, v_scale


def _reference_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: list[int],
    cu_q: torch.Tensor,
    block_size: int,
    causal: bool,
) -> torch.Tensor:
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    kv_group_size = num_heads // num_kv_heads
    softmax_scale = q.shape[2] ** -0.5
    out = torch.empty_like(q)

    block_table_cpu = block_table.cpu()
    cu_q_cpu = cu_q.cpu()
    k_cache_ref = k_cache.to(torch.float32)
    v_cache_ref = v_cache.to(torch.float32)
    for batch_idx, seq_len in enumerate(sequence_lengths):
        q_start = int(cu_q_cpu[batch_idx].item())
        q_end = int(cu_q_cpu[batch_idx + 1].item())
        q_len = q_end - q_start
        context_len = seq_len - q_len
        num_blocks = math.ceil(seq_len / block_size)
        block_ids = block_table_cpu[batch_idx, :num_blocks].to(q.device)
        k_blocks = k_cache_ref[block_ids]
        v_blocks = v_cache_ref[block_ids]
        ks_blocks = k_scale[block_ids]
        vs_blocks = v_scale[block_ids]

        for local_q_pos in range(q_len):
            token_idx = q_start + local_q_pos
            max_kv_pos = context_len + local_q_pos + 1 if causal else seq_len
            for q_head in range(num_heads):
                kv_head = q_head // kv_group_size
                k_seq = (
                    k_blocks[:, kv_head]
                    .reshape(-1, q.shape[2])[:max_kv_pos]
                    .to(torch.float32)
                )
                v_seq = (
                    v_blocks[:, kv_head]
                    .reshape(-1, q.shape[2])[:max_kv_pos]
                    .to(torch.float32)
                )
                ks_seq = ks_blocks[:, :, kv_head].reshape(-1)[:max_kv_pos]
                vs_seq = vs_blocks[:, :, kv_head].reshape(-1)[:max_kv_pos]
                scores = (k_seq * ks_seq[:, None]) @ q[token_idx, q_head].float()
                probs = torch.softmax(scores * softmax_scale, dim=0)
                out[token_idx, q_head] = torch.sum(
                    probs[:, None] * (v_seq * vs_seq[:, None]), dim=0
                )
    return out


class TritonFp8PagedMhaKernelTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(123)

    def _run_prefill_kernel(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_q: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        block_n: int,
        block_aligned: bool,
        out: torch.Tensor | None = None,
        num_warps: int = 2,
        num_stages: int = 2,
        scale_contiguous: bool = True,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(q)
        head_size_padded = triton.next_power_of_2(q.shape[2])
        _triton_fp8_paged_mha_kernel[(q.shape[0], q.shape[1])](
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            out,
            total_tokens=q.shape[0],
            batch_size=seq_lens.numel(),
            num_heads=q.shape[1],
            num_kv_heads=k_cache.shape[1],
            head_size=q.shape[2],
            block_size=block_size,
            block_table_stride=block_table.stride(0),
            q_stride_t=q.stride(0),
            q_stride_h=q.stride(1),
            q_stride_d=q.stride(2),
            k_stride_b=k_cache.stride(0),
            k_stride_h=k_cache.stride(1),
            k_stride_s=k_cache.stride(2),
            k_stride_d=k_cache.stride(3),
            v_stride_b=v_cache.stride(0),
            v_stride_h=v_cache.stride(1),
            v_stride_s=v_cache.stride(2),
            v_stride_d=v_cache.stride(3),
            ks_stride_b=k_scale.stride(0),
            ks_stride_s=k_scale.stride(1),
            ks_stride_h=k_scale.stride(2),
            vs_stride_b=v_scale.stride(0),
            vs_stride_s=v_scale.stride(1),
            vs_stride_h=v_scale.stride(2),
            out_stride_t=out.stride(0),
            out_stride_h=out.stride(1),
            out_stride_d=out.stride(2),
            softmax_scale=q.shape[2] ** -0.5,
            BLOCK_N=block_n,
            HEAD_SIZE_PADDED=head_size_padded,
            BLOCK_ALIGNED=block_aligned,
            SCALE_CONTIGUOUS=scale_contiguous,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    def _launch_split_decode_stage(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_q: torch.Tensor,
        seq_lens: torch.Tensor,
        partial_acc: torch.Tensor,
        partial_m: torch.Tensor,
        partial_l: torch.Tensor,
        block_size: int,
        block_n: int,
        split_size: int,
        block_aligned: bool,
        num_warps: int = 2,
        num_stages: int = 2,
        scale_contiguous: bool = True,
    ) -> None:
        num_splits = partial_m.shape[2]
        head_size_padded = partial_acc.shape[3]
        _triton_fp8_paged_mha_split_kernel[(q.shape[0], q.shape[1], num_splits)](
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            partial_acc,
            partial_m,
            partial_l,
            total_tokens=q.shape[0],
            batch_size=seq_lens.numel(),
            num_heads=q.shape[1],
            num_kv_heads=k_cache.shape[1],
            head_size=q.shape[2],
            block_size=block_size,
            block_table_stride=block_table.stride(0),
            q_stride_t=q.stride(0),
            q_stride_h=q.stride(1),
            q_stride_d=q.stride(2),
            k_stride_b=k_cache.stride(0),
            k_stride_h=k_cache.stride(1),
            k_stride_s=k_cache.stride(2),
            k_stride_d=k_cache.stride(3),
            v_stride_b=v_cache.stride(0),
            v_stride_h=v_cache.stride(1),
            v_stride_s=v_cache.stride(2),
            v_stride_d=v_cache.stride(3),
            ks_stride_b=k_scale.stride(0),
            ks_stride_s=k_scale.stride(1),
            ks_stride_h=k_scale.stride(2),
            vs_stride_b=v_scale.stride(0),
            vs_stride_s=v_scale.stride(1),
            vs_stride_h=v_scale.stride(2),
            partial_stride_t=partial_acc.stride(0),
            partial_stride_h=partial_acc.stride(1),
            partial_stride_s=partial_acc.stride(2),
            partial_stride_d=partial_acc.stride(3),
            stats_stride_t=partial_m.stride(0),
            stats_stride_h=partial_m.stride(1),
            stats_stride_s=partial_m.stride(2),
            softmax_scale=q.shape[2] ** -0.5,
            BLOCK_N=block_n,
            SPLIT_SIZE=split_size,
            HEAD_SIZE_PADDED=head_size_padded,
            DECODE_ONE_TOKEN=True,
            SPLIT_BLOCK_ALIGNED=block_aligned,
            SCALE_CONTIGUOUS=scale_contiguous,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_xqa_split_decode_stage(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        partial_acc: torch.Tensor,
        partial_m: torch.Tensor,
        partial_l: torch.Tensor,
        block_size: int,
        block_n: int,
        split_size: int,
        block_aligned: bool,
        query_group_size: int | None = None,
        num_warps: int = 4,
        num_stages: int = 2,
        scale_contiguous: bool = True,
    ) -> None:
        num_splits = partial_m.shape[2]
        head_size_padded = partial_acc.shape[3]
        if query_group_size is None:
            query_group_size = q.shape[1] // k_cache.shape[1]
        num_q_groups = triton.cdiv(q.shape[1], query_group_size)
        _triton_fp8_paged_xqa_split_kernel[(q.shape[0], num_q_groups, num_splits)](
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            seq_lens,
            partial_acc,
            partial_m,
            partial_l,
            num_heads=q.shape[1],
            num_kv_heads=k_cache.shape[1],
            head_size=q.shape[2],
            block_size=block_size,
            block_table_stride=block_table.stride(0),
            q_stride_t=q.stride(0),
            q_stride_h=q.stride(1),
            q_stride_d=q.stride(2),
            k_stride_b=k_cache.stride(0),
            k_stride_h=k_cache.stride(1),
            k_stride_s=k_cache.stride(2),
            k_stride_d=k_cache.stride(3),
            v_stride_b=v_cache.stride(0),
            v_stride_h=v_cache.stride(1),
            v_stride_s=v_cache.stride(2),
            v_stride_d=v_cache.stride(3),
            ks_stride_b=k_scale.stride(0),
            ks_stride_s=k_scale.stride(1),
            ks_stride_h=k_scale.stride(2),
            vs_stride_b=v_scale.stride(0),
            vs_stride_s=v_scale.stride(1),
            vs_stride_h=v_scale.stride(2),
            partial_stride_t=partial_acc.stride(0),
            partial_stride_h=partial_acc.stride(1),
            partial_stride_s=partial_acc.stride(2),
            partial_stride_d=partial_acc.stride(3),
            stats_stride_t=partial_m.stride(0),
            stats_stride_h=partial_m.stride(1),
            stats_stride_s=partial_m.stride(2),
            softmax_scale=q.shape[2] ** -0.5,
            BLOCK_N=block_n,
            SPLIT_SIZE=split_size,
            HEAD_SIZE_PADDED=head_size_padded,
            SPLIT_BLOCK_ALIGNED=block_aligned,
            SCALE_CONTIGUOUS=scale_contiguous,
            GROUP_SIZE=query_group_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _launch_xqa_dot_split_decode_stage(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        partial_acc: torch.Tensor,
        partial_m: torch.Tensor,
        partial_l: torch.Tensor,
        block_size: int,
        block_n: int,
        split_size: int,
        block_aligned: bool,
        block_m: int = 16,
        num_warps: int = 4,
        num_stages: int = 2,
        scale_contiguous: bool = True,
    ) -> None:
        num_splits = partial_m.shape[2]
        head_size_padded = partial_acc.shape[3]
        _triton_fp8_paged_xqa_dot_split_kernel[
            (q.shape[0], k_cache.shape[1], num_splits)
        ](
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            seq_lens,
            partial_acc,
            partial_m,
            partial_l,
            num_heads=q.shape[1],
            num_kv_heads=k_cache.shape[1],
            head_size=q.shape[2],
            block_size=block_size,
            block_table_stride=block_table.stride(0),
            q_stride_t=q.stride(0),
            q_stride_h=q.stride(1),
            q_stride_d=q.stride(2),
            k_stride_b=k_cache.stride(0),
            k_stride_h=k_cache.stride(1),
            k_stride_s=k_cache.stride(2),
            k_stride_d=k_cache.stride(3),
            v_stride_b=v_cache.stride(0),
            v_stride_h=v_cache.stride(1),
            v_stride_s=v_cache.stride(2),
            v_stride_d=v_cache.stride(3),
            ks_stride_b=k_scale.stride(0),
            ks_stride_s=k_scale.stride(1),
            ks_stride_h=k_scale.stride(2),
            vs_stride_b=v_scale.stride(0),
            vs_stride_s=v_scale.stride(1),
            vs_stride_h=v_scale.stride(2),
            partial_stride_t=partial_acc.stride(0),
            partial_stride_h=partial_acc.stride(1),
            partial_stride_s=partial_acc.stride(2),
            partial_stride_d=partial_acc.stride(3),
            stats_stride_t=partial_m.stride(0),
            stats_stride_h=partial_m.stride(1),
            stats_stride_s=partial_m.stride(2),
            softmax_scale=q.shape[2] ** -0.5,
            BLOCK_N=block_n,
            SPLIT_SIZE=split_size,
            HEAD_SIZE_PADDED=head_size_padded,
            BLOCK_M=block_m,
            SPLIT_BLOCK_ALIGNED=block_aligned,
            SCALE_CONTIGUOUS=scale_contiguous,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def _run_xqa_dot_split_decode_kernel(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        block_n: int,
        split_size: int,
        block_aligned: bool,
    ) -> torch.Tensor:
        num_splits = triton.cdiv(block_table.shape[1] * block_size, split_size)
        head_size_padded = triton.next_power_of_2(q.shape[2])
        partial_acc = torch.empty(
            (q.shape[0], q.shape[1], num_splits, head_size_padded),
            device=q.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (q.shape[0], q.shape[1], num_splits), device=q.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)
        out = torch.empty_like(q)
        self._launch_xqa_dot_split_decode_stage(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            seq_lens,
            partial_acc,
            partial_m,
            partial_l,
            block_size,
            block_n,
            split_size,
            block_aligned,
        )
        self._launch_split_combine_stage(q, partial_acc, partial_m, partial_l, out)
        return out

    def _run_xqa_split_decode_kernel(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        block_n: int,
        split_size: int,
        block_aligned: bool,
        query_group_size: int | None = None,
    ) -> torch.Tensor:
        num_splits = triton.cdiv(block_table.shape[1] * block_size, split_size)
        head_size_padded = triton.next_power_of_2(q.shape[2])
        partial_acc = torch.empty(
            (q.shape[0], q.shape[1], num_splits, head_size_padded),
            device=q.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (q.shape[0], q.shape[1], num_splits), device=q.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)
        out = torch.empty_like(q)
        self._launch_xqa_split_decode_stage(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            seq_lens,
            partial_acc,
            partial_m,
            partial_l,
            block_size,
            block_n,
            split_size,
            block_aligned,
            query_group_size,
        )
        self._launch_split_combine_stage(q, partial_acc, partial_m, partial_l, out)
        return out

    def _run_split_decode_kernel(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_q: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        block_n: int,
        split_size: int,
        block_aligned: bool,
    ) -> torch.Tensor:
        num_splits = triton.cdiv(block_table.shape[1] * block_size, split_size)
        head_size_padded = triton.next_power_of_2(q.shape[2])
        partial_acc = torch.empty(
            (q.shape[0], q.shape[1], num_splits, head_size_padded),
            device=q.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (q.shape[0], q.shape[1], num_splits), device=q.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)
        out = torch.empty_like(q)
        self._launch_split_decode_stage(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            partial_acc,
            partial_m,
            partial_l,
            block_size,
            block_n,
            split_size,
            block_aligned,
        )
        self._launch_split_combine_stage(q, partial_acc, partial_m, partial_l, out)
        return out

    def _launch_split_combine_stage(
        self,
        q: torch.Tensor,
        partial_acc: torch.Tensor,
        partial_m: torch.Tensor,
        partial_l: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        head_size_padded = partial_acc.shape[3]
        _triton_fp8_paged_mha_split_combine_kernel[(q.shape[0], q.shape[1])](
            partial_acc,
            partial_m,
            partial_l,
            out,
            num_splits=partial_m.shape[2],
            head_size=q.shape[2],
            partial_stride_t=partial_acc.stride(0),
            partial_stride_h=partial_acc.stride(1),
            partial_stride_s=partial_acc.stride(2),
            partial_stride_d=partial_acc.stride(3),
            stats_stride_t=partial_m.stride(0),
            stats_stride_h=partial_m.stride(1),
            stats_stride_s=partial_m.stride(2),
            out_stride_t=out.stride(0),
            out_stride_h=out.stride(1),
            out_stride_d=out.stride(2),
            HEAD_SIZE_PADDED=head_size_padded,
        )

    def test_prefill_matches_reference(self) -> None:
        sequence_lengths = [17, 43]
        block_size = 16
        num_heads = 4
        num_kv_heads = 2
        head_size = 32
        cu_q = torch.tensor([0, 17, 60], device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)
        q = torch.randn(
            (sum(sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        block_table, k_cache, v_cache, k_scale, v_scale = _make_cache(
            sequence_lengths, num_kv_heads, head_size, block_size, torch.float8_e4m3fn
        )

        actual = self._run_prefill_kernel(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            block_size,
            block_size,
            True,
        )
        ref = _reference_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            sequence_lengths,
            cu_q,
            block_size,
            causal=True,
        )
        torch.testing.assert_close(actual.float(), ref.float(), rtol=2e-2, atol=2e-2)

    def test_split_decode_matches_reference(self) -> None:
        sequence_lengths = [129, 255, 384]
        block_size = 64
        num_heads = 8
        num_kv_heads = 2
        head_size = 64
        cu_q = torch.arange(len(sequence_lengths) + 1, device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)
        q = torch.randn(
            (len(sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        block_table, k_cache, v_cache, k_scale, v_scale = _make_cache(
            sequence_lengths, num_kv_heads, head_size, block_size, torch.float8_e4m3fn
        )

        actual = self._run_split_decode_kernel(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            block_size,
            128,
            512,
            False,
        )
        ref = _reference_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            sequence_lengths,
            cu_q,
            block_size,
            causal=False,
        )
        torch.testing.assert_close(actual.float(), ref.float(), rtol=2e-2, atol=2e-2)

        dot_actual = self._run_xqa_dot_split_decode_kernel(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            seq_lens,
            block_size,
            64,
            512,
            False,
        )
        torch.testing.assert_close(
            dot_actual.float(), ref.float(), rtol=2e-2, atol=2e-2
        )

    def test_xqa_split_decode_matches_reference(self) -> None:
        sequence_lengths = [129, 255, 384]
        block_size = 64
        num_heads = 8
        num_kv_heads = 2
        head_size = 64
        cu_q = torch.arange(len(sequence_lengths) + 1, device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)
        q = torch.randn(
            (len(sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        block_table, k_cache, v_cache, k_scale, v_scale = _make_cache(
            sequence_lengths, num_kv_heads, head_size, block_size, torch.float8_e4m3fn
        )

        actual = self._run_xqa_split_decode_kernel(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            seq_lens,
            block_size,
            128,
            512,
            False,
            2,
        )
        ref = _reference_attention(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            sequence_lengths,
            cu_q,
            block_size,
            causal=False,
        )
        torch.testing.assert_close(actual.float(), ref.float(), rtol=2e-2, atol=2e-2)

    def test_aligned_path_matches_generic_path(self) -> None:
        sequence_lengths = [256, 512]
        block_size = 64
        num_heads = 8
        num_kv_heads = 2
        head_size = 64
        cu_q = torch.arange(len(sequence_lengths) + 1, device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)
        q = torch.randn(
            (len(sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        block_table, k_cache, v_cache, k_scale, v_scale = _make_cache(
            sequence_lengths, num_kv_heads, head_size, block_size, torch.float8_e4m3fn
        )
        generic = self._run_split_decode_kernel(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            block_size,
            block_size,
            256,
            False,
        )
        aligned = self._run_split_decode_kernel(
            q,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            block_table,
            cu_q,
            seq_lens,
            block_size,
            block_size,
            256,
            True,
        )
        torch.testing.assert_close(aligned.float(), generic.float(), rtol=0, atol=0)

    def test_kernel_micro_benchmark(self) -> None:
        block_size = 64
        num_heads = 32
        num_kv_heads = 8
        head_size = 128

        prefill_sequence_lengths = [512, 512, 512, 512]
        prefill_cu_q = torch.tensor(
            [0, 512, 1024, 1536, 2048], device="cuda", dtype=torch.int32
        )
        prefill_seq_lens = torch.tensor(
            prefill_sequence_lengths, device="cuda", dtype=torch.int32
        )
        prefill_q = torch.randn(
            (sum(prefill_sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        (
            prefill_block_table,
            prefill_k_cache,
            prefill_v_cache,
            prefill_k_scale,
            prefill_v_scale,
        ) = _make_cache(
            prefill_sequence_lengths,
            num_kv_heads,
            head_size,
            block_size,
            torch.float8_e4m3fn,
        )
        prefill_out = torch.empty_like(prefill_q)

        def run_prefill_old_default() -> None:
            self._run_prefill_kernel(
                prefill_q,
                prefill_k_cache,
                prefill_v_cache,
                prefill_k_scale,
                prefill_v_scale,
                prefill_block_table,
                prefill_cu_q,
                prefill_seq_lens,
                block_size,
                block_size,
                False,
                prefill_out,
                4,
                3,
            )

        def run_prefill_optimized() -> None:
            self._run_prefill_kernel(
                prefill_q,
                prefill_k_cache,
                prefill_v_cache,
                prefill_k_scale,
                prefill_v_scale,
                prefill_block_table,
                prefill_cu_q,
                prefill_seq_lens,
                block_size,
                128,
                False,
                prefill_out,
                2,
                2,
            )

        run_prefill_old_default()
        run_prefill_optimized()
        torch.cuda.synchronize()
        prefill_old_ms = triton.testing.do_bench(
            run_prefill_old_default, warmup=10, rep=50
        )
        prefill_optimized_ms = triton.testing.do_bench(
            run_prefill_optimized, warmup=10, rep=50
        )
        print(
            "triton_fp8_prefill_kernel "
            f"old_ms={prefill_old_ms:.6f} "
            f"optimized_ms={prefill_optimized_ms:.6f} "
            f"speedup={prefill_old_ms / prefill_optimized_ms:.4f}x"
        )

        sequence_lengths = [8192, 8192, 8192, 8192]
        old_split_size = 256
        optimized_split_size = 512
        cu_q = torch.arange(len(sequence_lengths) + 1, device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)
        q = torch.randn(
            (len(sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        block_table, k_cache, v_cache, k_scale, v_scale = _make_cache(
            sequence_lengths, num_kv_heads, head_size, block_size, torch.float8_e4m3fn
        )
        num_splits = triton.cdiv(block_table.shape[1] * block_size, old_split_size)
        optimized_num_splits = triton.cdiv(
            block_table.shape[1] * block_size, optimized_split_size
        )
        head_size_padded = triton.next_power_of_2(head_size)
        partial_acc = torch.empty(
            (q.shape[0], num_heads, num_splits, head_size_padded),
            device=q.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (q.shape[0], num_heads, num_splits), device=q.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)
        optimized_partial_acc = torch.empty(
            (q.shape[0], num_heads, optimized_num_splits, head_size_padded),
            device=q.device,
            dtype=torch.float32,
        )
        optimized_partial_m = torch.empty(
            (q.shape[0], num_heads, optimized_num_splits),
            device=q.device,
            dtype=torch.float32,
        )
        optimized_partial_l = torch.empty_like(optimized_partial_m)
        out = torch.empty_like(q)

        def run_old_default() -> None:
            self._launch_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                cu_q,
                seq_lens,
                partial_acc,
                partial_m,
                partial_l,
                block_size,
                block_size,
                old_split_size,
                False,
                4,
                3,
            )

        def run_optimized() -> None:
            self._launch_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                cu_q,
                seq_lens,
                optimized_partial_acc,
                optimized_partial_m,
                optimized_partial_l,
                block_size,
                128,
                optimized_split_size,
                False,
                2,
                2,
            )

        def run_old_combine() -> None:
            self._launch_split_combine_stage(q, partial_acc, partial_m, partial_l, out)

        def run_optimized_combine() -> None:
            self._launch_split_combine_stage(
                q, optimized_partial_acc, optimized_partial_m, optimized_partial_l, out
            )

        run_old_default()
        run_optimized()
        torch.cuda.synchronize()
        old_ms = triton.testing.do_bench(run_old_default, warmup=10, rep=50)
        optimized_ms = triton.testing.do_bench(run_optimized, warmup=10, rep=50)
        old_combine_ms = triton.testing.do_bench(run_old_combine, warmup=10, rep=50)
        optimized_combine_ms = triton.testing.do_bench(
            run_optimized_combine, warmup=10, rep=50
        )
        print(
            "triton_fp8_split_decode_kernel "
            f"old_ms={old_ms:.6f} optimized_ms={optimized_ms:.6f} "
            f"speedup={old_ms / optimized_ms:.4f}x "
            f"old_combine_ms={old_combine_ms:.6f} "
            f"optimized_combine_ms={optimized_combine_ms:.6f}"
        )

        xqa_partial_acc = torch.empty_like(optimized_partial_acc)
        xqa_partial_m = torch.empty_like(optimized_partial_m)
        xqa_partial_l = torch.empty_like(optimized_partial_m)
        xqa_partial_acc_half = torch.empty(
            optimized_partial_acc.shape,
            device=q.device,
            dtype=torch.float16,
        )

        def run_xqa_like_group2() -> None:
            self._launch_xqa_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                128,
                optimized_split_size,
                False,
                2,
                2,
                2,
            )

        def run_xqa_like_group4() -> None:
            self._launch_xqa_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                128,
                optimized_split_size,
                False,
                4,
                4,
                2,
            )

        def run_xqa_dot_64() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                optimized_split_size,
                False,
                16,
                4,
                2,
            )

        def run_xqa_dot_128() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                128,
                optimized_split_size,
                False,
                16,
                4,
                2,
            )

        def run_xqa_dot_256() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                256,
                optimized_split_size,
                False,
                16,
                4,
                2,
            )

        def run_xqa_dot_64_split1024() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                1024,
                False,
                16,
                4,
                2,
            )

        def run_xqa_dot_64_split2048() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                2048,
                False,
                16,
                4,
                2,
            )

        def run_xqa_dot_64_warps8() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                optimized_split_size,
                False,
                16,
                8,
                2,
            )

        def run_xqa_dot_64_warps8_half_acc() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc_half,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                optimized_split_size,
                False,
                16,
                8,
                2,
            )

        def run_xqa_dot_64_warps8_aligned() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                optimized_split_size,
                True,
                16,
                8,
                2,
            )

        def run_xqa_dot_64_stages3() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                optimized_split_size,
                False,
                16,
                4,
                3,
            )

        def run_xqa_dot_64_stages4() -> None:
            self._launch_xqa_dot_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                seq_lens,
                xqa_partial_acc,
                xqa_partial_m,
                xqa_partial_l,
                block_size,
                64,
                optimized_split_size,
                False,
                16,
                4,
                4,
            )

        run_xqa_like_group2()
        run_xqa_like_group4()
        run_xqa_dot_64()
        run_xqa_dot_128()
        run_xqa_dot_256()
        run_xqa_dot_64_split1024()
        run_xqa_dot_64_split2048()
        run_xqa_dot_64_warps8()
        run_xqa_dot_64_warps8_half_acc()
        run_xqa_dot_64_warps8_aligned()
        run_xqa_dot_64_stages3()
        run_xqa_dot_64_stages4()
        torch.cuda.synchronize()
        xqa_group2_ms = triton.testing.do_bench(run_xqa_like_group2, warmup=10, rep=50)
        xqa_group4_ms = triton.testing.do_bench(run_xqa_like_group4, warmup=10, rep=50)
        xqa_dot64_ms = triton.testing.do_bench(run_xqa_dot_64, warmup=10, rep=50)
        xqa_dot128_ms = triton.testing.do_bench(run_xqa_dot_128, warmup=10, rep=50)
        xqa_dot256_ms = triton.testing.do_bench(run_xqa_dot_256, warmup=10, rep=50)
        xqa_dot64_s1024_ms = triton.testing.do_bench(
            run_xqa_dot_64_split1024, warmup=10, rep=50
        )
        xqa_dot64_s2048_ms = triton.testing.do_bench(
            run_xqa_dot_64_split2048, warmup=10, rep=50
        )
        xqa_dot64_warps8_ms = triton.testing.do_bench(
            run_xqa_dot_64_warps8, warmup=10, rep=50
        )
        xqa_dot64_warps8_half_acc_ms = triton.testing.do_bench(
            run_xqa_dot_64_warps8_half_acc, warmup=10, rep=50
        )
        xqa_dot64_warps8_aligned_ms = triton.testing.do_bench(
            run_xqa_dot_64_warps8_aligned, warmup=10, rep=50
        )
        xqa_dot64_stages3_ms = triton.testing.do_bench(
            run_xqa_dot_64_stages3, warmup=10, rep=50
        )
        xqa_dot64_stages4_ms = triton.testing.do_bench(
            run_xqa_dot_64_stages4, warmup=10, rep=50
        )
        xqa_dot_ms = min(
            xqa_dot64_ms,
            xqa_dot128_ms,
            xqa_dot256_ms,
            xqa_dot64_s1024_ms,
            xqa_dot64_s2048_ms,
            xqa_dot64_warps8_ms,
            xqa_dot64_warps8_half_acc_ms,
            xqa_dot64_warps8_aligned_ms,
            xqa_dot64_stages3_ms,
            xqa_dot64_stages4_ms,
        )
        print(
            "triton_fp8_xqa_like_split_decode_kernel "
            f"mha_split_ms={optimized_ms:.6f} "
            f"xqa_like_group2_split_ms={xqa_group2_ms:.6f} "
            f"xqa_like_group2_speedup={optimized_ms / xqa_group2_ms:.4f}x "
            f"xqa_like_group4_split_ms={xqa_group4_ms:.6f} "
            f"xqa_like_group4_speedup={optimized_ms / xqa_group4_ms:.4f}x "
            f"xqa_dot64_split_ms={xqa_dot64_ms:.6f} "
            f"xqa_dot128_split_ms={xqa_dot128_ms:.6f} "
            f"xqa_dot256_split_ms={xqa_dot256_ms:.6f} "
            f"xqa_dot64_s1024_split_ms={xqa_dot64_s1024_ms:.6f} "
            f"xqa_dot64_s2048_split_ms={xqa_dot64_s2048_ms:.6f} "
            f"xqa_dot64_warps8_split_ms={xqa_dot64_warps8_ms:.6f} "
            f"xqa_dot64_warps8_half_acc_split_ms={xqa_dot64_warps8_half_acc_ms:.6f} "
            f"xqa_dot64_warps8_aligned_split_ms={xqa_dot64_warps8_aligned_ms:.6f} "
            f"xqa_dot64_stages3_split_ms={xqa_dot64_stages3_ms:.6f} "
            f"xqa_dot64_stages4_split_ms={xqa_dot64_stages4_ms:.6f} "
            f"xqa_dot_best_split_ms={xqa_dot_ms:.6f} "
            f"xqa_dot_speedup={optimized_ms / xqa_dot_ms:.4f}x"
        )

    @unittest.skipUnless(_FLASHINFER_AVAILABLE, "flashinfer is required")
    def test_flashinfer_mha_decode_benchmark(self) -> None:
        import flashinfer

        block_size = 64
        num_heads = 32
        num_kv_heads = 8
        head_size = 128
        split_size = 512
        sequence_lengths = [8192, 8192, 8192, 8192]
        cu_q = torch.arange(len(sequence_lengths) + 1, device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32)
        q = torch.randn(
            (len(sequence_lengths), num_heads, head_size),
            device="cuda",
            dtype=torch.float16,
        )
        block_table, k_cache, v_cache, k_scale, v_scale = _make_cache(
            sequence_lengths, num_kv_heads, head_size, block_size, torch.float8_e4m3fn
        )
        k_scale.fill_(1.0)
        v_scale.fill_(1.0)

        num_splits = triton.cdiv(block_table.shape[1] * block_size, split_size)
        head_size_padded = triton.next_power_of_2(head_size)
        partial_acc = torch.empty(
            (q.shape[0], num_heads, num_splits, head_size_padded),
            device=q.device,
            dtype=torch.float32,
        )
        partial_m = torch.empty(
            (q.shape[0], num_heads, num_splits), device=q.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)
        triton_out = torch.empty_like(q)

        def run_triton_decode() -> None:
            self._launch_split_decode_stage(
                q,
                k_cache,
                v_cache,
                k_scale,
                v_scale,
                block_table,
                cu_q,
                seq_lens,
                partial_acc,
                partial_m,
                partial_l,
                block_size,
                128,
                split_size,
                False,
                2,
                2,
            )
            self._launch_split_combine_stage(
                q, partial_acc, partial_m, partial_l, triton_out
            )

        blocks_per_seq = math.ceil(sequence_lengths[0] / block_size)
        paged_kv_indptr = torch.arange(
            0,
            (len(sequence_lengths) + 1) * blocks_per_seq,
            blocks_per_seq,
            device="cuda",
            dtype=torch.int32,
        )
        paged_kv_indices = block_table.reshape(-1).contiguous()
        paged_kv_last_page_len = torch.full(
            (len(sequence_lengths),), block_size, device="cuda", dtype=torch.int32
        )
        workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="HND", use_tensor_cores=False
        )
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            sm_scale=head_size**-0.5,
        )
        flashinfer_out = torch.empty_like(q)

        def run_flashinfer_decode() -> None:
            wrapper.run(
                q,
                (k_cache, v_cache),
                out=flashinfer_out,
                k_scale=1.0,
                v_scale=1.0,
            )

        run_triton_decode()
        run_flashinfer_decode()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            triton_out.float(), flashinfer_out.float(), rtol=2e-2, atol=2e-2
        )

        triton_ms = triton.testing.do_bench(run_triton_decode, warmup=10, rep=50)
        flashinfer_ms = triton.testing.do_bench(
            run_flashinfer_decode, warmup=10, rep=50
        )
        print(
            "fp8_decode_mha_kernel_compare "
            f"triton_per_token_head_scale1_ms={triton_ms:.6f} "
            f"flashinfer_mha_fp8_scale1_ms={flashinfer_ms:.6f} "
            f"triton_over_flashinfer={triton_ms / flashinfer_ms:.4f}x"
        )


if __name__ == "__main__":
    unittest.main()
