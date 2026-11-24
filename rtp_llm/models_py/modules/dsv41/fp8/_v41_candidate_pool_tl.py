# SPDX-License-Identifier: Apache-2.0
# Adapted from DeepInfer's candidate_block_max_tl, commit d585e6542c0571176272d703f03c4807ae901284.
"""Persistent 16K-token tasks for length-bounded BF16 candidate pooling.

The upstream task scheduler is retained. Partial query groups, masked output
tails and NaN propagation follow RTP's existing bounded-pool contract.
"""

from typing import Any

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[4, 5],
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
    },
)
def candidate_pool(max_model_len: int, block_size: int, num_sms: int) -> Any:
    num_rows = T.dynamic("num_rows")
    logits_stride = T.dynamic("logits_stride")
    score_stride = T.dynamic("score_stride")
    num_blocks = T.ceildiv(max_model_len, block_size)
    threads = 1024
    rows_per_group = 2
    blocks_per_task = 16384 // block_size
    num_groups = T.ceildiv(num_rows, rows_per_group)
    prep_threads = 128
    prep_warps = prep_threads // 32
    max_tasks = T.ceildiv(max_model_len, 16384)

    @T.prim_func
    def _candidate_pool_persistent(
        logits: T.StridedTensor(
            (num_rows, max_model_len), (logits_stride, 1), T.bfloat16
        ),
        lengths: T.Tensor((num_rows,), T.int32),
        scores: T.StridedTensor((num_rows, num_blocks), (score_stride, 1), T.bfloat16),
        block_lengths: T.Tensor((num_rows,), T.int32),
        task_ids: T.Tensor((num_groups * max_tasks,), T.int32),
        num_tasks: T.Tensor((1,), T.int32),
    ) -> None:
        T.assume(num_rows > 0)
        T.assume(score_stride % 16 == 0)
        T.assume(logits_stride % 16 == 0)
        with T.Kernel(
            T.ceildiv(num_groups, prep_threads), threads=prep_threads
        ) as prep:
            tid = T.get_thread_binding()
            group_lengths = T.alloc_fragment((prep_threads, rows_per_group), T.int32)
            group_max = T.alloc_fragment((prep_threads,), T.int32)
            task_ends = T.alloc_shared((prep_threads,), T.int32)
            offset = T.alloc_var(T.int32, 0)
            T.pdl_sync()
            if prep > 0:
                prefix = T.alloc_reducer((1,), T.int32, op="sum", replication="all")
                T.fill(prefix, 0)
                for chunk in T.serial(prep):
                    for i, r in T.Parallel(prep_threads, rows_per_group):
                        row = (chunk * prep_threads + i) * rows_per_group + r
                        group_lengths[i, r] = T.max(
                            T.min(lengths[row], max_model_len), 0
                        )
                    T.reduce_max(group_lengths, group_max, dim=1)
                    for i in T.Parallel(prep_threads):
                        prefix[0] += T.ceildiv(group_max[i], 16384)
                T.finalize_reducer(prefix)
                offset = prefix[0]
            T.fill(group_lengths, 0)
            for i, r in T.Parallel(prep_threads, rows_per_group):
                row = (prep * prep_threads + i) * rows_per_group + r
                if row < num_rows:
                    group_lengths[i, r] = T.max(T.min(lengths[row], max_model_len), 0)
                    block_lengths[row] = T.ceildiv(group_lengths[i, r], block_size)
            T.reduce_max(group_lengths, group_max, dim=1)
            for i in T.Parallel(prep_threads):
                task_ends[i] = T.ceildiv(group_max[i], 16384)
            T.cumsum(task_ends, dim=0)
            if prep == T.ceildiv(num_groups, prep_threads) - 1 and tid == 0:
                num_tasks[0] = offset + task_ends[prep_threads - 1]
            active = T.min(prep_threads, num_groups - prep * prep_threads)
            for group in T.serial(tid // 32, active, prep_warps):
                start = T.if_then_else(group == 0, 0, task_ends[group - 1])
                count = task_ends[group] - start
                for task in T.serial(tid % 32, count, 32):
                    task_ids[offset + start + task] = (
                        prep * prep_threads + group
                    ) * max_tasks + task

        with T.Kernel(num_sms, threads=threads) as cta:
            values = T.alloc_fragment(
                (rows_per_group, blocks_per_task, block_size), T.bfloat16
            )
            maxima = T.alloc_fragment((rows_per_group, blocks_per_task), T.bfloat16)
            valid = T.alloc_fragment((rows_per_group,), T.int32)
            T.pdl_sync()
            total = num_tasks[0]
            T.assume(total >= 0)
            T.assume(total <= num_groups * max_tasks)
            for iteration in T.serial(cta, total, num_sms):
                task = task_ids[iteration]
                T.assume(task >= 0)
                T.assume(task < num_groups * max_tasks)
                group = task // max_tasks
                tile = task % max_tasks
                row_start = group * rows_per_group
                col_start = tile * blocks_per_task * block_size
                T.copy(
                    logits[
                        row_start : row_start + rows_per_group,
                        col_start : col_start + blocks_per_task * block_size,
                    ],
                    T.reshape(values, (rows_per_group, blocks_per_task * block_size)),
                )
                # CUDA __hmax_nan retains the existing Triton max-with-NaN
                # contract instead of silently dropping a NaN in a pooled block.
                T.reduce_max(values, maxima, dim=2, annotations={"nan_propagate": True})
                T.copy(block_lengths[row_start : row_start + rows_per_group], valid)
                for r, b in T.Parallel(rows_per_group, blocks_per_task):
                    row = row_start + r
                    block = tile * blocks_per_task + b
                    if row < num_rows and block < valid[r]:
                        scores[row, block] = T.if_then_else(
                            block == valid[r] - 1,
                            T.infinity(T.bfloat16),
                            maxima[r, b],
                        )

    return _candidate_pool_persistent
