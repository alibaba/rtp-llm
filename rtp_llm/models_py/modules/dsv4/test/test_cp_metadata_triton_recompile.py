"""Cache-clear recompile coverage for the fused DSV4 CP metadata kernels.

Split from ``test_cp_metadata_triton`` because clearing the Triton device
caches forces full JIT recompilation of all three metadata kernels, which
dominates wall-clock on the ARM lane. The target is tagged ``manual`` (with
``heavy`` documenting the reason) so the wildcard/PR lanes skip it; run it
explicitly when touching the warmup or specialization contract:

    bazelisk test --config=cuda13 \\
        //rtp_llm/models_py/modules/dsv4/test:test_cp_metadata_triton_recompile

The no-recompile property is proven at the two end batch blocks (4 and 64):
after a cold clear, each block warmup adds exactly one cache entry per
kernel, and the production batch (block - 1) reuses the warm entry without
recompiling. The interior blocks (8, 16, 32) share the same launch shape
family (power-of-two grid), so the endpoints bracket the contract.
"""

from __future__ import annotations

import random
import unittest

import torch

from rtp_llm.models_py.modules.dsv4 import _cp_metadata_triton
from rtp_llm.models_py.modules.dsv4._cp_metadata_triton import (
    try_build_cp_forward_metadata,
    try_build_cp_full_prefill_positions,
    try_build_cp_restore_indices,
)
from rtp_llm.models_py.modules.dsv4.cp import cp_padded_local_kv_len


def _triton_cache_size(kernel_fn) -> int:
    return sum(
        len(kernel_cache) for kernel_cache, *_ in kernel_fn.device_caches.values()
    )


def _random_varlen_case(batch_size: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    lengths = [rng.randrange(0, 48) for _ in range(batch_size)]
    lengths[0] = max(lengths[0], 1)
    prefixes = [rng.randrange(0, 1 << 20) for _ in range(batch_size)]
    return lengths, prefixes


def _cp_forward_inputs(
    lengths: list[int], prefixes: list[int], cp_size: int, cp_rank: int
) -> dict[str, torch.Tensor | int]:
    alignment = 2 * cp_size
    chunks = [((length + alignment - 1) // alignment) * 2 for length in lengths]
    total_chunk = sum(chunks)
    padded_size = cp_size * total_chunk
    padding = torch.zeros(padded_size, dtype=torch.int32)
    restore = torch.empty(padded_size, dtype=torch.int32)
    shuffle_for_rank: list[int] = []
    chunk_offset = 0
    padded_offset = 0
    for length, chunk in zip(lengths, chunks):
        padded = chunk * cp_size
        pair = chunk // 2
        padding[padded_offset : padded_offset + length] = 1
        for rank in range(cp_size):
            rank_shuffle = list(range(rank * pair, (rank + 1) * pair)) + list(
                range(padded - (rank + 1) * pair, padded - rank * pair)
            )
            for local_idx, padded_idx in enumerate(rank_shuffle):
                restore[padded_offset + padded_idx] = (
                    rank * total_chunk + chunk_offset + local_idx
                )
            if rank == cp_rank:
                shuffle_for_rank.extend(rank_shuffle)
        chunk_offset += chunk
        padded_offset += padded

    return {
        "lengths": torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        "chunks": torch.tensor(chunks, dtype=torch.int32, device="cuda"),
        "prefixes": torch.tensor(prefixes, dtype=torch.int32, device="cuda"),
        "padding": padding.cuda(),
        "restore": restore.cuda(),
        "shuffle": torch.tensor(shuffle_for_rank, dtype=torch.int32, device="cuda"),
        "chunk_length": total_chunk,
        "seq_len_full": sum(lengths),
        "cp_size": cp_size,
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CpMetadataTritonRecompileTest(unittest.TestCase):
    def test_power_of_two_warmup_covers_non_power_batches_without_recompile(self):
        kernels = (
            _cp_metadata_triton._cp_restore_varlen_kernel,
            _cp_metadata_triton._cp_positions_varlen_kernel,
            _cp_metadata_triton._cp_forward_metadata_kernel,
        )
        for kernel in kernels:
            kernel.device_caches.clear()

        def launch_all(batch_size: int) -> None:
            lengths_host, prefixes_host = _random_varlen_case(
                batch_size, seed=2026082200 + batch_size
            )
            lengths = torch.tensor(lengths_host, dtype=torch.int64, device="cuda")
            prefixes = torch.tensor(prefixes_host, dtype=torch.int64, device="cuda")
            total_tokens = sum(lengths_host)
            total_local = sum(
                cp_padded_local_kv_len(length, 4, 4) for length in lengths_host
            )
            self.assertIsNotNone(
                try_build_cp_restore_indices(
                    lengths,
                    cp_size=4,
                    owner_block_size=4,
                    total_tokens=total_tokens,
                    total_local_kv=total_local,
                )
            )
            self.assertIsNotNone(
                try_build_cp_full_prefill_positions(
                    lengths, prefixes, total_tokens=total_tokens
                )
            )
            forward_inputs = _cp_forward_inputs(
                lengths_host, prefixes_host, cp_size=4, cp_rank=0
            )
            self.assertIsNotNone(
                try_build_cp_forward_metadata(
                    forward_inputs["lengths"],
                    forward_inputs["chunks"],
                    forward_inputs["prefixes"],
                    forward_inputs["padding"],
                    forward_inputs["restore"],
                    forward_inputs["shuffle"],
                    cp_size=4,
                    cp_rank=0,
                    chunk_length=forward_inputs["chunk_length"],
                    seq_len_full=forward_inputs["seq_len_full"],
                )
            )

        # The no-recompile contract is bracketed by the two endpoint batch
        # blocks; interior powers of two share the launch-shape family.
        for batch_block in (4, 64):
            with self.subTest(batch_block=batch_block):
                cache_before_warmup = tuple(
                    _triton_cache_size(kernel) for kernel in kernels
                )
                launch_all(batch_block)
                torch.cuda.synchronize()
                cache_after_warmup = tuple(
                    _triton_cache_size(kernel) for kernel in kernels
                )
                self.assertEqual(
                    cache_after_warmup,
                    tuple(count + 1 for count in cache_before_warmup),
                )

                production_batch = batch_block - 1
                self.assertEqual(
                    _cp_metadata_triton._batch_block(production_batch),
                    batch_block,
                )
                launch_all(production_batch)
                torch.cuda.synchronize()
                self.assertEqual(
                    tuple(_triton_cache_size(kernel) for kernel in kernels),
                    cache_after_warmup,
                )


if __name__ == "__main__":
    unittest.main()
