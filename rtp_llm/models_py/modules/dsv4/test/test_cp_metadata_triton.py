"""Exact-equivalence tests for fused DSV4 CP metadata builders."""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.dsv4 import _cp_metadata_triton
from rtp_llm.models_py.modules.dsv4._cp_metadata_triton import (
    try_build_cp_forward_metadata,
    try_build_cp_full_prefill_positions,
    try_build_cp_restore_indices,
)
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context,
    build_kv_allgather_restore_indices,
    cp_padded_local_kv_len,
)


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


def _restore_reference(lengths: list[int], cp_size: int, block_size: int) -> list[int]:
    local_lens = [
        ((length + cp_size * block_size - 1) // (cp_size * block_size)) * block_size
        for length in lengths
    ]
    total_local = sum(local_lens)
    local_start = 0
    expected: list[int] = []
    for length, local_len in zip(lengths, local_lens):
        for position in range(length):
            block_idx, token_in_block = divmod(position, block_size)
            owner = block_idx % cp_size
            local_block_idx = block_idx // cp_size
            expected.append(
                owner * total_local
                + local_start
                + local_block_idx * block_size
                + token_in_block
            )
        local_start += local_len
    return expected


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


def _cp_forward_reference(
    inputs: dict[str, torch.Tensor | int],
) -> tuple[torch.Tensor, ...]:
    lengths = inputs["lengths"].cpu().tolist()
    chunks = inputs["chunks"].cpu().tolist()
    prefixes = inputs["prefixes"].cpu().tolist()
    padding = inputs["padding"]
    restore = inputs["restore"]
    shuffle = inputs["shuffle"].long()
    relative_parts = []
    global_parts = []
    request_parts = []
    padded_offset = 0
    local_offset = 0
    for request_id, (length, chunk, prefix) in enumerate(
        zip(lengths, chunks, prefixes)
    ):
        request_shuffle = shuffle[local_offset : local_offset + chunk]
        relative_parts.append(request_shuffle + padded_offset)
        global_parts.append(request_shuffle.clamp_max(max(length - 1, 0)) + prefix)
        request_parts.append(
            torch.full((chunk,), request_id, dtype=torch.int32, device="cuda")
        )
        padded_offset += chunk * int(inputs["cp_size"])
        local_offset += chunk
    relative = torch.cat(relative_parts)
    lengths_cuda = inputs["lengths"]
    return (
        relative,
        torch.cat(global_parts),
        torch.cat(request_parts),
        padding[relative] != 0,
        restore[padding != 0].long(),
        torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device="cuda"),
                lengths_cuda.cumsum(0).to(torch.int32),
            ]
        ),
        inputs["prefixes"].long(),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CPMetadataTritonTest(unittest.TestCase):
    def assert_forward_metadata_equal(
        self, actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
            self.assertEqual(actual_tensor.dtype, expected_tensor.dtype)
            self.assertTrue(actual_tensor.is_contiguous())

    def test_forward_b64_runtime_loop_configuration(self):
        self.assertEqual(_cp_metadata_triton._BLOCK_SIZE, 256)
        self.assertEqual(_cp_metadata_triton._FORWARD_B64_LOOP_UNROLL, 2)

    def test_forward_metadata_exact_for_batches_ranks_and_padding(self):
        cases = (
            ([1], [0], 2),
            ([7, 8], [0, 23], 2),
            ([9, 14, 1, 31], [5, 100, 0, 2048], 4),
        )
        for lengths, prefixes, cp_size in cases:
            for cp_rank in range(cp_size):
                with self.subTest(lengths=lengths, cp_size=cp_size, cp_rank=cp_rank):
                    inputs = _cp_forward_inputs(lengths, prefixes, cp_size, cp_rank)
                    actual = try_build_cp_forward_metadata(
                        inputs["lengths"],
                        inputs["chunks"],
                        inputs["prefixes"],
                        inputs["padding"],
                        inputs["restore"],
                        inputs["shuffle"],
                        cp_size=cp_size,
                        cp_rank=cp_rank,
                        chunk_length=inputs["chunk_length"],
                        seq_len_full=inputs["seq_len_full"],
                    )
                    self.assertIsNotNone(actual)
                    self.assert_forward_metadata_equal(
                        actual, _cp_forward_reference(inputs)
                    )

    def test_forward_metadata_exact_for_b32_b64_randomized_varlen(self):
        for batch_size in (32, 64):
            lengths, prefixes = _random_varlen_case(
                batch_size, seed=20260822 + batch_size
            )
            for cp_rank in range(4):
                with self.subTest(batch_size=batch_size, cp_rank=cp_rank):
                    inputs = _cp_forward_inputs(lengths, prefixes, 4, cp_rank)
                    actual = try_build_cp_forward_metadata(
                        inputs["lengths"],
                        inputs["chunks"],
                        inputs["prefixes"],
                        inputs["padding"],
                        inputs["restore"],
                        inputs["shuffle"],
                        cp_size=4,
                        cp_rank=cp_rank,
                        chunk_length=inputs["chunk_length"],
                        seq_len_full=inputs["seq_len_full"],
                    )
                    self.assertIsNotNone(actual)
                    self.assert_forward_metadata_equal(
                        actual, _cp_forward_reference(inputs)
                    )

    def test_forward_metadata_shape_validation_uses_fallback(self):
        inputs = _cp_forward_inputs([9, 14], [0, 100], 2, 0)
        kwargs = dict(
            cp_size=2,
            cp_rank=0,
            chunk_length=inputs["chunk_length"],
            seq_len_full=inputs["seq_len_full"],
        )
        args = (
            inputs["lengths"],
            inputs["chunks"],
            inputs["prefixes"],
            inputs["padding"],
            inputs["restore"],
            inputs["shuffle"],
        )
        self.assertIsNone(
            try_build_cp_forward_metadata(*args[:-1], args[-1][:-1], **kwargs)
        )

    def test_forward_metadata_hot_path_is_one_kernel(self):
        inputs = _cp_forward_inputs(
            [16385, 8193, 4097, 1025], [0, 16384, 32768, 49152], 4, 2
        )

        def run():
            return try_build_cp_forward_metadata(
                inputs["lengths"],
                inputs["chunks"],
                inputs["prefixes"],
                inputs["padding"],
                inputs["restore"],
                inputs["shuffle"],
                cp_size=4,
                cp_rank=2,
                chunk_length=inputs["chunk_length"],
                seq_len_full=inputs["seq_len_full"],
            )

        for _ in range(2):
            self.assertIsNotNone(run())
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            actual = run()
        torch.cuda.synchronize()
        self.assertIsNotNone(actual)
        self.assert_forward_metadata_equal(actual, _cp_forward_reference(inputs))
        events = list(prof.key_averages())
        fused = [
            event for event in events if "_cp_forward_metadata_kernel" in event.key
        ]
        self.assertEqual(sum(event.count for event in fused), 1)
        keys = {event.key for event in events}
        for forbidden in (
            "aten::arange",
            "aten::cat",
            "aten::cumsum",
            "aten::item",
            "aten::_local_scalar_dense",
            "aten::repeat_interleave",
        ):
            self.assertNotIn(forbidden, keys)

    def test_build_cp_context_b64_fused_for_both_cache_topologies(self):
        lengths, prefixes = _random_varlen_case(64, seed=2026082264)
        cp_size = 4
        cp_rank = 3
        inputs = _cp_forward_inputs(lengths, prefixes, cp_size, cp_rank)
        cp_info = SimpleNamespace(
            prefill_qkv_padding_mask=inputs["padding"],
            prefill_qkv_restore_indice=inputs["restore"],
            prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
            prefill_cp_chunk_lengths=inputs["chunks"],
            prefill_shuffle_indices=inputs["shuffle"],
        )
        for kv_cache_sharded in (False, True):
            with self.subTest(kv_cache_sharded=kv_cache_sharded):
                kwargs = dict(
                    cp_info=cp_info,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    chunk_length=inputs["chunk_length"],
                    device=torch.device("cuda"),
                    position_offset=inputs["prefixes"],
                    position_offset_host=torch.tensor(prefixes, dtype=torch.int32),
                    chunk_lengths_device=inputs["chunks"],
                    kv_cache_sharded=kv_cache_sharded,
                )
                with patch.object(
                    _cp_metadata_triton,
                    "try_build_cp_forward_metadata",
                    return_value=None,
                ) as fallback_builder:
                    expected = build_cp_context(**kwargs)
                fallback_builder.assert_called_once()
                for _ in range(2):
                    actual = build_cp_context(**kwargs)
                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    actual = build_cp_context(**kwargs)
                torch.cuda.synchronize()

                for field in (
                    "relative_positions",
                    "global_positions",
                    "req_id_per_token",
                    "prefix_lengths",
                    "local_is_real",
                    "unpad_restore",
                    "input_lengths_global",
                    "cu_seqlens_global",
                ):
                    torch.testing.assert_close(
                        getattr(actual, field),
                        getattr(expected, field),
                        rtol=0,
                        atol=0,
                    )
                for field in (
                    "cp_size",
                    "cp_rank",
                    "chunk_length",
                    "padded_seq_len",
                    "seq_len_full",
                    "prefix_length",
                    "seq_len_total",
                    "chunk_lengths_per_req",
                    "kv_cache_sharded",
                    "input_lengths_global_host",
                    "prefix_lengths_host",
                ):
                    self.assertEqual(getattr(actual, field), getattr(expected, field))
                fused = [
                    event
                    for event in prof.key_averages()
                    if "_cp_forward_metadata_kernel" in event.key
                ]
                self.assertEqual(sum(event.count for event in fused), 1)

    def test_restore_indices_exact_for_shape_matrix(self):
        lengths_b32, _ = _random_varlen_case(32, seed=2026082232)
        lengths_b64, _ = _random_varlen_case(64, seed=2026082264)
        cases = (
            ([1], 2, 1),
            ([17], 2, 4),
            ([20], 4, 4),
            ([8, 12, 4], 2, 4),
            ([8, 0, 4], 2, 4),
            ([0, 9, 33], 4, 8),
            ([1, 2, 3, 31], 4, 2),
            (lengths_b32, 4, 4),
            (lengths_b64, 4, 4),
        )
        for lengths_host, cp_size, block_size in cases:
            with self.subTest(
                lengths=lengths_host, cp_size=cp_size, block_size=block_size
            ):
                lengths = torch.tensor(lengths_host, dtype=torch.int64, device="cuda")
                total_tokens = sum(lengths_host)
                total_local = sum(
                    ((length + cp_size * block_size - 1) // (cp_size * block_size))
                    * block_size
                    for length in lengths_host
                )
                actual = try_build_cp_restore_indices(
                    lengths,
                    cp_size=cp_size,
                    owner_block_size=block_size,
                    total_tokens=total_tokens,
                    total_local_kv=total_local,
                )
                self.assertIsNotNone(actual)
                expected = torch.tensor(
                    _restore_reference(lengths_host, cp_size, block_size),
                    dtype=torch.int64,
                    device="cuda",
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.int64)
                self.assertTrue(actual.is_contiguous())

    def test_full_positions_exact_for_single_and_varlen(self):
        b32 = _random_varlen_case(32, seed=2026082232)
        b64 = _random_varlen_case(64, seed=2026082264)
        for lengths_host, prefixes_host in (
            ([17], [0]),
            ([17], [123]),
            ([8, 0, 14], [10, 40, 100]),
            ([1, 2, 3, 31], [0, 5, 20, 1000]),
            b32,
            b64,
        ):
            with self.subTest(lengths=lengths_host, prefixes=prefixes_host):
                lengths = torch.tensor(lengths_host, dtype=torch.int64, device="cuda")
                prefixes = torch.tensor(prefixes_host, dtype=torch.int64, device="cuda")
                actual = try_build_cp_full_prefill_positions(
                    lengths, prefixes, total_tokens=sum(lengths_host)
                )
                self.assertIsNotNone(actual)
                positions, request_ids = actual
                expected_positions = []
                expected_request_ids = []
                for req_id, (length, prefix) in enumerate(
                    zip(lengths_host, prefixes_host)
                ):
                    expected_positions.extend(range(prefix, prefix + length))
                    expected_request_ids.extend([req_id] * length)
                torch.testing.assert_close(
                    positions,
                    torch.tensor(expected_positions, dtype=torch.int64, device="cuda"),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    request_ids,
                    torch.tensor(
                        expected_request_ids, dtype=torch.int64, device="cuda"
                    ),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(positions.dtype, torch.int64)
                self.assertEqual(request_ids.dtype, torch.int64)
                self.assertTrue(positions.is_contiguous())
                self.assertTrue(request_ids.is_contiguous())

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

        for batch_block in (4, 8, 16, 32, 64):
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

    def test_batch_bound_uses_fallback(self):
        too_many = torch.ones(65, dtype=torch.int64, device="cuda")
        self.assertIsNone(
            try_build_cp_restore_indices(
                too_many,
                cp_size=2,
                owner_block_size=4,
                total_tokens=65,
                total_local_kv=260,
            )
        )

    def test_integrated_builder_matches_fallback_and_is_one_kernel(self):
        lengths_host = [16384, 8192, 0, 4096]
        lengths = torch.tensor(lengths_host, dtype=torch.int64, device="cuda")
        total_tokens = sum(lengths_host)
        total_local = sum(
            cp_padded_local_kv_len(length, 4, 4) for length in lengths_host
        )
        kwargs = dict(
            per_req_total_kv_lens=lengths,
            cp_size=4,
            block_size=4,
            device=torch.device("cuda"),
            total_kv_len=total_tokens,
            total_local_kv=total_local,
        )
        with patch.object(
            _cp_metadata_triton,
            "try_build_cp_restore_indices",
            return_value=None,
        ) as fallback_builder:
            expected = build_kv_allgather_restore_indices(**kwargs)
        fallback_builder.assert_called_once()
        for _ in range(2):
            actual = build_kv_allgather_restore_indices(**kwargs)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            actual = build_kv_allgather_restore_indices(**kwargs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        key_averages = list(prof.key_averages())
        keys = [event.key for event in key_averages]
        self.assertIn("_cp_restore_varlen_kernel", keys)
        fused_events = [
            event for event in key_averages if event.key == "_cp_restore_varlen_kernel"
        ]
        self.assertEqual(len(fused_events), 1)
        self.assertEqual(fused_events[0].count, 1)
        self.assertNotIn("aten::repeat_interleave", keys)
        self.assertNotIn("aten::item", keys)
        self.assertNotIn("aten::_local_scalar_dense", keys)


if __name__ == "__main__":
    unittest.main()
