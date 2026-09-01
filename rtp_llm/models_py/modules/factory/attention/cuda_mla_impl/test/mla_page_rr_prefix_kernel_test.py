"""Bit-exact and execution-contract tests for the fused prefix CUDA path."""

import subprocess
import sys
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
    _restore_mla_page_rr_prefix,
)


def prefix_fixture(prefix_lens, page_tokens, shard_size, rank, features=576):
    """Build canonical rows independently, then scatter to reversed physical IDs."""
    width = max(
        (n + page_tokens * shard_size - 1) // (page_tokens * shard_size)
        for n in prefix_lens
    )
    blocks = len(prefix_lens) * width + 1
    cache = torch.full((blocks, page_tokens, features), -91, dtype=torch.bfloat16)
    table = torch.full((len(prefix_lens), width), -1, dtype=torch.int64)
    canonical = []
    for request, length in enumerate(prefix_lens):
        rows = (
            ((torch.arange(length * features) + request * 37) % 113)
            .reshape(length, features)
            .to(torch.bfloat16)
        )
        canonical.append(rows)
        for local_page, global_page in enumerate(
            range(rank, (length + page_tokens - 1) // page_tokens, shard_size)
        ):
            block = blocks - 1 - request * width - local_page
            table[request, local_page] = block
            start = global_page * page_tokens
            count = min(page_tokens, length - start)
            cache[block, :count].copy_(rows[start : start + count])
    return cache, table, torch.cat(canonical)


def pack(cache, table, lengths, page_tokens, shards, rank):
    adapter = MlaPageRRCacheAdapter(page_tokens, shards, rank)
    payload = adapter._pack_prefix(cache, table, lengths)
    return payload, adapter._prefix_descriptor


def expected_payload(canonical, lengths, page_tokens, shards, rank):
    pages = []
    for rows in canonical.split(lengths):
        padding = (-len(rows)) % (page_tokens * shards)
        padded = torch.nn.functional.pad(rows, (0, 0, 0, padding))
        pages.append(padded.reshape(-1, shards, page_tokens, rows.shape[1])[:, rank])
    return torch.cat(pages)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
class MlaPageRRPrefixKernelTest(unittest.TestCase):
    def test_size_one_token_dimension_allows_large_stride(self):
        # Run in a child because a regression aborts its CUDA context even though
        # the underlying cache storage is only four bytes.
        result = subprocess.run(
            [sys.executable, __file__, "--large-stride"],
            text=True,
            capture_output=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fused_pack_and_restore_each_launch_one_kernel(self):
        lengths = (1025, 129, 0)
        cache, table, _ = prefix_fixture(lengths, 128, 8, 0)
        cache, table = cache.cuda(), table.cuda()
        payload, descriptor = pack(cache, table, lengths, 128, 8, 0)
        gathered = torch.stack([payload] * 8)
        for name, operation in (
            ("pack", lambda: pack(cache, table, lengths, 128, 8, 0)),
            ("restore", lambda: _restore_mla_page_rr_prefix(gathered, descriptor)),
        ):
            operation()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as trace:
                operation()
                torch.cuda.synchronize()
            kernels = [
                e
                for e in trace.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
                and not e.name.startswith("Memcpy")
                and not e.name.startswith("Memset")
            ]
            with self.subTest(operation=name):
                self.assertEqual(len(kernels), 1, [e.name for e in kernels])

    def test_all_owners_match_cpu_reference_and_zero_padding(self):
        for page_tokens, shards in (
            (4, 2),
            (64, 4),
            (128, 8),
            (256, 2),
            (256, 4),
            (256, 8),
        ):
            stripe = page_tokens * shards
            lengths = (
                0,
                1,
                page_tokens - 1,
                page_tokens,
                page_tokens + 1,
                stripe - 1,
                stripe,
                stripe + 1,
                2 * stripe + 1,
            )
            batches = []
            for rank in range(shards):
                cache, table, canonical = prefix_fixture(
                    lengths, page_tokens, shards, rank
                )
                reference = expected_payload(
                    canonical, lengths, page_tokens, shards, rank
                )
                actual, descriptor = pack(
                    cache.cuda(), table.cuda(), lengths, page_tokens, shards, rank
                )
                torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
                batches.append(actual)
            restored = _restore_mla_page_rr_prefix(torch.stack(batches), descriptor)
            torch.testing.assert_close(restored.cpu(), canonical, rtol=0, atol=0)

    def test_strided_cache_and_table_preserve_features(self):
        lengths = (17, 3)
        batches = []
        for rank in range(2):
            cache, table, canonical = prefix_fixture(lengths, 4, 2, rank, 6)
            storage = torch.empty(
                (*cache.shape[:2], 2 * cache.shape[2]),
                dtype=cache.dtype,
                device="cuda",
            )
            strided_cache = storage[..., ::2]
            strided_cache.copy_(cache)
            cache = strided_cache
            table = table.t().contiguous().t().cuda()
            payload, descriptor = pack(cache, table, lengths, 4, 2, rank)
            batches.append(payload)
        restored = _restore_mla_page_rr_prefix(torch.stack(batches), descriptor)
        torch.testing.assert_close(restored.cpu(), canonical, rtol=0, atol=0)

    def test_layer_reuse_avoids_metadata_upload_and_refreshes_changed_prefix(self):
        lengths = (17, 3)
        cache, table, _ = prefix_fixture(lengths, 4, 2, 0, 3)
        cache, table = cache.cuda(), table.cuda()
        adapter = MlaPageRRCacheAdapter(4, 2, 0)
        adapter._pack_prefix(cache, table, lengths)
        cache.fill_(7)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as trace:
            batch = adapter._pack_prefix(cache, table, lengths)
            torch.cuda.synchronize()
        uploads = [
            e.name
            for e in trace.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and e.name.startswith("Memcpy HtoD")
        ]
        self.assertEqual(uploads, [])
        expected = torch.tensor(
            [
                [[7] * 3] * 4,
                [[7] * 3] * 4,
                [[7] * 3] + [[0] * 3] * 3,
                [[7] * 3] * 3 + [[0] * 3],
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(batch.cpu(), expected, rtol=0, atol=0)
        changed = adapter._pack_prefix(cache, table, (1, 0))
        self.assertEqual(changed.shape, (1, 4, 3))
        torch.testing.assert_close(
            changed[0, 0].cpu(), torch.full((3,), 7, dtype=torch.bfloat16)
        )
        self.assertEqual(torch.count_nonzero(changed[0, 1:]).item(), 0)

    def test_descriptor_metadata_is_ordered_across_streams(self):
        lengths = (13, 3)
        cpu_cache, cpu_table, canonical = prefix_fixture(lengths, 4, 2, 0, 3)
        gathered = torch.stack(
            [expected_payload(canonical, lengths, 4, 2, rank) for rank in range(2)]
        ).cuda()
        cache, table = cpu_cache.cuda(), cpu_table.cuda()
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            torch.cuda._sleep(20_000_000)
            payload, descriptor = pack(cache, table, lengths, 4, 2, 0)
        # Gathered data is already available on this stream; only the descriptor
        # upload depends on producer, and restore must wait for that upload itself.
        restored = _restore_mla_page_rr_prefix(gathered, descriptor)
        torch.testing.assert_close(restored.cpu(), canonical, rtol=0, atol=0)

    def test_empty_and_non_owner_empty_table_are_zero(self):
        for lengths, rank, expected_pages in (((0, 0), 0, 0), ((1, 0), 7, 1)):
            payload, descriptor = pack(
                torch.full((1, 128, 3), 99, device="cuda", dtype=torch.bfloat16),
                torch.empty((2, 0), device="cuda", dtype=torch.int32),
                lengths,
                128,
                8,
                rank,
            )
            self.assertEqual(payload.shape[0], expected_pages)
            self.assertEqual(torch.count_nonzero(payload).item(), 0)

    def test_current_stream_and_graph_replay_observe_updated_cache(self):
        lengths = (13, 3)
        cache, table, _ = prefix_fixture(lengths, 4, 2, 0, 3)
        cache, table = cache.cuda(), table.cuda()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warm the compiler before capture. The graph must contain no host sync.
            payload, warm_descriptor = pack(cache, table, lengths, 4, 2, 0)
            gathered = torch.stack([payload] * 2)
            _restore_mla_page_rr_prefix(gathered, warm_descriptor)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured, descriptor = pack(cache, table, lengths, 4, 2, 0)
            gathered = torch.stack([captured] * 2)
            restored = _restore_mla_page_rr_prefix(gathered, descriptor)
        with torch.cuda.stream(stream):
            cache.fill_(7)
            graph.replay()
        stream.synchronize()
        expected = torch.full((sum(lengths), 3), 7, dtype=torch.bfloat16)
        torch.testing.assert_close(restored.cpu(), expected, rtol=0, atol=0)

    def test_invalid_owner_blocks_fail_in_isolated_cuda_processes(self):
        for block in (-1, 0, 2):
            result = subprocess.run(
                [sys.executable, __file__, "--invalid-block", str(block)],
                text=True,
                capture_output=True,
                timeout=90,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("assert", result.stderr.lower())


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--large-stride":
        cache = torch.ones((2, 1, 1), device="cuda", dtype=torch.bfloat16)
        cache = cache.as_strided((2, 1, 1), (1, 1 << 20, 1))
        actual, descriptor = pack(
            cache,
            torch.tensor([[1]], device="cuda", dtype=torch.int32),
            (1,),
            1,
            2,
            0,
        )
        torch.testing.assert_close(
            actual.cpu(),
            torch.ones((1, 1, 1), dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
    elif len(sys.argv) == 3 and sys.argv[1] == "--invalid-block":
        pack(
            torch.ones((2, 128, 3), device="cuda", dtype=torch.bfloat16),
            torch.tensor([[int(sys.argv[2])]], device="cuda", dtype=torch.int32),
            (1,),
            128,
            8,
            0,
        )
        torch.cuda.synchronize()
    else:
        unittest.main()
