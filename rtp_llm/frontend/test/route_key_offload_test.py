"""Async regression test: the '/' route key hash and tensor construction must
run OFF the event loop.

backend_rpc_server_visitor hashes the full prompt's block-cache keys and
pipeline.generate_stream builds the token tensor per request. Both are pure
per-request CPU; on the uvicorn event loop they serialize concurrent requests
(the same pathology the 00032 tokenize had). These tests pin:

1. KEY EQUIVALENCE: ``get_block_cache_keys_async`` returns keys bit-identical
   to the sync ``get_block_cache_keys`` (plain path and multimodal-span path;
   lengths including a non-multiple-of-block remainder; zero and non-zero
   seeds) — the async wrapper runs the UNMODIFIED sync function.
2. OFF-LOOP: while a monkeypatched slow (0.3s) key hash runs, a concurrent
   10ms ticker coroutine keeps ticking on the same loop.
3. TENSOR equivalence: the off-loop ``torch.tensor`` construction used in
   ``generate_stream`` matches the inline construction (values, dtype, shape).

The C++ binding is stubbed with the existing pure-Python reference hasher so
the test runs anywhere torch is importable; the real key VALUES are already
pinned by cache_key_routing_test.py.
"""

import asyncio
import time
import unittest
from unittest.mock import patch

import torch

import rtp_llm.ops
import rtp_llm.server.cache_key_routing as cache_key_routing
from rtp_llm.server.cache_key_routing import (
    get_block_cache_keys,
    get_block_cache_keys_async,
)


def _signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >= (1 << (bits - 1)) else value


def _hash_word(previous, value):
    value = _signed(value, 32)
    return _signed(
        previous ^ (value + 0x9E3779B97F4A7C15 + (previous << 12) + (previous >> 32)),
        64,
    )


def _python_cpp_get_block_cache_keys(chunks, initial_hash=0):
    """Deterministic stand-in for the C++ rolling hasher (same shape/values)."""
    keys, rolling = [], initial_hash
    for chunk in chunks:
        for word in chunk:
            rolling = _hash_word(rolling, word)
        keys.append(rolling)
    return keys


class RouteKeyOffloadTest(unittest.TestCase):
    def setUp(self):
        # get_block_cache_keys imports the binding at call time, so patching
        # the module attribute is enough; the sync and async paths see the
        # SAME implementation, which is exactly the equivalence contract.
        patcher = patch.object(
            rtp_llm.ops, "cpp_get_block_cache_keys", _python_cpp_get_block_cache_keys
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_async_keys_identical_to_sync(self):
        tokens = [(i * 997 + 31) % 129280 for i in range(1000)]
        tokens[17:20] = [-1, -(1 << 31), (1 << 31) - 1]
        spans = [(300, [11, 22, 33]), (600, [44, 55])]
        cases = [
            # (token_ids, block, seed, spans)
            (tokens, 128, 0, None),
            (tokens, 128, 0x4453563431494231, None),
            (tokens[:128], 128, 0, None),  # exactly one block
            (tokens[:1000], 128, 0, None),  # 1000 = 7*128 + 104 remainder
            (tokens, 256, 7, spans),
            ([], 128, 0, None),
        ]

        async def run():
            results = []
            for token_ids, block, seed, mm in cases:
                sync = get_block_cache_keys(
                    token_ids, block, cache_key_seed=seed, multimodal_cache_spans=mm
                )
                async_ = await get_block_cache_keys_async(
                    token_ids, block, cache_key_seed=seed, multimodal_cache_spans=mm
                )
                results.append((sync, async_))
            return results

        for sync, async_ in asyncio.run(run()):
            self.assertEqual(sync, async_)

    def test_slow_key_hash_does_not_block_ticker(self):
        tokens = list(range(128 * 20))
        original = cache_key_routing.get_block_cache_keys

        def slow_hash(token_ids, block_size, cache_key_seed=0, multimodal_cache_spans=None):
            time.sleep(0.3)
            return original(
                token_ids,
                block_size,
                cache_key_seed=cache_key_seed,
                multimodal_cache_spans=multimodal_cache_spans,
            )

        async def main():
            ticks = []

            async def ticker():
                for _ in range(50):
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            async def hash_one():
                with patch.object(
                    cache_key_routing, "get_block_cache_keys", slow_hash
                ):
                    return await get_block_cache_keys_async(tokens, 128)

            task = asyncio.create_task(ticker())
            keys = await hash_one()
            await task
            return ticks, keys

        ticks, keys = asyncio.run(main())
        self.assertEqual(len(keys), 20)
        # A synchronous 0.3s hash on the loop would gap the 10ms ticker by
        # ~0.3s; off-loop the maximum gap stays a few scheduler ticks.
        gaps = [
            round(ticks[i + 1] - ticks[i], 3) for i in range(len(ticks) - 1)
        ]
        self.assertEqual([g for g in gaps if g > 0.25], [])
        self.assertGreaterEqual(len(ticks), 40)

    def test_off_loop_tensor_identical_to_inline(self):
        token_ids = [ord(c) % 5000 for c in "off-loop-tensor-check" * 200]

        async def main():
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tok-encode")
            try:
                loop = asyncio.get_running_loop()
                built = await loop.run_in_executor(
                    pool, torch.tensor, token_ids, torch.int
                )
            finally:
                pool.shutdown(wait=False)
            return built

        built = asyncio.run(main())
        inline = torch.tensor(token_ids, dtype=torch.int)
        self.assertEqual(built.dtype, inline.dtype)
        self.assertEqual(tuple(built.shape), tuple(inline.shape))
        self.assertTrue(torch.equal(built, inline))


if __name__ == "__main__":
    unittest.main()
