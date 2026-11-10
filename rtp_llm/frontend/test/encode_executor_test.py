import asyncio
import threading
import unittest

from rtp_llm.pipeline.encode_executor import BoundedEncodeExecutor


class EchoTokenizer:
    def encode(self, prompt):
        return [len(prompt), sum(ord(ch) for ch in prompt) % 997]


class BarrierTokenizer:
    """Both calls must overlap; a serial executor would hit the barrier timeout."""

    def __init__(self):
        self.barrier = threading.Barrier(2)

    def encode(self, prompt):
        self.barrier.wait(timeout=2.0)
        return [len(prompt), sum(ord(ch) for ch in prompt) % 997]


class BoundedEncodeExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def test_two_encodes_overlap_and_heartbeat_runs(self):
        pool = BoundedEncodeExecutor(max_workers=2)
        self.addCleanup(pool.shutdown)
        tokenizer = BarrierTokenizer()

        heartbeat = asyncio.Event()

        async def beat():
            await asyncio.sleep(0.01)
            heartbeat.set()

        encodes = [
            asyncio.create_task(pool.run(tokenizer.encode, "a" * 100)),
            asyncio.create_task(pool.run(tokenizer.encode, "b" * 200)),
        ]
        beat_task = asyncio.create_task(beat())
        try:
            await asyncio.wait_for(heartbeat.wait(), timeout=1.0)
            results = await asyncio.wait_for(asyncio.gather(*encodes), timeout=3.0)
        finally:
            await beat_task
        self.assertEqual(results, [[100, 727], [200, 657]])

    async def test_concurrent_token_ids_equal_serial(self):
        pool = BoundedEncodeExecutor(max_workers=2)
        self.addCleanup(pool.shutdown)
        tokenizer = EchoTokenizer()
        prompts = ["x" * 52, "y" * 200, "z" * 490]
        serial = [tokenizer.encode(prompt) for prompt in prompts]
        concurrent = await asyncio.gather(
            *(pool.run(tokenizer.encode, prompt) for prompt in prompts)
        )
        self.assertEqual(concurrent, serial)

    async def test_shutdown_rejects_new_work(self):
        pool = BoundedEncodeExecutor(max_workers=1)
        pool.shutdown()
        with self.assertRaises(RuntimeError):
            await pool.run(EchoTokenizer().encode, "closed")

    async def test_exception_propagates(self):
        pool = BoundedEncodeExecutor(max_workers=1)
        self.addCleanup(pool.shutdown)

        def fail():
            raise ValueError("encode failed")

        with self.assertRaisesRegex(ValueError, "encode failed"):
            await pool.run(fail)

