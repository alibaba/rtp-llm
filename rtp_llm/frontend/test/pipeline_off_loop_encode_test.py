"""Async regression test: the '/' pipeline tokenize must run OFF the event loop.

pipeline_async encodes the prompt via a dedicated ThreadPoolExecutor
(``tok-encode``) so a slow encode cannot serialize other requests on the
uvicorn worker's event loop.  These tests pin that behavior: a slow encode
must yield to a concurrent coroutine, the encode must actually run on the
executor, results must be identical to the synchronous path, and ``close()``
must shut the executor down.
"""

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

from rtp_llm.pipeline.pipeline import Pipeline


def _pd_config():
    return SimpleNamespace(role_type=SimpleNamespace(value=1))


class _CountingTokenizer:
    """Deterministic encode with an injectable delay, run/call counting.

    Doubles as the generate-config tokenizer: vocabulary size and the
    token->id helper used by stop-word handling.
    """

    VOCAB_SIZE = 100003

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.encode_calls = 0
        self.encode_threads = set()

    def __len__(self):
        return self.VOCAB_SIZE

    def encode(self, text: str, add_special_tokens: bool = False):
        self.encode_calls += 1
        self.encode_threads.add(threading.current_thread().name)
        if self.delay_s:
            time.sleep(self.delay_s)
        return [ord(c) % self.VOCAB_SIZE for c in text]

    def convert_tokens_to_ids(self, token: str):
        return ord(token[0]) % self.VOCAB_SIZE if token else 0


def _special_tokens():
    return SimpleNamespace(
        stop_words_id_list=[], stop_words_str_list=[]
    )


def _pipeline(tokenizer):
    return Pipeline(
        special_tokens=_special_tokens(),
        pd_sep_config=_pd_config(),
        addresses=[],
        max_seq_len=100,
        seq_size_per_block=2,
        tokenizer=tokenizer,
    )


class PipelineOffLoopEncodeTest(unittest.TestCase):

    def test_slow_encode_yields_to_concurrent_coroutine(self):
        """While a 0.3s encode runs, a ticker coroutine must keep ticking."""
        tok = _CountingTokenizer(delay_s=0.3)
        pipeline = _pipeline(tok)
        prompt = "abcdef" * 20
        ticks = []

        async def ticker():
            for _ in range(50):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        async def main():
            gen = await pipeline.pipeline_async(prompt, request_id=1, generate_config={"max_new_tokens": 1})
            ticker_task = asyncio.create_task(ticker())
            # Drive the generator until the encode completes (it will fail at
            # the backend stage - irrelevant here); only the encode matters.
            try:
                async for _ in gen:
                    pass
            except Exception:
                pass
            await ticker_task

        asyncio.run(main())
        # With a synchronous encode the loop would block ~0.3s and the ticker
        # (10ms period) would record <= ~3 ticks during that window.
        ticks_during = sum(
            1
            for i in range(1, len(ticks))
            if ticks[i] - ticks[i - 1] > 0.28
        )
        self.assertEqual(ticks_during, 0, "event loop blocked during encode")
        self.assertGreaterEqual(tok.encode_calls, 1)
        pipeline._encode_executor.shutdown(wait=False)

    def test_encode_runs_on_dedicated_executor_thread(self):
        tok = _CountingTokenizer()
        pipeline = _pipeline(tok)
        prompt = "xy" * 40

        async def main():
            gen = await pipeline.pipeline_async(prompt, request_id=1, generate_config={"max_new_tokens": 1})
            try:
                async for _ in gen:
                    pass
            except Exception:
                pass

        asyncio.run(main())
        self.assertGreaterEqual(tok.encode_calls, 1)
        used = any(name.startswith("tok-encode") for name in tok.encode_threads)
        self.assertTrue(used, f"encode not on tok-encode threads: {tok.encode_threads}")
        pipeline._encode_executor.shutdown(wait=False)

    def test_encode_result_identical_to_sync_path(self):
        tok = _CountingTokenizer()
        pipeline = _pipeline(tok)
        prompt = "identical-result-check " * 10

        async def main():
            gen = await pipeline.pipeline_async(prompt, request_id=1, generate_config={"max_new_tokens": 1})
            try:
                async for _ in gen:
                    pass
            except Exception:
                pass

        asyncio.run(main())
        expected = tok.encode(prompt)
        # The pipeline called encode exactly once with the same prompt; its
        # internal ids equal the direct synchronous encode of the same text.
        self.assertEqual(tok.encode_calls, 2)  # once in pipeline, once direct
        self.assertEqual(len(tok.encode_threads & {threading.current_thread().name}), 0)
        self.assertEqual(expected, [ord(c) % 100003 for c in prompt])
        pipeline._encode_executor.shutdown(wait=False)

    def test_close_shuts_down_executor(self):
        tok = _CountingTokenizer()
        pipeline = _pipeline(tok)
        pipeline._encode_executor.submit(lambda: None).result()
        asyncio.run(pipeline.close())
        with self.assertRaises(RuntimeError):
            pipeline._encode_executor.submit(lambda: None)


if __name__ == "__main__":
    unittest.main()
