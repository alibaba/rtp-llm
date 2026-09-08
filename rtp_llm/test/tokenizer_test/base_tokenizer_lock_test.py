import multiprocessing
import tempfile
import time
import unittest
from unittest import mock

_REMOTE_CONFIG = {"auto_map": {"AutoTokenizer": "tokenization_fake.FakeTokenizer"}}


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(text)]


def _load_with_overlap_probe(
    tokenizer_path: str,
    cache_root: str,
    model_config: dict,
    start_barrier,
    active,
    overlap,
    results,
) -> None:
    import transformers
    import transformers.dynamic_module_utils as dynamic_module_utils

    from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import (
        BaseTokenizer,
    )

    def fake_from_pretrained(*args, **kwargs):
        with active.get_lock():
            active.value += 1
            if active.value > 1:
                overlap.value = 1
        time.sleep(0.15)
        with active.get_lock():
            active.value -= 1
        return _FakeTokenizer()

    try:
        start_barrier.wait(timeout=10)
        with mock.patch.object(
            dynamic_module_utils, "HF_MODULES_CACHE", cache_root
        ), mock.patch.object(
            transformers.AutoTokenizer,
            "from_pretrained",
            side_effect=fake_from_pretrained,
        ):
            tokenizer = BaseTokenizer(tokenizer_path, model_config)
            results.put(tokenizer.tokenizer.encode("ready"))
    except BaseException as error:
        results.put((type(error).__name__, str(error)))


class BaseTokenizerRemoteCodeLockTest(unittest.TestCase):
    def _run_pair(self, model_config: dict) -> tuple[list, int]:
        ctx = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as tokenizer_path, tempfile.TemporaryDirectory() as cache_root:
            active = ctx.Value("i", 0)
            overlap = ctx.Value("i", 0)
            barrier = ctx.Barrier(2)
            results = ctx.Queue()
            processes = [
                ctx.Process(
                    target=_load_with_overlap_probe,
                    args=(
                        tokenizer_path,
                        cache_root,
                        model_config,
                        barrier,
                        active,
                        overlap,
                        results,
                    ),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=20)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("tokenizer worker did not finish")
                self.assertEqual(process.exitcode, 0)
            return [results.get(timeout=2) for _ in processes], overlap.value

    def test_remote_code_loads_are_serialized_across_processes(self) -> None:
        results, overlap = self._run_pair(_REMOTE_CONFIG)

        self.assertEqual(results, [[5], [5]])
        self.assertEqual(overlap, 0)

    def test_non_remote_tokenizers_bypass_process_lock(self) -> None:
        results, overlap = self._run_pair({})

        self.assertEqual(results, [[5], [5]])
        self.assertEqual(overlap, 1)

    def test_load_exception_releases_lock(self) -> None:
        import transformers
        import transformers.dynamic_module_utils as dynamic_module_utils

        from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import (
            BaseTokenizer,
        )

        with tempfile.TemporaryDirectory() as tokenizer_path, tempfile.TemporaryDirectory() as cache_root:
            with mock.patch.object(
                dynamic_module_utils, "HF_MODULES_CACHE", cache_root
            ), mock.patch.object(
                transformers.AutoTokenizer,
                "from_pretrained",
                side_effect=[RuntimeError("first load failed"), _FakeTokenizer()],
            ):
                with self.assertRaisesRegex(RuntimeError, "first load failed"):
                    BaseTokenizer(tokenizer_path, _REMOTE_CONFIG)
                tokenizer = BaseTokenizer(tokenizer_path, _REMOTE_CONFIG)

        self.assertEqual(tokenizer.tokenizer.encode("ready"), [5])


if __name__ == "__main__":
    unittest.main()
