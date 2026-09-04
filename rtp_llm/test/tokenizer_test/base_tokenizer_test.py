import multiprocessing
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import (
    BaseTokenizer,
    _remote_tokenizer_load_lock,
)


def _hold_lock(tokenizer_path, ready, release):
    with _remote_tokenizer_load_lock(tokenizer_path):
        ready.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test did not release tokenizer lock")


def _wait_for_lock(tokenizer_path, attempting, acquired):
    attempting.set()
    with _remote_tokenizer_load_lock(tokenizer_path):
        acquired.set()


class RemoteTokenizerLoadLockTest(unittest.TestCase):
    def test_remote_code_loading_holds_the_lock(self):
        auto_tokenizer = mock.Mock()
        load_context = mock.MagicMock()

        def from_pretrained(*args, **kwargs):
            self.assertTrue(load_context.__enter__.called)
            self.assertFalse(load_context.__exit__.called)
            return mock.sentinel.tokenizer

        auto_tokenizer.from_pretrained.side_effect = from_pretrained
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = auto_tokenizer

        with tempfile.TemporaryDirectory() as tokenizer_path:
            Path(tokenizer_path, "tokenizer_config.json").write_text(
                '{"auto_map":{"AutoTokenizer":["tokenization_kimi.TikTokenTokenizer",null]}}'
            )
            with mock.patch.dict(sys.modules, {"transformers": transformers}):
                with mock.patch(
                    "rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer."
                    "_remote_tokenizer_load_lock",
                    return_value=load_context,
                ) as load_lock:
                    with mock.patch.object(BaseTokenizer, "_fix_post_processor"):
                        tokenizer = BaseTokenizer(tokenizer_path)

        load_lock.assert_called_once_with(tokenizer_path)
        self.assertIs(tokenizer.tokenizer, mock.sentinel.tokenizer)

    def test_serializes_processes_loading_the_same_tokenizer(self):
        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        release = context.Event()
        attempting = context.Event()
        acquired = context.Event()

        with tempfile.TemporaryDirectory() as tokenizer_path:
            holder = context.Process(
                target=_hold_lock, args=(tokenizer_path, ready, release)
            )
            waiter = context.Process(
                target=_wait_for_lock,
                args=(tokenizer_path, attempting, acquired),
            )
            holder.start()
            self.assertTrue(ready.wait(timeout=10))
            waiter.start()
            self.assertTrue(attempting.wait(timeout=10))
            self.assertFalse(acquired.wait(timeout=0.2))

            release.set()
            self.assertTrue(acquired.wait(timeout=10))
            holder.join(timeout=10)
            waiter.join(timeout=10)

        self.assertEqual(holder.exitcode, 0)
        self.assertEqual(waiter.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
