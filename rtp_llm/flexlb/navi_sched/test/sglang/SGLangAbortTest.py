import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from navi_sched.adapters.sglang.NaviSchedSGLangPlugin import _create_abort_task
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class SGLangAbortTest(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_http_abort_does_not_cancel_reused_ids(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                obj = GenerateReqInput(
                    input_ids=[[1], [2]] if batched else [1],
                    rid=["first", "second"] if batched else "first",
                )
                obj.normalize_batch_and_arguments()
                items = [obj[i] for i in range(2)] if batched else [obj]
                manager = SimpleNamespace(
                    rid_to_state={
                        item.rid: SimpleNamespace(obj=item) for item in items
                    },
                    abort_request=Mock(),
                )

                async def reuse_ids(delay):
                    self.assertEqual(delay, 2)
                    manager.rid_to_state = {
                        item.rid: SimpleNamespace(obj=GenerateReqInput(rid=item.rid))
                        for item in items
                    }

                task = _create_abort_task(
                    TokenizerManager.create_abort_task, manager, obj
                )
                with patch("asyncio.sleep", reuse_ids):
                    await task()
                manager.abort_request.assert_not_called()

    async def test_delayed_http_abort_cancels_only_remaining_original_items(self):
        obj = GenerateReqInput(input_ids=[[1], [2]], rid=["first", "second"])
        obj.normalize_batch_and_arguments()
        manager = SimpleNamespace(
            rid_to_state={"second": SimpleNamespace(obj=obj[1])},
            abort_request=Mock(),
        )
        task = _create_abort_task(TokenizerManager.create_abort_task, manager, obj)
        with patch("asyncio.sleep", AsyncMock()):
            await task()
        manager.abort_request.assert_called_once_with("second")

    async def test_other_request_types_keep_native_abort_task(self):
        obj = object()
        manager = object()
        original = Mock(return_value=object())
        self.assertIs(_create_abort_task(original, manager, obj), original.return_value)
        original.assert_called_once_with(manager, obj)


if __name__ == "__main__":
    unittest.main()
