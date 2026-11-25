"""Check modality selection and native fallback without loading model weights."""

import asyncio
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock

import msgspec

source = Path(__file__).parents[2] / "adapters/vllm/VLLMDeferredRenderer.py"
spec = importlib.util.spec_from_file_location("deferred_media_under_test", source)
deferred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deferred)


@dataclass
class Params:
    mm_processor_kwargs: dict | None = None


def message(*parts):
    return [{"role": "user", "content": list(parts)}]


def media(modality, url="https://media.invalid/item"):
    return {"type": modality + "_url", modality + "_url": {"url": url}}


class ModalitySelectionTest(TestCase):
    def test_model_media_tokens_include_audio_video_and_nested_thinker(self):
        for config in (
            SimpleNamespace(image_token_id=1, video_token_id=2, audio_token_index=3),
            SimpleNamespace(
                thinker_config=SimpleNamespace(
                    image_token_id=1, video_token_id=2, audio_token_index=3
                )
            ),
        ):
            renderer = SimpleNamespace(
                mm_processor=SimpleNamespace(
                    info=SimpleNamespace(get_hf_config=lambda: config)
                )
            )
            self.assertEqual(
                {"image": 1, "video": 2, "audio": 3},
                deferred._media_token_ids(renderer),
            )

    def test_url_media_and_inline_audio_can_defer_but_native_objects_do_not(self):
        for modality in ("image", "video", "audio"):
            self.assertTrue(
                deferred._can_defer_conversation(
                    message(media(modality)), {"image", "video", "audio"}, {}
                )
            )
        self.assertTrue(
            deferred._can_defer_conversation(
                message(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "YWJj", "format": "wav"},
                    },
                ),
                {"audio"},
                {},
            )
        )
        for part in (
            {"type": "image_embeds", "image_embeds": "opaque"},
            {"type": "image_pil", "image_pil": object()},
            {"type": "future_media", "data": "opaque"},
            media("video", ""),
        ):
            self.assertFalse(
                deferred._can_defer_conversation(
                    message(media("image"), part), {"image", "video", "audio"}, {}
                )
            )
        self.assertFalse(
            deferred._can_defer_conversation(
                message(media("audio")), {"image", "video"}, {}
            )
        )
        self.assertFalse(
            deferred._can_defer_conversation(
                message({"type": "text", "text": "hello"}), {"image"}, {}
            )
        )

    def test_video_with_audio_requires_both_modalities(self):
        conversation = message(media("video"))
        self.assertTrue(deferred._can_defer_conversation(conversation, {"video"}, {}))
        self.assertFalse(
            deferred._can_defer_conversation(
                conversation, {"video"}, {"use_audio_in_video": True}
            )
        )
        self.assertTrue(
            deferred._can_defer_conversation(
                conversation, {"video", "audio"}, {"use_audio_in_video": True}
            )
        )


class MixedBatchFallbackTest(IsolatedAsyncioTestCase):
    def setUp(self):
        config = SimpleNamespace(
            image_token_id=1, video_token_id=2, audio_token_index=3
        )
        self.native = SimpleNamespace(
            mm_processor=SimpleNamespace(
                info=SimpleNamespace(get_hf_config=lambda: config)
            )
        )

        async def render(conversations, *args, **kwargs):
            return conversations, [
                {"native": conversation} for conversation in conversations
            ]

        async def predict(conversations, *args, **kwargs):
            return conversations, [
                {"deferred": conversation, "_navi_sched_tokenized_prompt": b"tokens"}
                for conversation in conversations
            ]

        self.native.render_chat_async = AsyncMock(side_effect=render)
        self.native.render_cmpl_async = AsyncMock(return_value=["native-completion"])
        self.renderer = deferred.VLLMDeferredRenderer(self.native)
        self.renderer._prediction.render_chat_async = AsyncMock(side_effect=predict)

    async def test_mixed_batch_keeps_order_and_only_attaches_supported_continuations(
        self,
    ):
        conversations = [
            message({"type": "text", "text": "text"}),
            message(media("image")),
            message(media("video")),
            message(media("audio")),
            message(media("image"), media("video"), media("audio")),
            message({"type": "image_embeds", "image_embeds": "opaque"}),
        ]
        extra = {"cache_salt": "tenant"}
        outputs, prompts = await self.renderer.render_chat_async(
            conversations, Params(), prompt_extras=extra
        )
        self.assertEqual(conversations, outputs)
        self.assertEqual(6, len(prompts))
        self.assertEqual(
            [False, True, True, True, True, False],
            ["navi_sched_render_call" in prompt for prompt in prompts],
        )
        for conversation, prompt in zip(conversations, prompts):
            if "navi_sched_render_call" in prompt:
                call = msgspec.msgpack.decode(prompt["navi_sched_render_call"])
                self.assertEqual([conversation], call["inputs"])
                self.assertEqual(extra, call["prompt_extras"])
                self.assertEqual(b"tokens", call["tokenized_prompt"])
            else:
                self.assertEqual(conversation, prompt["native"])
        self.assertEqual(2, self.native.render_chat_async.await_count)
        self.assertEqual(4, self.renderer._prediction.render_chat_async.await_count)

    async def test_fallback_does_not_hide_native_errors(self):
        error = ValueError("original model validation")
        self.native.render_chat_async.side_effect = error
        with self.assertRaises(ValueError) as caught:
            await self.renderer.render_chat_async(
                [message({"type": "image_embeds", "image_embeds": "opaque"})], Params()
            )
        self.assertIs(error, caught.exception)
        self.renderer._prediction.render_chat_async.assert_not_awaited()

    async def test_materialized_completion_inputs_keep_native_path(self):
        opaque_pixels = object()
        prompts = [{"prompt": "image", "multi_modal_data": {"image": opaque_pixels}}]
        self.assertEqual(
            ["native-completion"], await self.renderer.render_cmpl_async(prompts)
        )
        self.native.render_cmpl_async.assert_awaited_once_with(
            prompts, None, prompt_extras=None
        )


if __name__ == "__main__":
    main()
