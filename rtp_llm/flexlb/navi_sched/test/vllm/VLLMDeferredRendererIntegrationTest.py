"""CPU integration with a real vLLM renderer/tokenizer and HTTP image loader.

Run with the packaged vLLM Python environment and this repository on PYTHONPATH.
No GPU weights are loaded; the tiny local tokenizer/config is synthetic.
"""

import asyncio
import importlib
import importlib.util
import io
import os
import socket
import sys
import tempfile
import threading
import unittest
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

import msgspec
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, PreTrainedTokenizerFast

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.renderers import ChatParams, TokenizeParams
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs.deepseek_vl import DeepseekVLConfig
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

_module_path = Path(__file__).parents[2] / "navi_sched/adapters/vllm/VLLMDeferredRenderer.py"
_spec = importlib.util.spec_from_file_location("deferred_renderer", _module_path)
deferred = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deferred)
_prepare_package = types.ModuleType("_navi_prepare_test")
_prepare_package.__path__ = [str(_module_path.parent)]
sys.modules["_navi_prepare_test"] = _prepare_package
sys.modules["_navi_prepare_test.VLLMDeferredRenderer"] = deferred
deferred.__package__ = "_navi_prepare_test"
_prepare_spec = importlib.util.spec_from_file_location(
    "_navi_prepare_test.VLLMNativePreparation",
    _module_path.with_name("VLLMNativePreparation.py"),
)
preparation = importlib.util.module_from_spec(_prepare_spec)
_prepare_spec.loader.exec_module(preparation)


class PreparationProgressTest(unittest.IsolatedAsyncioTestCase):
    async def test_completed_requests_wait_for_slow_sibling_and_cancel_cleans_counts(self):
        client = preparation.NativePreparationClient(1)
        slow_started = asyncio.Event()
        release = asyncio.Event()

        async def prepare(request, *args):
            if request == "slow":
                slow_started.set()
                await release.wait()
            return request

        with mock.patch.object(client, "_prepare_one", side_effect=prepare):
            batch = asyncio.create_task(client.resume_async(["fast", "slow"]))
            await slow_started.wait()
            progress = client._progress
            self.assertEqual(1, progress.active_batches)
            self.assertEqual(1, progress.active_requests)
            self.assertEqual(1, progress.ready_requests)
            self.assertEqual(1, progress.completed_requests)
            batch.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await batch
            self.assertEqual(0, progress.active_batches)
            self.assertEqual(0, progress.active_requests)
            self.assertEqual(0, progress.ready_requests)
            self.assertEqual(1, progress.cancelled_requests)
            self.assertEqual(0, progress.failed_requests)

    async def test_overlapping_batches_and_failure_do_not_mix_ready_ownership(self):
        client = preparation.NativePreparationClient(1)
        slow_started = asyncio.Event()
        release = asyncio.Event()

        async def prepare(request, *args):
            if request == "slow":
                slow_started.set()
                await release.wait()
            if request == "bad":
                raise ValueError("media failure")
            return request

        with mock.patch.object(client, "_prepare_one", side_effect=prepare):
            first = asyncio.create_task(client.resume_async(["ready", "slow"]))
            await slow_started.wait()
            second = await client.resume_async(["other", "bad"], return_exceptions=True)
            self.assertEqual("other", second[0])
            self.assertIsInstance(second[1], ValueError)
            progress = client._progress
            self.assertEqual(1, progress.active_batches)
            self.assertEqual(1, progress.active_requests)
            self.assertEqual(1, progress.ready_requests)
            self.assertEqual(2, progress.completed_requests)
            self.assertEqual(1, progress.failed_requests)
            release.set()
            self.assertEqual(["ready", "slow"], await first)
            self.assertEqual(0, progress.active_batches)
            self.assertEqual(0, progress.active_requests)
            self.assertEqual(0, progress.ready_requests)
            self.assertEqual(3, progress.completed_requests)

    async def test_progress_logging_is_throttled_and_cannot_fail_preparation(self):
        progress = preparation._PreparationProgress(active_requests=17, ready_requests=3)
        with (
            mock.patch.object(preparation.time, "monotonic", side_effect=[5.0, 5.5, 6.0]),
            mock.patch.object(preparation.logger, "info") as log,
        ):
            progress.log("api", 1234)
            progress.log("api", 4321)
            progress.log("api", 5678)
            self.assertEqual(2, log.call_count)
            self.assertEqual(1234, log.call_args_list[0].args[-2])
            self.assertEqual(5678, log.call_args_list[1].args[-2])
        client = preparation.NativePreparationClient(1)
        with (
            mock.patch.object(client, "_prepare_one", new=mock.AsyncMock(return_value="ok")),
            mock.patch.object(preparation.logger, "info", side_effect=RuntimeError("logger failed")),
        ):
            self.assertEqual(["ok"], await client.resume_async(["request"]))
        self.assertEqual(0, client._progress.active_batches)
        self.assertEqual(0, client._progress.active_requests)
        self.assertEqual(0, client._progress.ready_requests)


class DeferredRendererTestBase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="navi-deferred-render-test-")
        cls.config = cls.create_model_config(cls.temp.name)
        cls.http_requests = []
        cls.image_color = "red"
        cls.download_gate = threading.Event()
        cls.downloads_started = threading.Event()
        cls.gated_download_count = 0
        cls.gated_download_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.http_requests.append(self.path)
                if self.path.startswith("/gated.png"):
                    with cls.gated_download_lock:
                        cls.gated_download_count += 1
                        if cls.gated_download_count == 17:
                            cls.downloads_started.set()
                    if not cls.download_gate.wait(10):
                        self.send_error(504)
                        return
                if self.path == "/missing.png":
                    self.send_response(404)
                    self.send_header("Error-Code", "NO_DATA")
                    self.send_header("Error-Message", "no data found from transcode cluster")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path == "/limited.png":
                    self.send_response(503)
                    self.send_header("Error-Code", "RATE_LIMIT_EXCEEDED")
                    self.send_header("Error-Message", "Rate limit exceeded")
                    self.send_header("X-Trace-Id", "image-limit-trace-123")
                    self.send_header("Retry-After", "2")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                buffer = io.BytesIO()
                Image.new("RGB", (137, 61), cls.image_color).save(buffer, format="PNG")
                data = buffer.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.socket.listen(128)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d/image.png" % cls.server.server_port

    @staticmethod
    def create_model_config(path):
        AutoConfig.register("deepseek_vl", DeepseekVLConfig, exist_ok=True)
        config = DeepseekVLConfig(
            image_token_index=1, num_image_tokens=576,
            architectures=["MultiModalityCausalLM"],
        )
        config.save_pretrained(path)
        inner = Tokenizer(models.WordLevel({
            "[UNK]": 0, "<image_placeholder>": 1, "hello": 2,
            "world": 3, "user": 4, "assistant": 5,
        }, unk_token="[UNK]"))
        inner.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image_placeholder>"]})
        tokenizer.chat_template = (
            "{% for message in messages %}{{message['role']}} "
            "{{message['content']}} {% endfor %}"
        )
        tokenizer.save_pretrained(path)
        return VllmConfig(
            model_config=ModelConfig(
                model=path, tokenizer=path, dtype="float32",
                max_model_len=2048, limit_mm_per_prompt={"image": 2},
            ),
            device_config=DeviceConfig(device="cpu"),
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def setUp(self):
        self.image_metrics = {}
        monitor = mock.Mock()
        self.image_monitor = monitor
        monitor.register_acc_metric.side_effect = lambda name, tags: self.image_metrics.setdefault(
            name, mock.Mock()
        )
        metrics_module = types.ModuleType("llm_plugin.metrics")
        metrics_module.__dict__.update(importlib.import_module("llm_plugin.metrics").__dict__)
        metrics_module._kmonitor = monitor
        previous_metrics_module = sys.modules.get("llm_plugin.metrics")
        sys.modules["llm_plugin.metrics"] = metrics_module
        def restore_metrics_module():
            if previous_metrics_module is None:
                sys.modules.pop("llm_plugin.metrics", None)
            else:
                sys.modules["llm_plugin.metrics"] = previous_metrics_module
        self.addCleanup(restore_metrics_module)
        deferred._image_preparation_metrics = None
        deferred._image_preparation_metrics_failed = False
        self.source_processor = InputProcessor(self.config)
        self.native = self.source_processor.renderer
        self.native.mm_processor.cache = MULTIMODAL_REGISTRY.processor_only_cache_from_config(
            self.config
        )
        self.source = deferred.VLLMDeferredRenderer(self.native)
        self.target = deferred.NativeRenderContinuation(self.config, ("generate",))
        self.chat_params = ChatParams(chat_template_content_format="string")
        self.tok_params = TokenizeParams(max_total_tokens=2048, add_special_tokens=False)
        self.params = SamplingParams(max_tokens=3, temperature=0, n=1)
        self.http_requests.clear()
        type(self).image_color = "red"
        self.download_gate.clear()
        self.downloads_started.clear()
        type(self).gated_download_count = 0

    def tearDown(self):
        self.source_processor.renderer.shutdown()
        self.target._processor.renderer.shutdown()
        self.target.close()

    async def asyncTearDown(self):
        from vllm.connections import global_http_connection

        client = global_http_connection._async_client
        if client is not None:
            await client.close()
            global_http_connection._async_client = None

    def messages(self, count):
        content = [{"type": "text", "text": "hello world"}]
        content.extend({"type": "image_url", "image_url": {"url": self.url}}
                       for _ in range(count))
        return [{"role": "user", "content": content}]

    def request(self, prompt):
        request = self.source_processor.process_inputs(
            "original-child-7", prompt, self.params, ("generate",), arrival_time=11.0
        )
        request.external_req_id = "parent-request"
        request.client_index = 71
        return request

    def wire_roundtrip(self, request):
        frames = [bytes(frame) for frame in MsgpackEncoder().encode(request)]
        return MsgpackDecoder(EngineCoreRequest).decode(msgspec.msgpack.decode(
            msgspec.msgpack.encode(frames)
        ))

    @asynccontextmanager
    async def local_preparation_server(self):
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        app.state.vllm_config = self.config
        app.state.engine_client = types.SimpleNamespace(
            get_supported_tasks=mock.AsyncMock(return_value=("generate",)),
        )
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        with mock.patch.dict(os.environ, {
            "VLLM_NAVI_SCHED_DEFER_RENDER": "1",
            "VLLM_NAVI_SCHED_SERVER_PORT": str(listener.getsockname()[1]),
        }):
            preparation.configure_native_preparation(app)
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        client = preparation.NativePreparationClient(listener.getsockname()[1])
        try:
            async def ready():
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("native preparation server exited before startup")
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(ready(), 10)
            yield client
        finally:
            await client.aclose()
            server.should_exit = True
            await asyncio.wait_for(task, 10)
            listener.close()
            self.assertEqual((), preparation._worker_sockets(client._port))


    async def assert_native_image_cache_mapping(self):
        from vllm.utils.hashing import sha256_cbor
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.v1.request import Request
        from _navi_prepare_test.VLLMImageCacheMap import VLLMImageCacheMap
        from _navi_prepare_test.VLLMBlockHash import calculate_vllm_block_hashes

        messages = self.messages(2)
        messages[0]["content"][0]["text"] = "hello world " * 16
        _, prompts = await self.source.render_chat_async(
            [messages], self.chat_params, self.tok_params,
        )
        prediction = self.wire_roundtrip(self.request(prompts[0]))
        actual = self.wire_roundtrip(prediction)
        await self.target.resume_async([actual])
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
            init_none_hash(sha256_cbor)
        native = Request.from_engine_core_request(actual, get_request_block_hasher(4, sha256_cbor))
        cache = types.SimpleNamespace(_full_ready=set(), _tail_ready=set())
        cache._flat_ready = cache._full_ready
        mapping = VLLMImageCacheMap(cache, False)
        self.assertTrue(mapping.register(71, prediction, native, 4).empty())
        self.assertEqual(2, sum(f.mm_position.length for f in prediction.mm_features))
        position = native.mm_features[0].mm_position
        missing = native.block_hashes[(position.offset + position.length - 1) // 4]
        cache._full_ready.update(native.block_hashes)
        cache._full_ready.remove(missing)
        self.assertTrue(mapping.update(native.block_hashes).empty())
        cache._full_ready.add(missing)
        delta = mapping.update([missing])
        actual_hashes = {int.from_bytes(value, "big") for value in delta.added}
        expected = calculate_vllm_block_hashes(prediction.prompt_token_ids, 4, request=prediction)
        first_image_block = prediction.mm_features[0].mm_position.offset // 4
        shift = len(native.prompt_token_ids) - len(prediction.prompt_token_ids)
        expected = {value for index, value in enumerate(expected)
                    if index >= first_image_block
                    and ((index + 1) * 4 + shift + 3) // 4 <= len(native.block_hashes)}
        self.assertEqual(expected, actual_hashes)
        self.assertGreater(len(actual_hashes), 2)
        mapping.finish(71)
        self.assertEqual({}, mapping._owners)
        cache._full_ready.remove(missing)
        self.assertEqual(set(delta.added), set(mapping.update([missing]).removed))
        self.assertEqual({}, mapping._records)


class DeferredRendererTest(DeferredRendererTestBase):
    async def test_native_cache_events_confirm_scheduling_hashes_only_after_entire_image(self):
        await self.assert_native_image_cache_mapping()

    async def test_source_does_not_download_before_transport(self):
        await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        self.assertEqual([], self.http_requests)

    async def test_preparation_rpc_burst_waits_without_losing_requests(self):
        class Continuation:
            def __init__(self, *args):
                pass

            async def resume_async(self, requests, **kwargs):
                await asyncio.sleep(0)
                for request in requests:
                    request.navi_sched_render_call = None
                    request.mm_features = []

            async def aclose(self):
                pass

        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        prototype = self.request(preview[0])
        requests = [self.wire_roundtrip(prototype) for _ in range(512)]
        for index, request in enumerate(requests):
            request.request_id = "rpc-burst-%d" % index
        with mock.patch.object(deferred, "NativeRenderContinuation", Continuation):
            async with self.local_preparation_server() as client:
                results = await asyncio.wait_for(
                    client.resume_async(requests, return_exceptions=True), 20
                )
        errors = [repr(result) for result in results if isinstance(result, BaseException)]
        self.assertEqual([], errors)
        self.assertEqual(["rpc-burst-%d" % index for index in range(512)],
                         [request.request_id for request in results])
        self.assertTrue(all(request.navi_sched_render_call is None for request in results))
        self.assertEqual([], self.http_requests)

    async def test_text_chat_uses_native_renderer_without_continuation(self):
        conversations = [self.messages(0), [{"role": "user", "content": "hello"}]]
        with mock.patch.object(
            self.source._prediction, "render_chat_async",
            side_effect=AssertionError("text must bypass prediction renderer"),
        ), mock.patch.object(
            self.native, "render_chat_async", wraps=self.native.render_chat_async,
        ) as native_entry:
            _, prompts = await self.source.render_chat_async(
                conversations, self.chat_params, self.tok_params,
                prompt_extras={"cache_salt": "text-tenant"},
            )
        self.assertEqual(1, native_entry.await_count)
        self.assertEqual(conversations, native_entry.call_args.args[0])
        self.assertEqual(2, len(prompts))
        for prompt in prompts:
            self.assertNotIn("navi_sched_render_call", prompt)
            request = self.wire_roundtrip(self.request(prompt))
            self.assertIsNone(request.navi_sched_render_call)
            self.assertEqual("text-tenant", request.cache_salt)
            self.assertIsNone(request.mm_features)
        self.assertEqual([], self.http_requests)

    async def test_text_completion_uses_native_renderer_without_continuation(self):
        inputs = [{"prompt": "hello world"}, {"prompt": "hello", "multi_modal_data": {}},
                  {"prompt_token_ids": [2, 3]}]
        with mock.patch.object(
            self.source._prediction, "render_cmpl_async",
            side_effect=AssertionError("text must bypass prediction renderer"),
        ), mock.patch.object(
            self.native, "render_cmpl_async", wraps=self.native.render_cmpl_async,
        ) as native_entry:
            prompts = await self.source.render_cmpl_async(inputs, self.tok_params)
        self.assertEqual(1, native_entry.await_count)
        self.assertEqual(inputs, native_entry.call_args.args[0])
        self.assertEqual(3, len(prompts))
        for prompt in prompts:
            self.assertNotIn("navi_sched_render_call", prompt)
            self.assertIsNone(self.wire_roundtrip(self.request(prompt)).navi_sched_render_call)
        self.assertEqual([], self.http_requests)

    async def test_prediction_payload_contains_no_pixels(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        predicted = self.request(preview[0])
        source_bytes = sum(len(frame) for frame in MsgpackEncoder().encode(predicted))
        _, normal = await self.native.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        real = self.request(normal[0])
        old_bytes = sum(len(frame) for frame in MsgpackEncoder().encode(real))
        self.assertLess(source_bytes, 8192)
        self.assertGreater(old_bytes, 1024 * 1024)
        print("PAYLOAD_BYTES prediction=%d native=%d" % (source_bytes, old_bytes))

    async def test_chat_text_and_images_resume_same_native_entry(self):
        for count in (0, 1, 2):
            with self.subTest(image_count=count):
                self.http_requests.clear()
                messages = self.messages(count)
                _, preview = await self.source.render_chat_async(
                    [messages], self.chat_params, self.tok_params,
                    prompt_extras={"cache_salt": "isolated-tenant"},
                )
                self.assertEqual([], self.http_requests)
                request = self.wire_roundtrip(self.request(preview[0]))
                if count:
                    call = msgspec.msgpack.decode(request.navi_sched_render_call)
                    self.assertEqual("render_chat_async", call["entry"])
                    self.assertEqual([self.messages(count)], call["inputs"])
                else:
                    self.assertIsNone(request.navi_sched_render_call)
                source_ids = [f.identifier for f in request.mm_features or []]
                self.assertTrue(all(f.data is None for f in request.mm_features or []))
                with mock.patch.object(
                    self.target._processor.renderer, "render_chat_async",
                    wraps=self.target._processor.renderer.render_chat_async,
                ) as native_entry:
                    await self.target.resume_async([request])
                    self.assertEqual(1 if count else 0, native_entry.await_count)
                self.assertEqual(count, len(self.http_requests))
                self.assertIsNone(request.navi_sched_render_call)
                self.assertEqual("original-child-7", request.request_id)
                self.assertEqual("parent-request", request.external_req_id)
                self.assertEqual(71, request.client_index)
                self.assertEqual(11.0, request.arrival_time)
                self.assertEqual("isolated-tenant", request.cache_salt)
                _, normal = await self.native.render_chat_async(
                    [self.messages(count)], self.chat_params, self.tok_params,
                    prompt_extras={"cache_salt": "isolated-tenant"},
                )
                baseline = self.request(normal[0])
                self.assertEqual(baseline.prompt_token_ids, request.prompt_token_ids)
                self.assertEqual(baseline.params, request.params)
                for actual, expected, prediction in zip(
                    request.mm_features or [], baseline.mm_features or [], source_ids,
                    strict=True,
                ):
                    self.assertEqual(expected.identifier, actual.identifier)
                    self.assertNotEqual(prediction, actual.identifier)
                    self.assertEqual(expected.mm_position, actual.mm_position)
                    self.assertTrue(torch.equal(
                        expected.data["pixel_values"].data,
                        actual.data["pixel_values"].data,
                    ))

    async def test_image_resume_reuses_tokens_without_second_tokenization(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(2)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        predicted_tokens = list(request.prompt_token_ids)
        with mock.patch.object(
            self.target._processor.renderer.get_tokenizer(), "apply_chat_template",
            side_effect=AssertionError("Agent must reuse Client tokenization"),
        ):
            await self.target.resume_async([request])
        self.assertEqual(len(predicted_tokens) + 2 * (576 - 1), len(request.prompt_token_ids))
        self.assertEqual(2, predicted_tokens.count(1))
        self.assertEqual(1152, request.prompt_token_ids.count(1))
        self.assertEqual(2, len(request.mm_features))
        self.assertTrue(all(feature.data is not None for feature in request.mm_features))
        self.assertIsNone(request.navi_sched_render_call)
        self.assertEqual(["/image.png", "/image.png"], self.http_requests)

    async def test_legacy_continuation_without_saved_tokens_still_uses_native_path(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        call = msgspec.msgpack.decode(request.navi_sched_render_call)
        del call["tokenized_prompt"]
        request.navi_sched_render_call = msgspec.msgpack.encode(call)
        await self.target.resume_async([request])
        _, normal = await self.native.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        expected = self.request(normal[0])
        self.assertEqual(expected.prompt_token_ids, request.prompt_token_ids)
        self.assertEqual(expected.mm_features[0].identifier, request.mm_features[0].identifier)
        self.assertTrue(torch.equal(
            expected.mm_features[0].data["pixel_values"].data,
            request.mm_features[0].data["pixel_values"].data,
        ))
        self.assertIsNone(request.navi_sched_render_call)

    async def test_native_rpc_failure_isolated_and_next_request_succeeds(self):
        from aiohttp import ClientResponseError

        messages = [self.messages(1), self.messages(1)]
        messages[0][0]["content"][1]["image_url"]["url"] = self.url.replace(
            "/image.png", "/missing.png"
        )
        _, preview = await self.source.render_chat_async(
            messages, self.chat_params, self.tok_params
        )
        failed, good = [self.wire_roundtrip(self.request(prompt)) for prompt in preview]
        failed.request_id = "missing-image"
        async with self.local_preparation_server() as client:
            results = await client.resume_async([failed, good], return_exceptions=True)
            self.assertIsInstance(results[0], ClientResponseError)
            self.assertEqual(422, results[0].status)
            self.assertIsNotNone(failed.navi_sched_render_call)
            self.assertIsNone(failed.mm_features[0].data)
            self.assertIs(good, results[1])
            self.assertIsNone(good.navi_sched_render_call)
            self.assertIsNotNone(good.mm_features[0].data)
            next_request = self.wire_roundtrip(self.request(preview[1]))
            self.assertEqual([next_request], await client.resume_async([next_request]))
            self.assertIsNone(next_request.navi_sched_render_call)
        self.assertIsNone(client._session)

    async def test_cancelled_native_rpc_cannot_mutate_agent_request_later(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        request.request_id = "cancelled-rpc"
        original_payload = request.navi_sched_render_call
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original_resume = deferred.NativeRenderContinuation._resume_one

        async def resume(continuation, request, *args):
            if request.request_id == "cancelled-rpc":
                started.set()
                await release.wait()
            try:
                return await original_resume(continuation, request, *args)
            finally:
                finished.set()

        with mock.patch.object(deferred.NativeRenderContinuation, "_resume_one", resume):
            async with self.local_preparation_server() as client:
                task = asyncio.create_task(client.resume_async([request]))
                try:
                    await asyncio.wait_for(started.wait(), 5)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                finally:
                    release.set()
                await asyncio.wait_for(finished.wait(), 5)
                self.assertEqual(original_payload, request.navi_sched_render_call)
                self.assertIsNone(request.mm_features[0].data)
                next_request = self.wire_roundtrip(self.request(preview[0]))
                self.assertEqual([next_request], await client.resume_async([next_request]))
                self.assertIsNone(next_request.navi_sched_render_call)
        self.assertIsNone(client._session)

    async def test_native_failure_logs_correlate_original_error_across_rpc(self):
        from aiohttp import ClientResponseError

        messages = self.messages(1)
        missing_url = self.url.replace("/image.png", "/missing.png")
        messages[0]["content"][1]["image_url"]["url"] = missing_url
        _, preview = await self.source.render_chat_async(
            [messages], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        request.request_id = "failed-media-27"
        original_payload = request.navi_sched_render_call
        with self.assertLogs("vllm.navi_sched", level="INFO") as logs:
            async with self.local_preparation_server() as client:
                results = await client.resume_async(
                    [request], return_exceptions=True,
                    schedule_batch_id=87654321, local_batch_id=123,
                )
        self.assertIsInstance(results[0], ClientResponseError)
        self.assertEqual(422, results[0].status)
        self.assertEqual(original_payload, request.navi_sched_render_call)
        self.assertIsNone(request.mm_features[0].data)
        for marker in ("NATIVE_RENDER_REQUEST", "NATIVE_PREPARE_SERVER", "NATIVE_PREPARE_RPC"):
            records = [line for line in logs.output if marker in line]
            self.assertEqual(1, len(records), marker)
            self.assertIn("request_id=failed-media-27", records[0])
            self.assertIn("schedule_batch_id=87654321 local_batch_id=123", records[0])
        native = next(line for line in logs.output if "NATIVE_RENDER_REQUEST" in line)
        self.assertIn("phase=native_render", native)
        self.assertIn("error_type=ClientResponseError http_status=404", native)
        self.assertIn(missing_url, native)
        self.assertIn("upstream_code='NO_DATA'", native)
        self.assertIn("upstream_message='no data found from transcode cluster'", native)
        rpc = next(line for line in logs.output if "NATIVE_PREPARE_RPC" in line)
        self.assertIn("http_status=422", rpc)
        self.assertIn("prepare_pid=" + str(os.getpid()), rpc)
        self.assertEqual(["/missing.png"], self.http_requests)

    async def test_rate_limit_trace_survives_native_download_failure(self):
        from aiohttp import ClientResponseError

        messages = self.messages(1)
        messages[0]["content"][1]["image_url"]["url"] = self.url.replace(
            "/image.png", "/limited.png"
        )
        _, preview = await self.source.render_chat_async(
            [messages], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        with self.assertLogs("vllm.navi_sched", level="INFO") as logs:
            async with self.local_preparation_server() as client:
                results = await client.resume_async([request], return_exceptions=True)
        self.assertIsInstance(results[0], ClientResponseError)
        self.assertEqual(422, results[0].status)
        native = next(line for line in logs.output if "NATIVE_RENDER_REQUEST" in line)
        self.assertIn("http_status=503", native)
        self.assertIn("upstream_code='RATE_LIMIT_EXCEEDED'", native)
        self.assertIn("upstream_trace_id='image-limit-trace-123'", native)
        self.assertIn("upstream_retry_after='2'", native)
        self.assertIn("upstream_server='BaseHTTP/", native)
        self.assertEqual(["/limited.png"], self.http_requests)

    async def test_image_preparation_metrics_count_all_outcomes_without_sampling(self):
        from aiohttp import ClientResponseError

        for path, expected in (("/image.png", (1, 1, 0, 0, 0)),
                               ("/limited.png", (1, 0, 1, 0, 0)),
                               ("/missing.png", (1, 0, 0, 1, 0))):
            messages = self.messages(1)
            messages[0]["content"][1]["image_url"]["url"] = self.url.replace("/image.png", path)
            _, preview = await self.source.render_chat_async(
                [messages], self.chat_params, self.tok_params
            )
            request = self.wire_roundtrip(self.request(preview[0]))
            results = await self.target.resume_async([request], return_exceptions=True)
            if path == "/image.png":
                self.assertIs(request, results[0])
            else:
                self.assertIsInstance(results[0], ClientResponseError)
            for name, value in zip(("completed", "success", "rate_limited", "other_error", "cancelled"), expected):
                metric = self.image_metrics["navi_sched.agent.image_prepare_" + name + "_qps"]
                self.assertEqual(mock.call(value), metric.report.call_args)
            self.assertEqual(5, len(self.image_metrics))
            self.assertEqual(5, self.image_monitor.register_acc_metric.call_count)
            for registration in self.image_monitor.register_acc_metric.call_args_list:
                self.assertEqual(str(os.getpid()), registration.args[1]["api_process_id"])

        before = {name: metric.report.call_count for name, metric in self.image_metrics.items()}
        text = types.SimpleNamespace(navi_sched_render_call=None)
        self.assertIs(text, await self.target._resume_one(text))
        self.assertEqual(before, {name: metric.report.call_count for name, metric in self.image_metrics.items()})
        deferred._report_image_preparation("cancelled", 0, "")
        for name, value in (("completed", 0), ("success", 0), ("rate_limited", 0),
                            ("other_error", 0), ("cancelled", 1)):
            self.assertEqual(mock.call(value), self.image_metrics[
                "navi_sched.agent.image_prepare_" + name + "_qps"
            ].report.call_args)

    async def test_image_preparation_metric_failure_does_not_fail_rendering(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        sys.modules["llm_plugin.metrics"]._kmonitor.register_acc_metric.side_effect = RuntimeError("metrics down")
        with self.assertLogs("vllm.navi_sched.render", level="WARNING"):
            self.assertEqual([request], await self.target.resume_async([request]))
        self.assertIsNotNone(request.mm_features[0].data)
        self.assertTrue(deferred._image_preparation_metrics_failed)

    async def test_native_rpc_uses_api_process_socket_without_tcp_or_files(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        async with self.local_preparation_server() as client:
            addresses = preparation._worker_sockets(client._port)
            self.assertEqual(("\0" + preparation._socket_prefix(client._port) + str(os.getpid()),), addresses)
            self.assertEqual([request], await client.resume_async([request]))
            self.assertIsNone(client._session)
            self.assertEqual(addresses, tuple(client._peers))
            self.assertEqual(addresses, preparation._worker_sockets(client._port))
            self.assertEqual(0, client._peers[addresses[0]].in_flight)
            self.assertIsNone(request.navi_sched_render_call)
            self.assertIsNotNone(request.mm_features[0].data)
        self.assertEqual({}, client._peers)

    async def test_native_rpc_keeps_loopback_compatibility_without_worker_sockets(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        async with self.local_preparation_server() as client:
            with mock.patch.object(preparation, "_worker_sockets", return_value=()):
                self.assertEqual([request], await client.resume_async([request]))
                self.assertIsNotNone(client._session)
                self.assertEqual({}, client._peers)
                self.assertIsNone(request.navi_sched_render_call)
        self.assertIsNone(client._session)

    async def test_crashed_api_process_leaves_no_worker_socket(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        code = (
            "import os,socket,signal; "
            "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); "
            "s.bind('\\0'+%r+str(os.getpid())); s.listen(1); "
            "print(os.getpid(),flush=True); signal.pause()"
        ) % preparation._socket_prefix(port)
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE,
        )
        try:
            pid = int(await asyncio.wait_for(process.stdout.readline(), 5))
            address = "\0" + preparation._socket_prefix(port) + str(pid)
            self.assertEqual((address,), preparation._worker_sockets(port))
            process.kill()
            await asyncio.wait_for(process.wait(), 5)
            self.assertEqual((), preparation._worker_sockets(port))
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            listener.close()

    async def test_worker_selection_balances_outstanding_preparations(self):
        addresses = tuple("\0navi_sched.native_prepare.1.%d" % pid for pid in range(8))
        client = preparation.NativePreparationClient(1)
        sessions = [types.SimpleNamespace(close=mock.AsyncMock()) for _ in addresses]
        with (
            mock.patch.object(preparation, "_worker_sockets", return_value=addresses),
            mock.patch("aiohttp.ClientSession", side_effect=sessions),
        ):
            peers = [await client._select_peer() for _ in range(64)]
            self.assertEqual(list(addresses) * 8, [address for address, _ in peers])
            self.assertEqual([8] * 8, [peer.in_flight for peer in client._peers.values()])
            client._peers[addresses[5]].in_flight = 3
            address, peer = await client._select_peer()
            self.assertEqual(addresses[5], address)
            self.assertEqual(4, peer.in_flight)
        await client.aclose()
        for session in sessions:
            session.close.assert_awaited_once()

    async def test_worker_retirement_waits_for_outstanding_preparations(self):
        addresses = ("\0navi_sched.native_prepare.1.101", "\0navi_sched.native_prepare.1.102")
        client = preparation.NativePreparationClient(1)
        sessions = [types.SimpleNamespace(close=mock.AsyncMock()) for _ in addresses]
        with (
            mock.patch.object(preparation, "_worker_sockets", return_value=addresses) as discover,
            mock.patch("aiohttp.ClientSession", side_effect=sessions),
        ):
            address, peer = await client._select_peer()
            self.assertEqual(addresses[0], address)
            client._discovered_at = 0
            discover.return_value = addresses[1:]
            self.assertEqual(addresses[1], (await client._select_peer())[0])
            self.assertTrue(peer.retired)
            sessions[0].close.assert_not_awaited()
            peer.in_flight = 0
            await client._select_peer()
            self.assertNotIn(addresses[0], client._peers)
            sessions[0].close.assert_awaited_once()
        await client.aclose()
        sessions[1].close.assert_awaited_once()

    async def test_local_api_prepares_native_request_without_tokenizing_or_executing(self):
        import httpx
        from fastapi import FastAPI

        app = FastAPI()
        app.state.vllm_config = self.config
        app.state.engine_client = types.SimpleNamespace(
            get_supported_tasks=mock.AsyncMock(return_value=("generate",)),
            generate=mock.Mock(side_effect=AssertionError("preparation must not execute")),
        )
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}):
            preparation.configure_native_preparation(app)
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        payload = msgspec.msgpack.encode({
            "frames": [bytes(frame) for frame in MsgpackEncoder().encode(request)],
            "schedule_batch_id": 111222333, "local_batch_id": 45,
        })
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=transport, base_url="http://localhost"
        ) as client:
            with mock.patch.object(
                self.native.get_tokenizer(), "apply_chat_template",
                side_effect=AssertionError("preparation must reuse source tokens"),
            ):
                response = await client.post(preparation.PREPARE_PATH, content=payload)
            self.assertEqual(200, response.status_code, response.text)
            ready = MsgpackDecoder(EngineCoreRequest).decode(msgspec.msgpack.decode(response.content))
            self.assertEqual(1, request.prompt_token_ids.count(1))
            self.assertEqual(576, ready.prompt_token_ids.count(1))
            self.assertEqual(request.request_id, ready.request_id)
            self.assertEqual(request.client_index, ready.client_index)
            self.assertEqual(request.params, ready.params)
            self.assertIsNone(ready.navi_sched_render_call)
            self.assertEqual(1, len(ready.mm_features))
            self.assertIsNotNone(ready.mm_features[0].data)
            self.assertEqual(str(os.getpid()), response.headers["X-Navi-Prepare-Pid"])
            self.assertEqual(["/image.png"], self.http_requests)
            app.state.engine_client.generate.assert_not_called()
        from vllm.connections import global_http_connection

        self.assertIsNone(global_http_connection._async_client)

    async def test_local_api_reuses_transport_native_processor_cache(self):
        import httpx
        from fastapi import FastAPI

        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        fallback_request = self.wire_roundtrip(request)
        await self.target.resume_async([fallback_request])
        cache = self.target._processor.renderer.mm_processor.cache._cache
        self.assertGreater(cache.currsize, 0)
        expected_cache_bytes = cache.currsize
        provider = mock.AsyncMock(return_value=self.target)
        app = FastAPI()
        app.state.vllm_config = self.config
        app.state.engine_client = types.SimpleNamespace(
            engine_core=types.SimpleNamespace(get_native_render_continuation=provider),
            get_supported_tasks=mock.AsyncMock(return_value=("generate",)),
        )
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}):
            preparation.configure_native_preparation(app)
        payload = msgspec.msgpack.encode({
            "frames": [bytes(frame) for frame in MsgpackEncoder().encode(request)],
            "schedule_batch_id": 111222333, "local_batch_id": 45,
        })
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        with mock.patch.object(
            deferred, "NativeRenderContinuation",
            side_effect=AssertionError("API must reuse the transport processor"),
        ):
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=transport, base_url="http://localhost"
            ) as client:
                response = await client.post(preparation.PREPARE_PATH, content=payload)
                self.assertEqual(200, response.status_code, response.text)
                ready = MsgpackDecoder(EngineCoreRequest).decode(
                    msgspec.msgpack.decode(response.content)
                )
                self.assertEqual(fallback_request.prompt_token_ids, ready.prompt_token_ids)
                self.assertEqual(fallback_request.mm_features[0].identifier,
                                 ready.mm_features[0].identifier)
                self.assertEqual(expected_cache_bytes, cache.currsize)
                provider.assert_awaited_once_with()
                app.state.engine_client.get_supported_tasks.assert_not_awaited()

    async def test_native_continuation_close_is_idempotent_in_both_orders(self):
        for async_first in (False, True):
            continuation = deferred.NativeRenderContinuation(self.config, ("generate",))
            renderer = continuation._processor.renderer
            with mock.patch.object(renderer, "shutdown", wraps=renderer.shutdown) as shutdown:
                if async_first:
                    await continuation.aclose()
                    continuation.close()
                else:
                    continuation.close()
                    await continuation.aclose()
                continuation.close()
                await continuation.aclose()
                shutdown.assert_called_once_with()

    async def test_native_api_builder_registers_preparation_only_when_enabled(self):
        import vllm.plugins as plugins
        from vllm.entrypoints.openai.api_server import build_app
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        with mock.patch.dict(os.environ, {"VLLM_PLUGINS": ""}):
            args = make_arg_parser(FlexibleArgumentParser()).parse_args(["--model", self.temp.name])
        for enabled in ("0", "1"):
            with (
                self.subTest(enabled=enabled),
                mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": enabled}),
                mock.patch.object(plugins, "_frontend_app_configurator", preparation.configure_native_preparation),
            ):
                app = build_app(args, supported_tasks=("generate",))
                paths = [getattr(route, "path", None) for route in app.routes]
                self.assertEqual(int(enabled), paths.count(preparation.PREPARE_PATH))

    async def test_prepare_endpoint_rejects_nonlocal_requests_before_decode(self):
        import httpx
        from fastapi import FastAPI

        app = FastAPI()
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}):
            preparation.configure_native_preparation(app)
        transport = httpx.ASGITransport(app=app, client=("192.0.2.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            response = await client.post(preparation.PREPARE_PATH, content=b"invalid-msgpack")
        self.assertEqual(403, response.status_code)
        self.assertEqual([], self.http_requests)

    async def test_prepare_client_keeps_text_local_without_http_session(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(0)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        client = preparation.NativePreparationClient(1)
        results = await client.resume_async([request])
        self.assertEqual([request], results)
        self.assertIsNone(client._session)
        await client.aclose()

    def test_native_serving_distinguishes_media_error_from_rate_limit(self):
        from vllm.entrypoints.openai.engine.protocol import GenerationError
        from vllm.entrypoints.openai.engine.serving import OpenAIServing

        serving = OpenAIServing.__new__(OpenAIServing)
        for reason, status in (
            ("vLLM media preprocessing failed", 500),
            ("navi_sched_retryable:agent response pipe reached eof", 429),
        ):
            with self.subTest(reason=reason), self.assertRaises(GenerationError) as raised:
                serving._raise_if_error("error", "failed-image", reason)
            self.assertEqual(status, raised.exception.status_code)

    async def test_native_render_timing_separates_render_from_process_inputs(self):
        _, preview = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.wire_roundtrip(self.request(preview[0]))
        clock_ns = [100_000_000]
        native_render = self.target._processor.renderer.render_chat_async
        native_process = self.target._processor.process_inputs

        async def render(*args, **kwargs):
            result = await native_render(*args, **kwargs)
            clock_ns[0] += 150_000_000
            return result

        def process(*args, **kwargs):
            result = native_process(*args, **kwargs)
            clock_ns[0] += 25_000_000
            return result

        with (
            mock.patch.object(self.target._processor.renderer, "render_chat_async", side_effect=render),
            mock.patch.object(self.target._processor, "process_inputs", side_effect=process),
            mock.patch("time.perf_counter_ns", side_effect=lambda: clock_ns[0]),
            self.assertLogs("vllm.navi_sched.render", level="INFO") as logs,
        ):
            results = await self.target.resume_async(
                [request], schedule_batch_id=87654321, local_batch_id=123
            )
        self.assertIs(request, results[0])
        timing = [line for line in logs.output if "NATIVE_RENDER_REQUEST" in line]
        self.assertEqual(1, len(timing))
        self.assertIn("request_id=original-child-7 status=ok", timing[0])
        self.assertIn("schedule_batch_id=87654321 local_batch_id=123", timing[0])
        self.assertIn("native_render_ms=150.000 process_inputs_ms=25.000 total_ms=175.000", timing[0])
        self.assertEqual(["/image.png"], self.http_requests)

    async def test_native_render_timing_records_failure_and_cancellation(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                _, preview = await self.source.render_chat_async(
                    [self.messages(1)], self.chat_params, self.tok_params
                )
                request = self.wire_roundtrip(self.request(preview[0]))
                clock_ns = [100_000_000]

                async def render(*args, **kwargs):
                    clock_ns[0] += 250_000_000
                    raise asyncio.CancelledError() if cancelled else ValueError("failed")

                with (
                    mock.patch.object(self.target._processor.renderer, "render_chat_async", side_effect=render),
                    mock.patch("time.perf_counter_ns", side_effect=lambda: clock_ns[0]),
                    self.assertLogs("vllm.navi_sched.render", level="INFO") as logs,
                ):
                    results = await self.target.resume_async([request], return_exceptions=True)
                self.assertIsInstance(results[0], asyncio.CancelledError if cancelled else ValueError)
                timing = [line for line in logs.output if "NATIVE_RENDER_REQUEST" in line]
                self.assertEqual(1, len(timing))
                self.assertIn("status=" + ("cancelled" if cancelled else "failed"), timing[0])
                self.assertIn("phase=native_render phase_ms=250.000", timing[0])
                self.assertIn("native_render_ms=-1.000 process_inputs_ms=-1.000 total_ms=250.000", timing[0])
                self.assertIsNotNone(request.navi_sched_render_call)
                self.assertEqual([], self.http_requests)

    async def test_text_completion_same_continuation_format(self):
        output = await self.source.render_cmpl_async(
            [{"prompt": "hello world"}], self.tok_params
        )
        request = self.wire_roundtrip(self.request(output[0]))
        self.assertIsNone(request.navi_sched_render_call)
        tokens = list(request.prompt_token_ids)
        with mock.patch.object(
            self.target._processor.renderer, "render_cmpl_async",
            side_effect=AssertionError("text must not render again on Agent"),
        ):
            await self.target.resume_async([request])
        self.assertEqual(tokens, request.prompt_token_ids)
        self.assertIsNone(request.mm_features)
        self.assertEqual([], self.http_requests)

    async def test_missing_image_does_not_fail_other_requests(self):
        conversations = [self.messages(1), self.messages(1), self.messages(0)]
        conversations[1][0]["content"][1]["image_url"]["url"] = (
            self.url.replace("/image.png", "/missing.png")
        )
        _, prompts = await self.source.render_chat_async(
            conversations, self.chat_params, self.tok_params
        )
        self.assertEqual([], self.http_requests)
        requests = [self.wire_roundtrip(self.request(prompt)) for prompt in prompts]
        results = await self.target.resume_async(requests, return_exceptions=True)
        from aiohttp import ClientResponseError

        self.assertIs(requests[0], results[0])
        self.assertIsInstance(results[1], ClientResponseError)
        self.assertEqual(404, results[1].status)
        self.assertIs(requests[2], results[2])
        self.assertIsNone(requests[0].navi_sched_render_call)
        self.assertIsNotNone(requests[0].mm_features[0].data)
        self.assertIsNotNone(requests[1].navi_sched_render_call)
        self.assertIsNone(requests[2].navi_sched_render_call)
        self.assertIsNone(requests[2].mm_features)
        self.assertCountEqual(["/image.png", "/missing.png"], self.http_requests)

    async def test_native_downloads_are_not_limited_to_sixteen_requests(self):
        conversations = [self.messages(1) for _ in range(17)]
        for index, messages in enumerate(conversations):
            messages[0]["content"][1]["image_url"]["url"] = (
                self.url.replace("/image.png", "/gated.png?request=%d" % index)
            )
        _, prompts = await self.source.render_chat_async(
            conversations, self.chat_params, self.tok_params
        )
        self.assertEqual([], self.http_requests)
        requests = [self.wire_roundtrip(self.request(prompt)) for prompt in prompts]
        task = asyncio.create_task(self.target.resume_async(requests))
        try:
            all_started = await asyncio.to_thread(self.downloads_started.wait, 2)
        finally:
            self.download_gate.set()
            results = await task
        self.assertTrue(all_started, "independent native downloads were serialized")
        self.assertEqual(17, self.gated_download_count)
        self.assertEqual(17, len(self.http_requests))
        self.assertEqual(17, len(results))
        for request, result in zip(requests, results, strict=True):
            self.assertIs(request, result)
            self.assertIsNone(request.navi_sched_render_call)
            self.assertEqual(1, len(request.mm_features))
            self.assertIsNotNone(request.mm_features[0].data)

    async def test_overwritten_url_changes_execution_hash(self):
        async def route():
            _, prompts = await self.source.render_chat_async(
                [self.messages(1)], self.chat_params, self.tok_params
            )
            request = self.wire_roundtrip(self.request(prompts[0]))
            prediction = request.mm_features[0].identifier
            await self.target.resume_async([request])
            return prediction, request.mm_features[0].identifier

        before = await route()
        type(self).image_color = "blue"
        after = await route()
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])

    async def test_prediction_tokens_are_replaced_with_native_execution_tokens(self):
        _, prompts = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.request(prompts[0])
        self.assertEqual(1, request.prompt_token_ids.count(1))
        prediction_identity = request.mm_features[0].identifier
        await self.target.resume_async([request])
        _, native = await self.native.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        expected = self.request(native[0])
        self.assertEqual(expected.prompt_token_ids, request.prompt_token_ids)
        self.assertEqual(576, request.prompt_token_ids.count(1))
        self.assertEqual(expected.mm_features[0].identifier, request.mm_features[0].identifier)
        self.assertNotEqual(prediction_identity, request.mm_features[0].identifier)
        self.assertIsNone(request.navi_sched_render_call)
        self.assertIsNotNone(request.mm_features[0].data)

    async def test_unknown_entry_is_rejected_without_loading_media(self):
        _, prompts = await self.source.render_chat_async(
            [self.messages(1)], self.chat_params, self.tok_params
        )
        request = self.request(prompts[0])
        call = msgspec.msgpack.decode(request.navi_sched_render_call)
        call["entry"] = "arbitrary_function"
        request.navi_sched_render_call = msgspec.msgpack.encode(call)
        with self.assertRaisesRegex(ValueError, "unsupported native renderer continuation entry"):
            await self.target.resume_async([request])
        self.assertEqual([], self.http_requests)

    def test_factory_requires_explicit_enable(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(self.native, deferred.create_deferred_renderer(self.native))
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}):
            self.assertIsInstance(deferred.create_deferred_renderer(self.native),
                                  deferred.VLLMDeferredRenderer)


class QwenDeferredRendererTest(DeferredRendererTestBase):
    async def test_native_cache_events_confirm_scheduling_hashes_only_after_entire_image(self):
        await self.assert_native_image_cache_mapping()

    @staticmethod
    def create_model_config(path):
        from transformers import (
            Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLConfig,
            Qwen3VLProcessor, Qwen3VLVideoProcessor,
        )

        config = Qwen3VLConfig(
            image_token_id=1, video_token_id=6,
            vision_start_token_id=7, vision_end_token_id=8,
            architectures=["Qwen3VLForConditionalGeneration"],
            text_config={
                "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 4,
            },
            vision_config={"patch_size": 16, "spatial_merge_size": 2},
        )
        config.save_pretrained(path)
        inner = Tokenizer(models.WordLevel({
            "[UNK]": 0, "<|image_pad|>": 1, "hello": 2, "world": 3,
            "user": 4, "assistant": 5, "<|video_pad|>": 6,
            "<|vision_start|>": 7, "<|vision_end|>": 8,
        }, unk_token="[UNK]"))
        inner.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = Qwen2TokenizerFast(
            tokenizer_object=inner, unk_token="[UNK]",
            eos_token="[UNK]", pad_token="[UNK]",
        )
        tokenizer.add_special_tokens({"additional_special_tokens": [
            "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>",
        ]})
        tokenizer.chat_template = (
            "{% for message in messages %}{{message['role']}} "
            "{{message['content']}} {% endfor %}"
        )
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessorFast(
                min_pixels=4096, max_pixels=65536, patch_size=16,
            ), tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(),
        )
        processor.save_pretrained(path)
        return VllmConfig(
            model_config=ModelConfig(
                model=path, tokenizer=path, dtype="float32",
                max_model_len=2048, limit_mm_per_prompt={"image": 2},
            ), device_config=DeviceConfig(device="cpu"),
        )

    async def test_fixed_prediction_restores_native_qwen_tokens_and_content_hash(self):
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

        self.assertIsInstance(self.native.mm_processor, Qwen3VLMultiModalProcessor)
        with mock.patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}):
            self.assertIsInstance(
                deferred.create_deferred_renderer(self.native), deferred.VLLMDeferredRenderer,
            )
        for count in (0, 1, 2):
            with self.subTest(image_count=count):
                self.http_requests.clear()
                _, prompts = await self.source.render_chat_async(
                    [self.messages(count)], self.chat_params, self.tok_params,
                )
                self.assertEqual([], self.http_requests)
                request = self.wire_roundtrip(self.request(prompts[0]))
                self.assertEqual(count, request.prompt_token_ids.count(1))
                with mock.patch.object(
                    self.native.get_tokenizer(), "apply_chat_template",
                    side_effect=AssertionError("Agent must reuse source tokenization"),
                ):
                    await self.target.resume_async([request])
                self.assertEqual(count, len(self.http_requests))
                _, normal = await self.native.render_chat_async(
                    [self.messages(count)], self.chat_params, self.tok_params,
                )
                expected = self.request(normal[0])
                self.assertEqual(count * 8, request.prompt_token_ids.count(1))
                self.assertEqual(expected.prompt_token_ids, request.prompt_token_ids)
                self.assertEqual(
                    [f.identifier for f in expected.mm_features or []],
                    [f.identifier for f in request.mm_features or []],
                )
                for actual, native in zip(
                    request.mm_features or [], expected.mm_features or [], strict=True,
                ):
                    self.assertEqual(native.mm_position, actual.mm_position)
                    self.assertTrue(torch.equal(
                        native.data["pixel_values"].data, actual.data["pixel_values"].data,
                    ))


if __name__ == "__main__":
    unittest.main()
