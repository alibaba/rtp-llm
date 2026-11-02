from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import asdict
from typing import Any

import msgspec

logger = logging.getLogger("vllm.navi_sched.render")
SCHEDULING_IMAGE_TOKEN_COUNT = 1
_image_preparation_metrics = None
_image_preparation_metrics_failed = False


def _report_image_preparation(status, http_status, upstream_code):
    global _image_preparation_metrics, _image_preparation_metrics_failed
    if _image_preparation_metrics_failed:
        return
    try:
        if _image_preparation_metrics is None:
            from llm_plugin.metrics import _kmonitor
            from .VLLMAdapterUtils import make_kmon_tags

            tags = make_kmon_tags("agent")
            tags["api_process_id"] = str(os.getpid())
            _image_preparation_metrics = {
                name: _kmonitor.register_acc_metric(
                    "navi_sched.agent.image_prepare_" + name + "_qps", tags
                )
                for name in ("completed", "success", "rate_limited", "other_error", "cancelled")
            }
        cancelled = status == "cancelled"
        limited = status == "failed" and http_status == 503 and upstream_code == "RATE_LIMIT_EXCEEDED"
        values = {
            "completed": not cancelled,
            "success": status == "ok",
            "rate_limited": limited,
            "other_error": status == "failed" and not limited,
            "cancelled": cancelled,
        }
        for name, value in values.items():
            _image_preparation_metrics[name].report(int(value))
    except Exception:
        _image_preparation_metrics_failed = True
        logger.warning("NaviSched image preparation metrics unavailable", exc_info=True)


class _PredictionRenderer:
    """Use native chat/token processing, replacing only media preparation."""

    def __init__(self, renderer):
        self._renderer = renderer

    def __getattr__(self, name):
        value = getattr(self._renderer, name)
        if getattr(value, "__self__", None) is self._renderer:
            return value.__func__.__get__(self, type(self))
        return value

    def process_for_engine(self, prompt, arrival_time):
        tokenized = msgspec.msgpack.encode(prompt)
        output = self._renderer.process_for_engine.__func__(self, prompt, arrival_time)
        output["_navi_sched_tokenized_prompt"] = tokenized
        return output

    @staticmethod
    def _chat_mm_tracker_factory(model_config, **kwargs):
        from vllm.entrypoints.chat_utils import (
            AsyncMultiModalContentParser,
            AsyncMultiModalItemTracker,
        )

        class ReferenceParser(AsyncMultiModalContentParser):
            async def _image_with_uuid_async(self, image_url, uuid, preprocess_config=None):
                if not image_url:
                    raise ValueError("deferred rendering requires an image URL")
                identity = hashlib.sha256(msgspec.json.encode(
                    ["navi-scheduling-image-url-v1", image_url, preprocess_config]
                )).hexdigest()
                return None, identity

            def _unsupported(self, *args, **kwargs):
                raise ValueError("deferred rendering requires URL images")

            parse_image_embeds = _unsupported
            parse_audio = _unsupported
            parse_input_audio = _unsupported
            parse_video = _unsupported

        class ReferenceTracker(AsyncMultiModalItemTracker):
            def create_parser(self, mm_processor_kwargs=None):
                return ReferenceParser(self, mm_processor_kwargs=mm_processor_kwargs)

        return ReferenceTracker(model_config, **kwargs)

    def _process_multimodal(
        self, prompt, mm_data, mm_uuids, mm_processor_kwargs, tokenization_kwargs
    ):
        from vllm.multimodal.inputs import mm_inputs
        from vllm.multimodal.processing import PromptReplacement

        if set(mm_data) != {"image"} or not mm_uuids or "image" not in mm_uuids:
            raise ValueError("deferred prediction supports URL images only")
        processor = self._renderer.get_mm_processor()
        count = len(mm_uuids["image"])
        config = processor.info.get_hf_config()
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = config.image_token_index
        updates = [PromptReplacement(
            modality="image", target=[image_token_id],
            replacement=[image_token_id] * SCHEDULING_IMAGE_TOKEN_COUNT,
        )]
        grouped = processor._bind_and_group_updates(updates, {"image": count})
        tokens = processor._apply_hf_processor_tokens_only(prompt)
        tokens, positions = processor._apply_prompt_updates(tokens, grouped)
        processor._validate_mm_placeholders(positions, {"image": count})
        return mm_inputs(
            prompt_token_ids=tokens,
            mm_kwargs={"image": [None] * count},
            mm_hashes={"image": mm_uuids["image"]},
            mm_placeholders={"image": [item.to_range() for item in positions["image"]]},
        )


class _TokenizedRenderer:
    """Resume native media preparation using the source's tokenized prompt."""

    def __init__(self, renderer, tokenized_prompt):
        self._renderer = renderer
        self._tokenized_prompt = tokenized_prompt

    def __getattr__(self, name):
        value = getattr(self._renderer, name)
        if getattr(value, "__self__", None) is self._renderer:
            return value.__func__.__get__(self, type(self))
        return value

    async def render_messages_async(self, messages, params):
        from vllm.entrypoints.chat_utils import parse_chat_messages_async
        from vllm.renderers.hf import resolve_chat_template_content_format

        conversation, media, uuids = await parse_chat_messages_async(
            messages, self.model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=self.get_tokenizer(), model_config=self.model_config,
            ),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        prompt = {}
        if media is not None:
            prompt["multi_modal_data"] = media
        if uuids is not None:
            prompt["multi_modal_uuids"] = uuids
        return conversation, prompt

    async def tokenize_prompts_async(self, prompts, unused_params):
        if len(prompts) != 1:
            raise ValueError("a tokenized continuation must contain exactly one prompt")
        prompt = dict(self._tokenized_prompt)
        for key in ("multi_modal_data", "multi_modal_uuids", "mm_processor_kwargs"):
            if key in prompts[0]:
                prompt[key] = prompts[0][key]
            else:
                prompt.pop(key, None)
        return [prompt]


class VLLMDeferredRenderer:
    """Capture a native renderer invocation before any media download."""

    def __init__(self, renderer):
        self._renderer = renderer
        self._prediction = _PredictionRenderer(renderer)
        logger.info("NaviSched deferred native rendering enabled; engine transport entry unchanged")

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    async def render_chat_async(
        self, conversations, chat_params, tok_params=None, *, prompt_extras=None
    ):
        media = [any(
            isinstance(message.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") != "text"
                for part in message["content"]
            ) for message in conversation
        ) for conversation in conversations]
        if not any(media):
            return await self._renderer.render_chat_async(
                conversations, chat_params, tok_params, prompt_extras=prompt_extras
            )
        if not all(media):
            results = await asyncio.gather(*(self.render_chat_async(
                [conversation], chat_params, tok_params, prompt_extras=prompt_extras
            ) for conversation in conversations))
            return ([conversation for conversations, _ in results for conversation in conversations],
                    [prompt for _, prompts in results for prompt in prompts])
        calls = [msgspec.msgpack.encode({
            "version": 1,
            "entry": "render_chat_async",
            "inputs": [conversation],
            "chat_params": asdict(chat_params),
            "tok_params": asdict(tok_params) if tok_params is not None else None,
            "prompt_extras": prompt_extras,
        }) for conversation in conversations]
        conversation_outputs, prompts = await self._prediction.render_chat_async(
            conversations, chat_params, tok_params, prompt_extras=prompt_extras
        )
        for prompt, call in zip(prompts, calls, strict=True):
            continuation = msgspec.msgpack.decode(call)
            continuation["tokenized_prompt"] = prompt.pop("_navi_sched_tokenized_prompt")
            prompt["navi_sched_render_call"] = msgspec.msgpack.encode(continuation)
        return conversation_outputs, prompts

    async def render_cmpl_async(self, prompts, tok_params=None, *, prompt_extras=None):
        media = [isinstance(prompt, dict) and bool(prompt.get("multi_modal_data"))
                 for prompt in prompts]
        if not any(media):
            return await self._renderer.render_cmpl_async(
                prompts, tok_params, prompt_extras=prompt_extras
            )
        if not all(media):
            results = await asyncio.gather(*(self.render_cmpl_async(
                [prompt], tok_params, prompt_extras=prompt_extras
            ) for prompt in prompts))
            return [prompt for outputs in results for prompt in outputs]
        calls = [msgspec.msgpack.encode({
            "version": 1,
            "entry": "render_cmpl_async",
            "inputs": [prompt],
            "tok_params": asdict(tok_params) if tok_params is not None else None,
            "prompt_extras": prompt_extras,
        }) for prompt in prompts]
        outputs = await self._prediction.render_cmpl_async(
            prompts, tok_params, prompt_extras=prompt_extras
        )
        for output, call in zip(outputs, calls, strict=True):
            continuation = msgspec.msgpack.decode(call)
            continuation["tokenized_prompt"] = output.pop("_navi_sched_tokenized_prompt")
            output["navi_sched_render_call"] = msgspec.msgpack.encode(continuation)
        return outputs


def create_deferred_renderer(renderer):
    if os.environ.get("VLLM_NAVI_SCHED_DEFER_RENDER", "0") != "1":
        return renderer
    from vllm.model_executor.models.deepseek_vl import DeepseekVLMultiModalProcessor
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    from vllm.renderers.hf import HfRenderer

    if not isinstance(renderer, HfRenderer) or not isinstance(
        renderer.mm_processor, (DeepseekVLMultiModalProcessor, Qwen3VLMultiModalProcessor)
    ):
        raise ValueError("deferred rendering requires a DeepSeek-VL or Qwen3-VL HfRenderer")
    mm_config = renderer.model_config.get_multimodal_config()
    if mm_config.mm_processor_cache_gb > 0 and mm_config.mm_processor_cache_type == "shm":
        raise ValueError("deferred rendering does not support the shared-memory media transport")
    return VLLMDeferredRenderer(renderer)


class NativeRenderContinuation:
    """Resume whitelisted native calls; URL prediction identities never execute."""

    def __init__(self, vllm_config, supported_tasks):
        from vllm.v1.engine.input_processor import InputProcessor

        self._processor = InputProcessor(vllm_config)
        from vllm.multimodal import MULTIMODAL_REGISTRY

        renderer = self._processor.renderer
        if renderer.mm_processor is not None:
            renderer.shutdown()
            renderer.mm_processor.cache = MULTIMODAL_REGISTRY.processor_only_cache_from_config(
                vllm_config
            )
        self._supported_tasks = supported_tasks
        self._loop = None
        self._renderer_closed = False

    async def _resume_one(self, request, schedule_batch_id=0, local_batch_id=0):
        from vllm.renderers import ChatParams, TokenizeParams

        payload = request.navi_sched_render_call
        if payload is None:
            return request
        started_ns = time.perf_counter_ns()
        render_ms = process_inputs_ms = -1.0
        phase = "decode_call"
        phase_started_ns = started_ns
        status = "failed"
        error_type, http_status, error = "none", 0, ""
        upstream_code, upstream_message = "", ""
        upstream_trace_id, upstream_server, upstream_retry_after = "", "", ""
        try:
            call = msgspec.msgpack.decode(payload)
            if call.get("version") != 1:
                raise ValueError("unsupported native renderer continuation version")
            tok_params = call["tok_params"]
            tok_params = TokenizeParams(**tok_params) if tok_params is not None else None
            renderer = self._processor.renderer
            if "tokenized_prompt" in call:
                renderer = _TokenizedRenderer(
                    renderer, msgspec.msgpack.decode(call["tokenized_prompt"])
                )
            phase = "native_render"
            phase_started_ns = time.perf_counter_ns()
            if call["entry"] == "render_chat_async":
                _, prompts = await renderer.render_chat_async(
                    call["inputs"], ChatParams(**call["chat_params"]), tok_params,
                    prompt_extras=call["prompt_extras"],
                )
            elif call["entry"] == "render_cmpl_async":
                prompts = await renderer.render_cmpl_async(
                    call["inputs"], tok_params, prompt_extras=call["prompt_extras"]
                )
            else:
                raise ValueError("unsupported native renderer continuation entry")
            render_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
            if len(prompts) != 1:
                raise ValueError("a renderer continuation must contain exactly one prompt")
            phase = "process_inputs"
            phase_started_ns = time.perf_counter_ns()
            materialized = self._processor.process_inputs(
                request.request_id, prompts[0], request.params,
                supported_tasks=self._supported_tasks,
                arrival_time=request.arrival_time, lora_request=request.lora_request,
                trace_headers=request.trace_headers, priority=request.priority,
                data_parallel_rank=request.data_parallel_rank, resumable=request.resumable,
            )
            process_inputs_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
            phase = "restore_native_inputs"
            phase_started_ns = time.perf_counter_ns()
            request.prompt_token_ids = materialized.prompt_token_ids
            request.prompt_embeds = materialized.prompt_embeds
            request.mm_features = materialized.mm_features
            request.navi_sched_render_call = None
            status = "ok"
            return request
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            error_type = type(exc).__name__
            http_status = getattr(exc, "status", 0)
            error = str(exc)[:1024]
            headers = getattr(exc, "headers", None) or {}
            upstream_code = headers.get("Error-Code", "")[:128]
            upstream_message = headers.get("Error-Message", "")[:512]
            upstream_trace_id = headers.get("X-Trace-Id", "")[:128]
            upstream_server = headers.get("Server", "")[:128]
            upstream_retry_after = headers.get("Retry-After", "")[:128]
            raise
        finally:
            finished_ns = time.perf_counter_ns()
            total_ms = (finished_ns - started_ns) / 1_000_000.0
            _report_image_preparation(status, http_status, upstream_code)
            if status != "ok" or total_ms >= 100.0:
                logger.info(
                    "NAVI_SCHED_VLLM_NATIVE_RENDER_REQUEST request_id=%s status=%s "
                    "phase=%s phase_ms=%.3f native_render_ms=%.3f "
                    "process_inputs_ms=%.3f total_ms=%.3f "
                    "schedule_batch_id=%d local_batch_id=%d "
                    "error_type=%s http_status=%s error=%r "
                    "upstream_code=%r upstream_message=%r upstream_trace_id=%r "
                    "upstream_server=%r upstream_retry_after=%r",
                    request.request_id, status, phase,
                    (finished_ns - phase_started_ns) / 1_000_000.0,
                    render_ms, process_inputs_ms, total_ms,
                    schedule_batch_id, local_batch_id, error_type, http_status, error,
                    upstream_code, upstream_message, upstream_trace_id,
                    upstream_server, upstream_retry_after,
                )

    async def resume_async(
        self, requests, *, return_exceptions=False, schedule_batch_id=0, local_batch_id=0
    ):
        if self._renderer_closed:
            raise RuntimeError("native render continuation is closed")
        results = await asyncio.gather(
            *(self._resume_one(request, schedule_batch_id, local_batch_id)
              for request in requests), return_exceptions=True
        )
        if not return_exceptions:
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        return results

    def resume_sync(self, requests, *, return_exceptions=False):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        started = time.monotonic()
        try:
            return self._loop.run_until_complete(self.resume_async(
                requests, return_exceptions=return_exceptions
            ))
        finally:
            logger.info(
                "NAVI_SCHED_VLLM_NATIVE_RENDER request_count=%d elapsed_ms=%.3f",
                len(requests), (time.monotonic() - started) * 1000,
            )

    def close(self):
        if self._loop is not None:
            from vllm.connections import global_http_connection

            client = global_http_connection._async_client
            if client is not None and client._loop is self._loop:
                self._loop.run_until_complete(client.close())
                global_http_connection._async_client = None
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()
            self._loop = None
        self._close_renderer()

    def _close_renderer(self):
        if not self._renderer_closed:
            self._processor.renderer.shutdown()
            self._renderer_closed = True

    async def aclose(self):
        from vllm.connections import global_http_connection

        client = global_http_connection._async_client
        if client is not None and client._loop is asyncio.get_running_loop():
            await client.close()
            global_http_connection._async_client = None
        self._close_renderer()
