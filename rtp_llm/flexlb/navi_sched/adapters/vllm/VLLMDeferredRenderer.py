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
                for name in (
                    "completed",
                    "success",
                    "rate_limited",
                    "other_error",
                    "cancelled",
                )
            }
        cancelled = status == "cancelled"
        limited = (
            status == "failed"
            and http_status == 503
            and upstream_code == "RATE_LIMIT_EXCEEDED"
        )
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


def _media_http_error(error):
    """Find HTTP details retained by native media validation exceptions."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "status", 0):
            return current
        current = current.__cause__ or current.__context__
    return error


class _PredictionRenderer:
    """Use native chat/token processing, replacing only media preparation."""

    def __init__(self, renderer):
        self._renderer = renderer
        # Native async wrappers close over the original bound method. Rebind
        # only prediction work, retaining the native renderer's executor.
        for async_name, sync_name in (
            ("_process_multimodal_async", "_process_multimodal"),
            ("_process_for_engine_in_worker", "process_for_engine"),
        ):
            if hasattr(renderer, async_name):
                from vllm.utils.async_utils import make_async

                setattr(
                    self,
                    async_name,
                    make_async(
                        getattr(self, sync_name),
                        executor=getattr(renderer, "_mm_executor", None),
                    ),
                )

    def __getattr__(self, name):
        value = getattr(self._renderer, name)
        if getattr(value, "__self__", None) is self._renderer:
            return value.__func__.__get__(self, type(self))
        return value

    def process_for_engine(self, prompt, arrival_time, **kwargs):
        tokenized = msgspec.msgpack.encode(prompt)
        output = self._renderer.process_for_engine.__func__(
            self, prompt, arrival_time, **kwargs
        )
        output["_navi_sched_tokenized_prompt"] = tokenized
        return output

    async def process_for_engine_async(self, prompt, arrival_time, **kwargs):
        tokenized = msgspec.msgpack.encode(prompt)
        output = await self._renderer.process_for_engine_async.__func__(
            self, prompt, arrival_time, **kwargs
        )
        output["_navi_sched_tokenized_prompt"] = tokenized
        return output

    @staticmethod
    def _chat_mm_tracker_factory(model_config, **kwargs):
        from vllm.entrypoints.chat_utils import (
            AsyncMultiModalContentParser,
            AsyncMultiModalItemTracker,
        )

        class ReferenceParser(AsyncMultiModalContentParser):
            async def _image_with_uuid_async(
                self, image_url, uuid, preprocess_config=None
            ):
                if not image_url:
                    raise ValueError("deferred rendering requires an image URL")
                identity = hashlib.sha256(
                    msgspec.json.encode(
                        ["navi-scheduling-image-url-v1", image_url, preprocess_config]
                    )
                ).hexdigest()
                return None, identity

            async def _audio_with_uuid_async(self, audio_url, uuid):
                return self._media_reference("audio", audio_url)

            async def _video_with_uuid_async(self, video_url, uuid):
                return self._media_reference("video", video_url)

            def _media_reference(self, modality, url):
                identity = hashlib.sha256(
                    msgspec.json.encode(
                        [
                            f"navi-scheduling-{modality}-url-v1",
                            url,
                            self._mm_processor_kwargs,
                            self._tracker.media_io_kwargs,
                        ]
                    )
                ).hexdigest()
                return None, identity

            def parse_video(self, video_url, uuid=None):
                # Keep native placeholder ordering, including audio extracted
                # from a video, without invoking either media loader here.
                placeholder = self._tracker.add(
                    "video", self._video_with_uuid_async(video_url, uuid)
                )
                self._add_placeholder("video", placeholder)
                if (
                    video_url
                    and self._mm_processor_kwargs
                    and self._mm_processor_kwargs.get("use_audio_in_video", False)
                ):
                    placeholder = self._tracker.add(
                        "audio", self._audio_with_uuid_async(video_url, uuid)
                    )
                    self._add_placeholder("audio", placeholder)

            def parse_video_audio_together(self, video_url, uuid=None):
                self.parse_video(video_url, uuid)

        class ReferenceTracker(AsyncMultiModalItemTracker):
            def create_parser(self, mm_processor_kwargs=None):
                return ReferenceParser(self, mm_processor_kwargs=mm_processor_kwargs)

        return ReferenceTracker(model_config, **kwargs)

    def _process_multimodal(
        self,
        prompt,
        mm_data,
        mm_uuids,
        mm_processor_kwargs,
        tokenization_kwargs,
        *,
        skip_mm_cache=False,
    ):
        try:
            from vllm.inputs import mm_input as make_mm_input
        except ImportError:
            from vllm.multimodal.inputs import mm_inputs as make_mm_input
        from vllm.multimodal.processing import PromptReplacement

        token_ids = _media_token_ids(self._renderer)
        counts = {
            modality: len(identities)
            for modality, identities in (mm_uuids or {}).items()
        }
        if (
            not counts
            or set(mm_data) != set(counts)
            or not counts.keys() <= token_ids.keys()
        ):
            raise ValueError("deferred media placeholders do not match the model")
        processor = self._renderer.get_mm_processor()
        updates = [
            PromptReplacement(
                modality=modality,
                target=[token_ids[modality]],
                replacement=[token_ids[modality]],
            )
            for modality in counts
        ]
        grouped = processor._bind_and_group_updates(updates, counts)
        tokens = processor._apply_hf_processor_tokens_only(prompt)
        tokens, positions = processor._apply_prompt_updates(tokens, grouped)
        processor._validate_mm_placeholders(positions, counts)
        return make_mm_input(
            prompt_token_ids=tokens,
            mm_kwargs={modality: [None] * count for modality, count in counts.items()},
            mm_hashes=mm_uuids,
            mm_placeholders={
                modality: [item.to_range() for item in positions[modality]]
                for modality in counts
            },
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
            messages,
            self.model_config,
            content_format=resolve_chat_template_content_format(
                chat_template=params.chat_template,
                tools=params.chat_template_kwargs.get("tools"),
                given_format=params.chat_template_content_format,
                tokenizer=self.get_tokenizer(),
                model_config=self.model_config,
            ),
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs or {},
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


def _media_token_ids(renderer):
    config = renderer.mm_processor.info.get_hf_config()
    config = getattr(config, "thinker_config", config)
    tokens = {}
    for modality in ("image", "video", "audio"):
        for suffix in ("token_id", "token_index"):
            token = getattr(config, f"{modality}_{suffix}", None)
            if isinstance(token, int) and token >= 0:
                tokens[modality] = token
                break
    return tokens


def _can_defer_conversation(conversation, supported_modalities, processor_kwargs):
    modalities = set()
    for message in conversation:
        if not isinstance(message, dict):
            return False
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            return False
        for part in content:
            if not isinstance(part, dict):
                return False
            kind = part.get("type")
            if kind == "text":
                continue
            if kind in ("image_url", "video_url", "audio_url"):
                modality = kind.removesuffix("_url")
                reference = part.get(kind)
                url = reference.get("url") if isinstance(reference, dict) else reference
                if not isinstance(url, str) or not url:
                    return False
                modalities.add(modality)
            elif kind == "input_audio":
                audio = part.get("input_audio")
                if (
                    not isinstance(audio, dict)
                    or not audio.get("data")
                    or not audio.get("format")
                ):
                    return False
                modalities.add("audio")
            else:
                # Embeddings, decoded objects and future content types retain
                # the native renderer and its validation/error behavior.
                return False
    if "video" in modalities and processor_kwargs.get("use_audio_in_video", False):
        modalities.add("audio")
    return bool(modalities) and modalities <= supported_modalities


class VLLMDeferredRenderer:
    """Capture a native renderer invocation before any media download."""

    def __init__(self, renderer):
        self._renderer = renderer
        self._prediction = _PredictionRenderer(renderer)
        logger.info(
            "NaviSched deferred native rendering enabled; engine transport entry unchanged"
        )

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    async def render_chat_async(
        self, conversations, chat_params, tok_params=None, *, prompt_extras=None
    ):
        processor_kwargs = getattr(chat_params, "mm_processor_kwargs", None) or {}
        supported = _media_token_ids(self._renderer).keys()
        media = [
            _can_defer_conversation(conversation, supported, processor_kwargs)
            for conversation in conversations
        ]
        if not any(media):
            return await self._renderer.render_chat_async(
                conversations, chat_params, tok_params, prompt_extras=prompt_extras
            )
        if not all(media):
            results = await asyncio.gather(
                *(
                    self.render_chat_async(
                        [conversation],
                        chat_params,
                        tok_params,
                        prompt_extras=prompt_extras,
                    )
                    for conversation in conversations
                )
            )
            return (
                [
                    conversation
                    for conversations, _ in results
                    for conversation in conversations
                ],
                [prompt for _, prompts in results for prompt in prompts],
            )
        calls = [
            msgspec.msgpack.encode(
                {
                    "version": 1,
                    "entry": "render_chat_async",
                    "inputs": [conversation],
                    "chat_params": asdict(chat_params),
                    "tok_params": (
                        asdict(tok_params) if tok_params is not None else None
                    ),
                    "prompt_extras": prompt_extras,
                }
            )
            for conversation in conversations
        ]
        conversation_outputs, prompts = await self._prediction.render_chat_async(
            conversations, chat_params, tok_params, prompt_extras=prompt_extras
        )
        for prompt, call in zip(prompts, calls, strict=True):
            continuation = msgspec.msgpack.decode(call)
            continuation["tokenized_prompt"] = prompt.pop(
                "_navi_sched_tokenized_prompt"
            )
            prompt["navi_sched_render_call"] = msgspec.msgpack.encode(continuation)
        return conversation_outputs, prompts

    async def render_cmpl_async(self, prompts, tok_params=None, *, prompt_extras=None):
        return await self._renderer.render_cmpl_async(
            prompts, tok_params, prompt_extras=prompt_extras
        )


def create_deferred_renderer(renderer):
    if os.environ.get("VLLM_NAVI_SCHED_DEFER_RENDER", "0") != "1":
        return renderer
    if getattr(renderer, "mm_processor", None) is None:
        return renderer
    from vllm.renderers.hf import HfRenderer

    if not isinstance(renderer, HfRenderer) or not _media_token_ids(renderer):
        return renderer
    mm_config = renderer.model_config.get_multimodal_config()
    if (
        mm_config.mm_processor_cache_gb > 0
        and mm_config.mm_processor_cache_type == "shm"
    ) or getattr(renderer.model_config.hf_config, "use_unified_vision_chunk", False):
        return renderer
    return VLLMDeferredRenderer(renderer)


class NativeRenderContinuation:
    """Resume whitelisted native calls; URL prediction identities never execute."""

    def __init__(self, vllm_config, supported_tasks):
        from vllm.v1.engine.input_processor import InputProcessor

        self._processor = InputProcessor(vllm_config)
        from vllm.multimodal import MULTIMODAL_REGISTRY

        renderer = self._processor.renderer
        if renderer.mm_processor is not None:
            previous_cache = renderer.mm_processor.cache
            if previous_cache is not None:
                previous_cache.close()
            renderer.mm_processor.cache = (
                MULTIMODAL_REGISTRY.processor_only_cache_from_config(vllm_config)
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
            tok_params = (
                TokenizeParams(**tok_params) if tok_params is not None else None
            )
            renderer = self._processor.renderer
            if "tokenized_prompt" in call:
                renderer = _TokenizedRenderer(
                    renderer, msgspec.msgpack.decode(call["tokenized_prompt"])
                )
            phase = "native_render"
            phase_started_ns = time.perf_counter_ns()
            if call["entry"] == "render_chat_async":
                _, prompts = await renderer.render_chat_async(
                    call["inputs"],
                    ChatParams(**call["chat_params"]),
                    tok_params,
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
                raise ValueError(
                    "a renderer continuation must contain exactly one prompt"
                )
            phase = "process_inputs"
            phase_started_ns = time.perf_counter_ns()
            materialized = self._processor.process_inputs(
                request.request_id,
                prompts[0],
                request.params,
                supported_tasks=self._supported_tasks,
                arrival_time=request.arrival_time,
                lora_request=request.lora_request,
                trace_headers=request.trace_headers,
                priority=request.priority,
                data_parallel_rank=request.data_parallel_rank,
                resumable=request.resumable,
            )
            process_inputs_ms = (
                time.perf_counter_ns() - phase_started_ns
            ) / 1_000_000.0
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
            http_error = _media_http_error(exc)
            http_status = getattr(http_error, "status", 0)
            error = str(http_error)[:1024]
            headers = getattr(http_error, "headers", None) or {}
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
                    request.request_id,
                    status,
                    phase,
                    (finished_ns - phase_started_ns) / 1_000_000.0,
                    render_ms,
                    process_inputs_ms,
                    total_ms,
                    schedule_batch_id,
                    local_batch_id,
                    error_type,
                    http_status,
                    error,
                    upstream_code,
                    upstream_message,
                    upstream_trace_id,
                    upstream_server,
                    upstream_retry_after,
                )

    async def resume_async(
        self,
        requests,
        *,
        return_exceptions=False,
        schedule_batch_id=0,
        local_batch_id=0,
    ):
        if self._renderer_closed:
            raise RuntimeError("native render continuation is closed")
        results = await asyncio.gather(
            *(
                self._resume_one(request, schedule_batch_id, local_batch_id)
                for request in requests
            ),
            return_exceptions=True,
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
            return self._loop.run_until_complete(
                self.resume_async(requests, return_exceptions=return_exceptions)
            )
        finally:
            logger.info(
                "NAVI_SCHED_VLLM_NATIVE_RENDER request_count=%d elapsed_ms=%.3f",
                len(requests),
                (time.monotonic() - started) * 1000,
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
