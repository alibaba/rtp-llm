from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from navi_sched.util import naviSchedModule

from .NaviSchedAdapterConfig import NaviSchedAdapterConfig
from .NaviSchedClient import NaviSchedClient
from .VLLMAdapterUtils import NAVI_SCHED_MEDIA_ERROR_PREFIX, make_kmon_tags
from .VLLMCachePolicy import get_vllm_hash_block_size

logger = logging.getLogger("vllm.navi_sched.client")

NAVI_SCHED_RETRYABLE_ERROR_PREFIX = "navi_sched_retryable:"
NAVI_SCHED_ADMISSION_TIMEOUT = "NaviSched request admission timed out"


def _make_client_kmon_tags(client_index: int) -> dict[str, str]:
    tags = make_kmon_tags("client")
    tags["client_index"] = str(client_index)
    return tags


def _make_route_rejection(decision: Any) -> Exception:
    reason = str(getattr(decision, "reason", "") or "")
    from vllm.whale_vllm.vllm_server.utils.concurrency_controller import (
        ConcurrencyException,
    )

    if reason == "rate_limited":
        return ConcurrencyException("NaviSched rate limited request")
    if reason == "no_available_node":
        return ConcurrencyException("NaviSched has no available node")
    if reason == "reservation_failed":
        return ConcurrencyException("NaviSched reservation failed")
    if reason == "master_restarting":
        return ConcurrencyException("NaviSched master is restarting")
    detail = reason or "missing target"
    return ConcurrencyException(
        f"NaviSched scheduling rejected request: {detail}"
    )


def _make_route_failure(error: Exception) -> Exception:
    from vllm.whale_vllm.vllm_server.utils.concurrency_controller import (
        ConcurrencyException,
    )

    message = str(error)
    if NAVI_SCHED_ADMISSION_TIMEOUT in message:
        return ConcurrencyException(NAVI_SCHED_ADMISSION_TIMEOUT)
    if isinstance(error, TimeoutError):
        return ConcurrencyException(
            f"NaviSched routing timed out: {message or 'unknown timeout'}"
        )
    return ConcurrencyException(
        f"NaviSched routing failed: {message or 'unknown scheduling error'}"
    )


class VLLMEngineTransport:
    """Routes requests and streaming EngineCoreOutputs through one Navi graph."""

    def __init__(self, vllm_config: Any, engine_core: Any):
        import msgspec

        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self._engine_core = engine_core
        self._render_config = vllm_config
        self._render_continuation = None
        self._render_init_lock = asyncio.Lock()
        self._msgspec = msgspec
        self._adapter_config = NaviSchedAdapterConfig.from_env(
            block_size=get_vllm_hash_block_size(vllm_config)
        )
        client_index = int(getattr(engine_core, "client_index", 0))
        self._client = NaviSchedClient(
            self._adapter_config,
            enable_agent_forward=True,
            kmon_tags=_make_client_kmon_tags(client_index),
        )
        self._encoder = MsgpackEncoder()
        self._decoder = MsgpackDecoder(EngineCoreOutputs)
        self._engine_core_output_type = EngineCoreOutput
        self._engine_core_outputs_type = EngineCoreOutputs
        self._finish_reason = FinishReason
        self._response_kind = naviSchedModule().AgentResponseKind
        self._outputs: asyncio.Queue[Any] = asyncio.Queue()
        self._request_ids: dict[str, int] = {}
        self._external_request_ids: dict[int, str] = {}
        self._local_output_task: asyncio.Task | None = None
        self._remote_output_task: asyncio.Task | None = None
        self._response_poll_lock = threading.Lock()
        self._response_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="NaviSchedResponse",
        )
        self._closed = False
        self._fallback_metric_warning_logged = False
        logger.info(
            "enabled NaviSched VLLM frontend transport: client_index=%d "
            "response_transport=navi_stream",
            client_index,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine_core, name)

    async def add_request_async(self, request: Any) -> None:
        if request.prompt_token_ids is None:
            raise ValueError("NaviSched requires prompt token ids")
        external_request_id = str(request.request_id)
        if external_request_id in self._request_ids:
            raise ValueError(f"duplicate NaviSched VLLM request: {external_request_id}")
        request.navi_sched_local_fallback = False
        if not self._client.has_master_biz():
            await self._submit_local_fallback(request, "Master biz absent from discovery")
            return
        request_id = self._allocate_graph_request_id()
        self._ensure_remote_output_task()
        frames = [bytes(value) for value in self._encoder.encode(request)]
        payload = self._msgspec.msgpack.encode(frames)
        self._request_ids[external_request_id] = request_id
        self._external_request_ids[request_id] = external_request_id
        route_task = asyncio.create_task(
            self._client.route_async(
                request.prompt_token_ids,
                request_id=request_id,
                block_hash_request=request,
                forward_payload=payload,
            )
        )
        try:
            decision = await asyncio.shield(route_task)
        except asyncio.CancelledError:
            try:
                await route_task
            except Exception:
                logger.exception(
                    "failed to terminate NaviSched route after source cancellation: "
                    "request_id=%s",
                    external_request_id,
                )
            finally:
                self._abort_request(external_request_id)
            raise
        except Exception as error:
            self._forget_request(external_request_id, request_id)
            logger.exception(
                "NaviSched VLLM client route failed: request_id=%s "
                "master_addresses=%s master_vip_domain=%s target_biz=%s",
                external_request_id,
                self._adapter_config.master_addresses,
                self._adapter_config.master_vip_domain,
                self._adapter_config.target_biz_name,
            )
            raise _make_route_failure(error) from error
        if decision.target is None or not decision.target.address:
            self._forget_request(external_request_id, request_id)
            raise _make_route_rejection(decision)

    async def get_output_async(self) -> Any:
        output = await self._outputs.get()
        if isinstance(output, BaseException):
            raise output
        return output

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        local_request_ids = []
        for request_id in request_ids:
            request_id = str(request_id)
            logger.info(
                "[navi-vllm-client-abort] source=api request_id=%s "
                "navi_request_id=%s",
                request_id,
                self._request_ids.get(request_id),
            )
            if not self._abort_request(request_id):
                local_request_ids.append(request_id)
        if local_request_ids:
            await self._engine_core.abort_requests_async(local_request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for task in (self._local_output_task, self._remote_output_task):
            if task is not None:
                task.cancel()
        with self._response_poll_lock:
            self._client.close()
        self._response_executor.shutdown(wait=True, cancel_futures=True)
        self._request_ids.clear()
        self._external_request_ids.clear()
        continuation = getattr(self, "_render_continuation", None)
        if continuation is not None:
            continuation.close()
        if timeout is None:
            self._engine_core.shutdown()
        else:
            self._engine_core.shutdown(timeout=timeout)

    def _abort_request(self, external_request_id: str) -> bool:
        request_id = self._request_ids.get(external_request_id)
        if request_id is None:
            return False
        try:
            self._client.abort_forward_request(request_id)
        finally:
            self._forget_request(external_request_id, request_id)
        return True

    def _forget_request(self, external_request_id: str, request_id: int) -> None:
        if self._request_ids.get(external_request_id) == request_id:
            self._request_ids.pop(external_request_id, None)
        if self._external_request_ids.get(request_id) == external_request_id:
            self._external_request_ids.pop(request_id, None)

    def _allocate_graph_request_id(self) -> int:
        request_id = int(self._client.allocate_request_id())
        if not 0 < request_id < (1 << 64):
            raise RuntimeError("NaviSched client allocated an invalid request id")
        if request_id in self._external_request_ids:
            raise RuntimeError("NaviSched client allocated an active request id")
        return request_id

    async def get_native_render_continuation(self):
        if self._closed:
            raise RuntimeError("NaviSched VLLM transport is closed")
        if self._render_continuation is None:
            from .VLLMDeferredRenderer import NativeRenderContinuation

            async with self._render_init_lock:
                if self._render_continuation is None:
                    tasks = await self._engine_core.get_supported_tasks_async()
                    if self._closed:
                        raise RuntimeError("NaviSched VLLM transport is closed")
                    self._render_continuation = NativeRenderContinuation(
                        self._render_config, tasks,
                    )
        return self._render_continuation

    async def _submit_local_fallback(self, request: Any, error: Exception | str) -> None:
        if getattr(request, "navi_sched_render_call", None) is not None:
            continuation = await self.get_native_render_continuation()
            await continuation.resume_async([request])
        request.navi_sched_local_fallback = True
        self._ensure_local_output_task()
        try:
            self._client.report_local_fallback()
        except Exception:
            # A telemetry failure must not block native request submission.
            if not getattr(self, "_fallback_metric_warning_logged", False):
                self._fallback_metric_warning_logged = True
                logger.warning(
                    "NaviSched local fallback metric reporting failed; "
                    "continuing with native vLLM scheduling",
                    exc_info=True,
                )
        logger.warning(
            "falling back to local vLLM EngineCore for this request: "
            "request_id=%s reason=%s",
            request.request_id,
            error,
        )
        await self._engine_core.add_request_async(request)

    def _ensure_local_output_task(self) -> None:
        if self._closed:
            raise RuntimeError("NaviSched VLLM transport is closed")
        if self._local_output_task is None:
            self._local_output_task = asyncio.create_task(
                self._pump_local_outputs(), name="VLLMNaviSchedLocalOutputs"
            )

    def _ensure_remote_output_task(self) -> None:
        if self._closed:
            raise RuntimeError("NaviSched VLLM transport is closed")
        if self._remote_output_task is None:
            self._remote_output_task = asyncio.create_task(
                self._pump_remote_outputs(), name="VLLMNaviSchedRemoteOutputs"
            )

    async def _pump_local_outputs(self) -> None:
        try:
            while True:
                await self._outputs.put(await self._engine_core.get_output_async())
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - local engine failure
            logger.exception("NaviSched VLLM local output pump failed")
            await self._outputs.put(error)

    async def _pump_remote_outputs(self) -> None:
        try:
            while True:
                response = await asyncio.get_running_loop().run_in_executor(
                    self._response_executor,
                    self._take_next_forward_response,
                    100,
                )
                if response is None:
                    continue
                request_id = int(response.request_id)
                external_request_id = self._external_request_ids.get(request_id)
                if external_request_id is None:
                    logger.warning(
                        "dropping NaviSched response for inactive request: "
                        "request_id=%d",
                        request_id,
                    )
                    continue
                try:
                    output = self._decode_remote_response(
                        response, external_request_id
                    )
                except Exception:
                    logger.exception(
                        "invalid NaviSched VLLM response: request_id=%s",
                        external_request_id,
                    )
                    self._client.abort_forward_request(request_id)
                    self._forget_request(external_request_id, request_id)
                    output = self._make_error_output(
                        external_request_id, "invalid NaviSched VLLM response"
                    )
                else:
                    if response.terminal:
                        self._forget_request(external_request_id, request_id)
                await self._outputs.put(output)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - forward transport failure
            logger.exception("NaviSched VLLM response pump failed")
            await self._outputs.put(error)

    def _take_next_forward_response(self, timeout_ms: int):
        with self._response_poll_lock:
            if self._closed:
                return None
            return self._client.take_next_forward_response(timeout_ms)

    def _decode_remote_response(self, response, external_request_id: str):
        if response.kind == self._response_kind.ERROR:
            if not response.terminal or response.frames:
                raise ValueError("invalid terminal NaviSched error response")
            error_message = str(response.error_message or "")
            logger.warning(
                "NAVI_SCHED_VLLM_REMOTE_REQUEST_FAILED navi_request_id=%d "
                "external_request_id=%s category=%s",
                response.request_id,
                external_request_id,
                "media_preprocessing"
                if error_message.startswith(NAVI_SCHED_MEDIA_ERROR_PREFIX)
                else "remote_execution",
            )
            return self._make_error_output(
                external_request_id, error_message
            )
        if response.kind != self._response_kind.OUTPUT or not response.frames:
            raise ValueError(f"unexpected NaviSched response kind: {response.kind}")

        output = self._decoder.decode([bytes(frame) for frame in response.frames])
        output_request_ids = {str(value.request_id) for value in output.outputs}
        output_request_ids.update(str(value) for value in output.finished_requests or ())
        if len(output_request_ids) > 1:
            raise ValueError("NaviSched response contains another request")
        terminal = bool(output.finished_requests) or any(
            value.finished for value in output.outputs
        )
        if bool(response.terminal) != terminal:
            raise ValueError("NaviSched response terminal state does not match payload")
        for value in output.outputs:
            value.request_id = external_request_id
        if output.finished_requests:
            output.finished_requests = {external_request_id}
        return output

    def _make_error_output(self, external_request_id: str, error_message: str):
        error_message = error_message or "NaviSched remote request failed"
        # Native rendering failures are execution errors, not scheduling limits.
        stop_reason = (
            error_message.removeprefix(NAVI_SCHED_MEDIA_ERROR_PREFIX)
            if error_message.startswith(NAVI_SCHED_MEDIA_ERROR_PREFIX)
            else NAVI_SCHED_RETRYABLE_ERROR_PREFIX + error_message
        )
        output = self._engine_core_output_type(
            request_id=external_request_id,
            new_token_ids=[],
            finish_reason=self._finish_reason.ERROR,
            stop_reason=stop_reason,
        )
        return self._engine_core_outputs_type(
            outputs=[output],
            finished_requests={external_request_id},
        )


def create_vllm_engine_transport(vllm_config: Any, engine_core: Any):
    return VLLMEngineTransport(vllm_config, engine_core)
