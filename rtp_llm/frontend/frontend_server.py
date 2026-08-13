import asyncio
import json
import logging
import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, Union

from fastapi import Request
from fastapi import Request as RawRequest
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel

from rtp_llm.access_logger.access_logger import AccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import (
    update_stop_words_from_env,
    update_tokenizer_special_tokens,
)
from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import SpecialTokens, TaskType
from rtp_llm.server.misc import format_exception
from rtp_llm.structure.request_extractor import request_id_field_name
from rtp_llm.utils.complete_response_async_generator import (
    CloseDependencyRegistry,
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.base_model_datatypes import RequestDeadlineAnchor
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    ConcurrencyLease,
    get_global_controller,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import check_with_info

USAGE_HEADER = "USAGE"


class _AdmissionState(Enum):
    ENTRY = auto()
    STREAMING = auto()
    CLEANUP = auto()
    RELEASED = auto()
    RETAINED = auto()


class _AdmissionOwner:
    def __init__(self, lease: ConcurrencyLease) -> None:
        self._lease = lease
        self._state = _AdmissionState.ENTRY
        self._state_lock = threading.Lock()

    def _transition(self, expected, target) -> None:
        with self._state_lock:
            if self._state is not expected:
                raise RuntimeError(
                    f"invalid admission ownership transition: {self._state} -> {target}"
                )
            self._state = target

    def transfer_to_streaming(self) -> None:
        self._transition(_AdmissionState.ENTRY, _AdmissionState.STREAMING)

    def transfer_to_cleanup(self) -> None:
        self._transition(_AdmissionState.ENTRY, _AdmissionState.CLEANUP)

    def release_if_entry(self) -> bool:
        with self._state_lock:
            if self._state is not _AdmissionState.ENTRY:
                return False
            self._state = _AdmissionState.RELEASED
        return self._lease.release()

    def release_after_cleanup(self) -> bool:
        with self._state_lock:
            if self._state not in (
                _AdmissionState.STREAMING,
                _AdmissionState.CLEANUP,
            ):
                return False
            self._state = _AdmissionState.RELEASED
        return self._lease.release()

    def retain_after_cleanup_failure(self) -> None:
        with self._state_lock:
            if self._state in (
                _AdmissionState.STREAMING,
                _AdmissionState.CLEANUP,
            ):
                self._state = _AdmissionState.RETAINED


class AdmissionStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content,
        response: CompleteResponseAsyncGenerator,
        admission_owner: _AdmissionOwner,
        **kwargs,
    ):
        super().__init__(content, **kwargs)
        self._response = response
        self._admission_owner = admission_owner
        self._admission_owner.transfer_to_streaming()
        self._cleanup_task = None

    async def _close_and_release(self) -> None:
        body_close_error = None
        try:
            close_body = getattr(self.body_iterator, "aclose", None)
            if close_body is not None:
                await close_body()
        except BaseException as e:
            body_close_error = e
            logging.warning("failed to close streaming response body: %s", e)

        try:
            await self._response.aclose()
        except BaseException as e:
            self._admission_owner.retain_after_cleanup_failure()
            logging.warning("failed to close streaming response: %s", e)
            raise

        self._admission_owner.release_after_cleanup()
        if body_close_error is not None:
            raise body_close_error

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._close_and_release())
            cancelled_error = None
            while True:
                try:
                    await asyncio.shield(self._cleanup_task)
                    break
                except asyncio.CancelledError as e:
                    if self._cleanup_task.done():
                        self._cleanup_task.result()
                        raise
                    if cancelled_error is None:
                        cancelled_error = e
                except BaseException:
                    raise
            if cancelled_error is not None:
                raise cancelled_error


class FrontendServer(object):
    def __init__(
        self,
        rank_id: int,
        server_id: int,
        py_env_configs=None,
    ):
        self.py_env_configs = py_env_configs
        self._access_logger = AccessLogger(
            get_log_path(),
            py_env_configs.profiling_debug_logging_config.log_file_backup_count,
            rank_id,
            server_id,
        )
        self._frontend_worker = None
        self._openai_endpoint = None
        self._embedding_endpoint = None
        self.is_embedding = False
        self.thread_lock_ = threading.Lock()
        self._global_controller = get_global_controller()
        self.rank_id = str(rank_id)
        self.server_id = str(server_id)
        kmonitor.init()

    def start(self):
        if (
            self.py_env_configs.profiling_debug_logging_config.debug_start_fake_process
            == 1
        ):
            # for debug online
            logging.info("DEBUG_START_FAKE_PROCESS is set, start fake server")
            self._frontend_worker = None
            return

        model_config = ModelFactory.create_model_config(
            model_args=self.py_env_configs.model_args,
            lora_config=self.py_env_configs.lora_config,
            kv_cache_config=self.py_env_configs.kv_cache_config,
            profiling_debug_logging_config=self.py_env_configs.profiling_debug_logging_config,
            generate_env_config=self.py_env_configs.generate_env_config,
            embedding_config=self.py_env_configs.embedding_config,
            quantization_config=self.py_env_configs.quantization_config,
            render_config=self.py_env_configs.render_config,
            vit_config=self.py_env_configs.vit_config,
        )

        # Create a temporary tokenizer to initialize special_tokens
        # We'll update it with the actual tokenizer after FrontendWorker is created
        special_tokens = SpecialTokens()
        if self.py_env_configs.generate_env_config:
            update_stop_words_from_env(
                special_tokens, self.py_env_configs.generate_env_config
            )

        # Create FrontendWorker with special_tokens and config
        self._frontend_worker = FrontendWorker(
            self.py_env_configs,
            model_config,
            special_tokens,
        )

        # Update special_tokens with actual tokenizer
        update_tokenizer_special_tokens(special_tokens, self._frontend_worker.tokenizer)

        # Only initialize OpenaiEndpoint for LANGUAGE_MODEL task type
        if model_config.task_type == TaskType.LANGUAGE_MODEL:
            # Update model_config with the latest values
            model_config.special_tokens = special_tokens
            model_config.generate_env_config = self.py_env_configs.generate_env_config
            model_config.render_config = self.py_env_configs.render_config
            model_config.model_name = self.py_env_configs.model_args.model_type
            model_config.template_type = None

            self._openai_endpoint = OpenaiEndpoint(
                model_config=model_config,
                misc_config=self.py_env_configs.misc_config,
                vit_config=self.py_env_configs.vit_config,
                tokenizer=self._frontend_worker.tokenizer,
                backend_rpc_server_visitor=self._frontend_worker.backend_rpc_server_visitor,
            )
        else:
            from rtp_llm.embedding.embedding_endpoint import EmbeddingEndpoint

            self._embedding_endpoint = EmbeddingEndpoint(
                model_config=model_config,
                grpc_config=self.py_env_configs.grpc_config,
                server_config=self.py_env_configs.server_config,
                tokenizer=self._frontend_worker.tokenizer,
            )
            self.is_embedding = True

    def stop(self):
        if self._frontend_worker is not None:
            self._frontend_worker.stop()

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        start_time = time.time()
        try:
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(
                AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")}
            )
        except Exception as e:
            return self._handle_exception(request, e)

        try:
            admission_lease = self._global_controller.acquire()
        except ConcurrencyException as e:
            return self._handle_exception(request, e)

        try:
            try:
                sequence = admission_lease.sequence % 4096  # 12 bits
                request[request_id_field_name] = generate_request_id(
                    self.py_env_configs.server_config.ip,
                    self.py_env_configs.server_config.server_port,
                    self.server_id,
                    sequence,
                )
            except Exception as e:
                return self._handle_exception(request, e)

            try:
                assert (
                    self._embedding_endpoint is not None
                ), "embedding pipeline should not be None"
                result, logable_result = await self._embedding_endpoint.embedding(request)
                # do not log result since too big
                if logable_result is not None:
                    self._access_logger.log_success_access(request, logable_result)
                end_time = time.time()
                kmonitor.report(
                    GaugeMetrics.LANTENCY_METRIC, (end_time - start_time) * 1000
                )
                kmonitor.report(
                    AccMetrics.SUCCESS_QPS_METRIC,
                    1,
                    {"source": request.get("source", "unknown")},
                )
                usage = result.get("usage", {})
                if not isinstance(usage, dict):
                    usage = {}
                return ORJSONResponse(
                    result, headers={USAGE_HEADER: json.dumps(usage)}
                )
            except BaseException as e:
                return self._handle_exception(request, e)
        finally:
            admission_lease.release()

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146
    async def stream_response(
        self,
        request: Dict[str, Any],
        response: CompleteResponseAsyncGenerator,
        close_response: bool = True,
    ):
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            async for res in response:
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            await self._collect_complete_response_and_record_access_log(
                request, response
            )
            if is_openai_response:
                yield "data: [DONE]\r\n\r\n"
            else:
                yield f"data:[done]\r\n\r\n"
        except GeneratorExit:
            raise
        except asyncio.CancelledError as e:
            self._access_logger.log_exception_access(request, e)
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                },
            )
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            self._access_logger.log_exception_access(request, e)
            format_e = format_exception(e)
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unkown"),
                    "error_code": str(format_e.get("error_code_str", -1)),
                },
            )
            yield response_data_prefix + json.dumps(
                format_e, ensure_ascii=False
            ) + "\r\n\r\n"
        finally:
            if close_response:
                try:
                    await response.aclose()
                except asyncio.CancelledError:
                    pass
                except BaseException as e:
                    logging.warning("failed to close streaming response: %s", e)

    async def inference(self, req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
        except Exception as e:
            return self._handle_exception(req, e)

        try:
            admission_lease = self._global_controller.acquire()
        except ConcurrencyException as e:
            return self._handle_exception(req, e)

        admission_owner = _AdmissionOwner(admission_lease)
        try:
            try:
                sequence = admission_lease.sequence % 4096  # 12 bits
                req[request_id_field_name] = generate_request_id(
                    self.py_env_configs.server_config.ip,
                    self.py_env_configs.server_config.server_port,
                    self.server_id,
                    sequence,
                )
            except Exception as e:
                return self._handle_exception(req, e)

            def generate_call():
                assert self._frontend_worker is not None
                return self._frontend_worker.inference(**req)

            rep = await self._infer_wrap(
                req, raw_request, generate_call, admission_owner
            )
            return rep
        finally:
            admission_owner.release_if_entry()

    async def _infer_wrap(
        self,
        req: Dict[str, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
        admission_owner: _AdmissionOwner,
    ):
        try:
            rep = await self._infer_impl(
                req, raw_request, generate_call, admission_owner
            )
        except BaseException as e:
            rep = self._handle_exception(req, e)
        return rep

    async def chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        request_dict: Dict[str, Any] = {}
        try:
            admission_lease = self._global_controller.acquire()
        except ConcurrencyException as e:
            return self._handle_exception(request_dict, e)

        admission_owner = _AdmissionOwner(admission_lease)
        try:
            try:
                sequence = admission_lease.sequence % 4096  # 12 bits
                request_deadline_anchor = RequestDeadlineAnchor.now()
                request_dict = request.model_dump(exclude_none=True)
                request_id = generate_request_id(
                    self.py_env_configs.server_config.ip,
                    self.py_env_configs.server_config.server_port,
                    self.server_id,
                    sequence,
                )
            except Exception as e:
                return self._handle_exception(request_dict, e)

            def generate_call():
                assert self._openai_endpoint != None
                response = self._openai_endpoint.chat_completion(
                    request_id, request, raw_request, request_deadline_anchor
                )
                assert isinstance(
                    response, CompleteResponseAsyncGenerator
                ), f"error type: {type(response)}"
                return response

            request_dict[request_id_field_name] = request_id
            rep = await self._infer_wrap(
                request_dict, raw_request, generate_call, admission_owner
            )
            return rep
        finally:
            admission_owner.release_if_entry()

    async def batch_chat_completion(self, request, raw_request: Request):
        from rtp_llm.openai.api_datatype import BatchChatCompletionResponse

        request_dict: Dict[str, Any] = {}
        try:
            admission_lease = self._global_controller.acquire()
        except ConcurrencyException as e:
            return self._handle_exception(request_dict, e)

        try:
            try:
                sequence = admission_lease.sequence % 4096
                request_deadline_anchor = RequestDeadlineAnchor.now()
                request_id = generate_request_id(
                    self.py_env_configs.server_config.ip,
                    self.py_env_configs.server_config.server_port,
                    self.server_id,
                    sequence,
                )
            except Exception as e:
                return self._handle_exception(request_dict, e)

            assert self._openai_endpoint is not None
            responses = await self._openai_endpoint.batch_chat_completion(
                request_id, request, request_deadline_anchor
            )
            return ORJSONResponse(
                content=BatchChatCompletionResponse(
                    responses=[r.model_dump(exclude_none=True) for r in responses]
                ).model_dump()
            )
        finally:
            admission_lease.release()

    async def batch_infer(self, req: dict, raw_request: Request):
        from rtp_llm.frontend.frontend_worker import BatchPipelineResponse

        # Frontend admission currently counts an HTTP batch as one request.
        try:
            admission_lease = self._global_controller.acquire()
        except ConcurrencyException as e:
            return self._handle_exception(req, e)

        try:
            try:
                sequence = admission_lease.sequence % 4096
                request_id = generate_request_id(
                    self.py_env_configs.server_config.ip,
                    self.py_env_configs.server_config.server_port,
                    self.server_id,
                    sequence,
                )
            except Exception as e:
                return self._handle_exception(req, e)

            assert self._frontend_worker is not None
            prompts = req.get("prompt_batch", [])
            generate_config = req.get("generate_config", {})
            result = await self._frontend_worker.batch_infer(
                prompts=prompts,
                request_id=request_id,
                generate_config=generate_config,
            )
            return ORJSONResponse(content=result.model_dump(exclude_none=True))
        finally:
            admission_lease.release()

    async def chat_render(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            assert self._openai_endpoint != None
            return self._openai_endpoint.chat_render(request)
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def _handle_exception(self, request: Dict[str, Any], e: BaseException):
        exception_json = format_exception(e)
        error_code_str = exception_json.get("error_code_str", "")
        if isinstance(e, ConcurrencyException):
            kmonitor.report(AccMetrics.CONFLICT_QPS_METRIC)
        elif isinstance(e, asyncio.CancelledError):
            kmonitor.report(
                AccMetrics.CANCEL_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                },
            )
            self._access_logger.log_exception_access(request, e)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": request.get("source", "unknown"),
                    "error_code": error_code_str,
                },
            )
            self._access_logger.log_exception_access(request, e)

        status_code = 409 if isinstance(e, ConcurrencyException) else 500
        rep = ORJSONResponse(exception_json, status_code=status_code)
        return rep

    async def _call_generate_with_report(
        self, generate_call: Callable[[], CompleteResponseAsyncGenerator]
    ):
        async def __gen_response_with_report(start_time: float, response_generator):
            last_iterate_time = current_time_ms()
            first_token = True
            iter_count = 0
            try:
                async for response in response_generator:
                    end_time = current_time_ms()
                    if first_token:
                        first_token = False
                        kmonitor.report(
                            GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC,
                            end_time - last_iterate_time,
                        )
                    else:
                        step_output_len = 1
                        if hasattr(response, "aux_info"):
                            if isinstance(response.aux_info, list):
                                step_output_len = 0
                                for info in response.aux_info:
                                    step_output_len += info.get("step_output_len", 1)
                            elif isinstance(response.aux_info, dict):
                                step_output_len = max(
                                    response.aux_info.get("step_output_len", 1),
                                    step_output_len,
                                )

                        kmonitor.report(
                            GaugeMetrics.RESPONSE_ITER_RT_METRIC,
                            (end_time - last_iterate_time) / step_output_len,
                        )
                    kmonitor.report(
                        AccMetrics.ITER_QPS_METRIC,
                        1,
                        {
                            "rank_id": self.rank_id,
                            "server_id": self.server_id,
                        },
                    )
                    last_iterate_time = end_time
                    iter_count += 1
                    yield response
                kmonitor.report(
                    GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count
                )
                kmonitor.report(
                    GaugeMetrics.LANTENCY_METRIC, current_time_ms() - start_time
                )
                kmonitor.report(
                    AccMetrics.SUCCESS_QPS_METRIC,
                    1,
                    {
                        "rank_id": self.rank_id,
                        "server_id": self.server_id,
                    },
                )
            finally:
                await response_generator.aclose()

        assert self._frontend_worker is not None
        start_time = current_time_ms()
        response_generator = generate_call()
        close_dependencies = CloseDependencyRegistry()
        managed_response = close_dependencies.wrap(response_generator)
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, managed_response),
            response_generator._collect_complete_response_func,
            close_dependencies=close_dependencies,
        )

    async def _collect_complete_response_and_record_access_log(
        self, req: Dict[Any, Any], res: Any
    ):
        complete_response = await res.gen_complete_response_once()
        complete_response = (
            complete_response.model_dump(exclude_none=True)
            if isinstance(complete_response, BaseModel)
            else complete_response
        )
        self._access_logger.log_success_access(req, complete_response)

        return complete_response

    @staticmethod
    async def _wait_for_http_disconnect(raw_request: RawRequest) -> None:
        if await raw_request.is_disconnected():
            return
        while True:
            message = await raw_request.receive()
            if message.get("type") == "http.disconnect":
                return

    async def _drain_nonstream_response(
        self, req: Dict[Any, Any], res: CompleteResponseAsyncGenerator
    ):
        async for _ in res:
            pass
        return await self._collect_complete_response_and_record_access_log(req, res)

    async def _cleanup_nonstream_tasks(
        self,
        res: CompleteResponseAsyncGenerator,
        admission_owner: _AdmissionOwner,
        response_task: asyncio.Task,
        disconnect_task: asyncio.Task,
    ) -> None:
        async def cleanup() -> None:
            disconnect_task.cancel()
            if not response_task.done():
                response_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
            try:
                await self._close_nonstream_response(res, admission_owner)
            finally:
                await asyncio.gather(response_task, return_exceptions=True)

        cleanup_task = asyncio.create_task(cleanup())
        cancelled_error = None
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError as e:
                if cleanup_task.done():
                    cleanup_task.result()
                    if cancelled_error is None:
                        cancelled_error = e
                    break
                if cancelled_error is None:
                    cancelled_error = e
            except BaseException:
                raise

        if cancelled_error is not None:
            raise cancelled_error

    async def _infer_impl(
        self,
        req: Dict[Any, Any],
        raw_request: RawRequest,
        generate_call: Callable[[], CompleteResponseAsyncGenerator],
        admission_owner: _AdmissionOwner,
    ):
        assert self._frontend_worker is not None
        kmonitor.report(
            AccMetrics.QPS_METRIC,
            1,
            {
                "rank_id": self.rank_id,
                "server_id": self.server_id,
                "source": req.get("source", "unkown"),
            },
        )
        self._access_logger.log_query_access(req)
        is_streaming = self._frontend_worker.is_streaming(req)
        if is_streaming and await raw_request.is_disconnected():
            raise asyncio.CancelledError("client disconnects")
        res = await self._call_generate_with_report(generate_call)

        if is_streaming:
            return AdmissionStreamingResponse(
                self.stream_response(req, res, close_response=False),
                res,
                admission_owner,
                media_type="text/event-stream",
            )
        admission_owner.transfer_to_cleanup()
        response_task = asyncio.create_task(
            self._drain_nonstream_response(req, res)
        )
        disconnect_task = asyncio.create_task(
            self._wait_for_http_disconnect(raw_request)
        )
        primary_error = None
        complete_response = None
        try:
            await asyncio.wait(
                (response_task, disconnect_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if response_task.done():
                complete_response = response_task.result()
            else:
                disconnect_task.result()
                raise asyncio.CancelledError("client disconnects")
        except BaseException as e:
            primary_error = e
        finally:
            try:
                await self._cleanup_nonstream_tasks(
                    res, admission_owner, response_task, disconnect_task
                )
            except BaseException as close_error:
                if primary_error is not None and close_error is not primary_error:
                    raise close_error from primary_error
                raise

        if primary_error is not None:
            raise primary_error
        return ORJSONResponse(content=complete_response)

    @staticmethod
    async def _close_nonstream_response(
        response: CompleteResponseAsyncGenerator,
        admission_owner: _AdmissionOwner,
    ) -> None:
        close_task = asyncio.create_task(response.aclose())
        cancelled_error = None
        while True:
            try:
                await asyncio.shield(close_task)
                break
            except asyncio.CancelledError as e:
                if close_task.done():
                    try:
                        close_task.result()
                    except BaseException:
                        admission_owner.retain_after_cleanup_failure()
                        raise
                    if cancelled_error is None:
                        cancelled_error = e
                    break
                if cancelled_error is None:
                    cancelled_error = e
            except BaseException:
                admission_owner.retain_after_cleanup_failure()
                raise

        admission_owner.release_after_cleanup()
        if cancelled_error is not None:
            raise cancelled_error

    def tokenize(self, req: str | Dict[str, Any]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            if ChatCompletionRequest.is_openai_request(req):
                chat_request = ChatCompletionRequest(**req)
                token_ids = self._openai_endpoint.render_chat(chat_request).input_ids
            else:
                prompt = req.pop("prompt")
                token_ids = self._frontend_worker.pipeline.encode(prompt)
            return ORJSONResponse({"token_ids": token_ids})
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def tokenizer_encode(self, req: Union[str, Dict[Any, Any]]):
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            prompt = req.pop("prompt")
            assert self._frontend_worker is not None
            if req.get("return_offsets_mapping", None) == True:
                mapping = self._frontend_worker.tokenizer_offset_mapping(prompt)
                response = TokenizerEncodeResponse(
                    offset_mapping=mapping["offset_mapping"],
                    token_ids=mapping["input_ids"],
                )
            else:
                token_ids, tokens = self._frontend_worker.tokenizer_encode(prompt)
                response = TokenizerEncodeResponse(token_ids=token_ids, tokens=tokens)
            return ORJSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            return ORJSONResponse(format_exception(e), status_code=500)

    def check_health(self):
        assert self._frontend_worker is not None
        return (
            self._frontend_worker.backend_rpc_server_visitor.is_backend_service_ready(
                refresh=False
            )
        )
