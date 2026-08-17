import asyncio
import json
import logging
import threading
import time
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
from rtp_llm.frontend.frontend_request_metrics import (
    FrontendRequestMetrics,
    FrontendRequestMetricState,
)
from rtp_llm.frontend.frontend_worker import FrontendWorker, TokenizerEncodeResponse
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import SpecialTokens, TaskType
from rtp_llm.server.misc import format_exception
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)
from rtp_llm.utils.concurrency_controller import (
    ConcurrencyException,
    get_global_controller,
)
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import check_with_info

USAGE_HEADER = "USAGE"


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
        monitor_interval = getattr(
            getattr(py_env_configs, "server_config", None),
            "monitor_interval",
            1,
        )
        self._request_metrics = FrontendRequestMetrics(
            concurrency_report_interval_s=max(float(monitor_interval or 1), 0.1)
        )

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

        self._request_metrics.start()

    async def close(self):
        try:
            if self._frontend_worker is not None:
                # Drain backend generators while the metric heartbeat remains
                # alive, then flush the final partial TPS window below.
                await self._frontend_worker.close()
        finally:
            await asyncio.to_thread(self._request_metrics.close)

    def stop(self):
        if self._frontend_worker is not None:
            self._frontend_worker.stop()

    @staticmethod
    def _request_aux_info_enabled(request: Dict[str, Any]) -> bool:
        def parse(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"false", "0", "off", "no"}:
                    return False
                if normalized in {"true", "1", "on", "yes"}:
                    return True
            return bool(value)

        if "aux_info" in request:
            return parse(request["aux_info"])
        config = request.get(
            "generation_config",
            request.get("generate_config", {}),
        )
        if isinstance(config, dict) and "aux_info" in config:
            return parse(config["aux_info"])
        return True

    @staticmethod
    def _copy_for_internal_aux_info(request: Dict[str, Any]) -> Dict[str, Any]:
        """Copy only the dictionaries mutated by the private metrics path."""
        copied = dict(request)
        config_name = (
            "generation_config" if "generation_config" in request else "generate_config"
        )
        config = request.get(config_name)
        if isinstance(config, dict):
            copied[config_name] = dict(config)
        FrontendServer._force_internal_aux_info(copied)
        return copied

    @staticmethod
    def _force_internal_aux_info(request: Dict[str, Any]) -> None:
        config_name = (
            "generation_config" if "generation_config" in request else "generate_config"
        )
        config = request.get(config_name)
        if isinstance(config, dict):
            config["aux_info"] = True
        request["aux_info"] = True

    @staticmethod
    def _hide_aux_info(response: Any) -> None:
        response_batch = (
            response.get("response_batch")
            if isinstance(response, dict)
            else getattr(response, "response_batch", None)
        )
        if response_batch:
            for item in response_batch:
                FrontendServer._hide_aux_info(item)
            return
        if isinstance(response, dict):
            response.pop("aux_info", None)
            return
        if not hasattr(response, "aux_info"):
            return
        aux_info = getattr(response, "aux_info")
        if isinstance(aux_info, dict):
            response.aux_info = {}
        elif isinstance(aux_info, (list, tuple)):
            response.aux_info = []
        else:
            response.aux_info = None

    @staticmethod
    def _client_exception_payload(
        request: Dict[str, Any], exception_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = dict(exception_json)
        # Keep the established error aux_info contract. Only strip internal
        # stack locations from the client-facing message; the access logger
        # receives the original exception payload before this projection.
        message = payload.get("message")
        if isinstance(message, str) and "\n Traceback:" in message:
            payload["message"] = message.split("\n Traceback:", 1)[0].rstrip()
        return payload

    @staticmethod
    def _request_disables_speculative(request: Dict[str, Any]) -> bool:
        if "force_disable_sp_run" in request:
            return bool(request["force_disable_sp_run"])
        config = request.get(
            "generation_config",
            request.get("generate_config", {}),
        )
        if isinstance(config, dict) and "force_disable_sp_run" in config:
            return bool(config["force_disable_sp_run"])
        return False

    async def embedding(self, request: Dict[str, Any], raw_request: Request):
        start_time = time.time()
        try:
            if isinstance(request, str):
                request = json.loads(request)
            kmonitor.report(
                AccMetrics.QPS_METRIC, 1, {"source": request.get("source", "unknown")}
            )
            sequence = self._global_controller.increment() % 4096  # 12 bits
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
            return ORJSONResponse(result, headers={USAGE_HEADER: json.dumps(usage)})
        except BaseException as e:
            return self._handle_exception(request, e)
        finally:
            self._global_controller.decrement()

    # use asyncio.sleep(0) to correctly exit when client closed https://github.com/tiangolo/fastapi/issues/4146
    async def stream_response(
        self,
        request: Dict[str, Any],
        response: CompleteResponseAsyncGenerator,
    ):
        is_openai_response = request.get("stream", False)
        response_data_prefix = "data: " if is_openai_response else "data:"
        try:
            async for res in response:
                data_str = res.model_dump_json(exclude_none=True)
                yield response_data_prefix + data_str + "\r\n\r\n"
                await asyncio.sleep(0)
            if not is_openai_response:
                yield f"data:[done]\r\n\r\n"
            await self._collect_complete_response_and_record_access_log(
                request, response
            )
        except asyncio.CancelledError as e:
            try:
                await response.aclose()
            except Exception as close_error:
                logging.warning(
                    "close streaming response after cancellation failed: %s",
                    close_error,
                )
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
            raise
        except BaseException as e:
            # 捕获非Cancel以外所有的异常,所以使用BaseException
            format_e = format_exception(e)
            self._access_logger.log_exception_access(request, e, format_e)
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
            client_error = self._client_exception_payload(request, format_e)
            yield response_data_prefix + json.dumps(
                client_error, ensure_ascii=False
            ) + "\r\n\r\n"
        finally:
            self._global_controller.decrement()

    async def inference(self, req: Union[str, Dict[Any, Any]], raw_request: RawRequest):
        request_headers: Dict[str, str] = {}
        try:
            if isinstance(req, str):
                req = json.loads(req)
            assert isinstance(req, dict)
            sequence = self._global_controller.increment() % 4096  # 12 bits
            req[request_id_field_name] = generate_request_id(
                self.py_env_configs.server_config.ip,
                self.py_env_configs.server_config.server_port,
                self.server_id,
                sequence,
            )
            request_headers = extract_request_headers(
                getattr(raw_request, "headers", None)
            )
            generation_req = (
                self._copy_for_internal_aux_info(req)
                if self._request_metrics.enabled
                else req
            )
        except Exception as e:
            return self._handle_exception(req, e)

        def generate_call(_request_metrics: FrontendRequestMetricState):
            assert self._frontend_worker is not None
            metric_tags = (
                {
                    "rank_id": self.rank_id,
                    "server_id": self.server_id,
                    "source": str(req.get("source", "unknown")),
                }
                if self._request_metrics.enabled
                else {}
            )
            metric_observer = (
                _request_metrics.observe_tps if self._request_metrics.enabled else None
            )
            if request_headers:
                return self._frontend_worker.inference(
                    **generation_req,
                    headers=request_headers,
                    frontend_metric_tags=metric_tags,
                    frontend_metric_observer=metric_observer,
                )
            return self._frontend_worker.inference(
                **generation_req,
                frontend_metric_tags=metric_tags,
                frontend_metric_observer=metric_observer,
            )

        try:
            rep = await self._infer_wrap(req, raw_request, generate_call)
        except BaseException as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def _infer_wrap(
        self,
        req: Dict[str, Any],
        raw_request: RawRequest,
        generate_call: Callable[
            [FrontendRequestMetricState], CompleteResponseAsyncGenerator
        ],
    ):
        try:
            rep = await self._infer_impl(req, raw_request, generate_call)
        except BaseException as e:
            rep = self._handle_exception(req, e)
        return rep

    async def chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        sequence = self._global_controller.increment() % 4096  # 12 bits
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )

        if request.prompt_logprobs is not None:
            request.stream = False
        elif request.extra_configs is not None and getattr(
            request.extra_configs, "return_prompt_logits", False
        ):
            request.stream = False

        # Copy only after applying the public forced-nonstream contract so the
        # backend, renderer, and frontend response path agree on streaming.
        internal_request = (
            request.model_copy(update={"aux_info": True})
            if self._request_metrics.enabled
            else request
        )

        def generate_call(request_metrics: FrontendRequestMetricState):
            assert self._openai_endpoint != None
            response = self._openai_endpoint.chat_completion(
                request_id,
                internal_request,
                raw_request,
                frontend_metric_tags=(
                    {
                        "rank_id": self.rank_id,
                        "server_id": self.server_id,
                        "source": str(getattr(request, "source", "unknown")),
                    }
                    if self._request_metrics.enabled
                    else {}
                ),
                frontend_metric_observer=(
                    request_metrics.observe_tps
                    if self._request_metrics.enabled
                    else None
                ),
            )
            assert isinstance(
                response, CompleteResponseAsyncGenerator
            ), f"error type: {type(response)}"
            return response

        try:
            request_dict = request.model_dump(exclude_none=True)
            # Preserve the client's explicit Optional[bool] intent after the
            # backend-only copy above forces AuxInfo collection on.
            request_dict["aux_info"] = request.aux_info
            request_dict[request_id_field_name] = request_id
            rep = await self._infer_wrap(request_dict, raw_request, generate_call)
        except BaseException as e:
            self._global_controller.decrement()
            raise e

        if not isinstance(rep, StreamingResponse):
            self._global_controller.decrement()

        return rep

    async def batch_chat_completion(self, request, raw_request: Request):
        from rtp_llm.openai.api_datatype import BatchChatCompletionResponse

        sequence = self._global_controller.increment() % 4096
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        try:
            assert self._openai_endpoint is not None
            responses = await self._openai_endpoint.batch_chat_completion(
                request_id, request
            )
            return ORJSONResponse(
                content=BatchChatCompletionResponse(
                    responses=[r.model_dump(exclude_none=True) for r in responses]
                ).model_dump()
            )
        finally:
            self._global_controller.decrement()

    async def batch_infer(self, req: dict, raw_request: Request):
        from rtp_llm.frontend.frontend_worker import BatchPipelineResponse

        # Concurrency accounting: a batch counts as ONE scheduling unit because the engine
        # atomically enqueues all prompts via BatchGenerateCall. Per-item counting would over-
        # reject under the same concurrency_limit; the trade-off is that a large batch occupies
        # only one slot regardless of N.
        sequence = self._global_controller.increment() % 4096
        request_id = generate_request_id(
            self.py_env_configs.server_config.ip,
            self.py_env_configs.server_config.server_port,
            self.server_id,
            sequence,
        )
        try:
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
            self._global_controller.decrement()

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
            self._access_logger.log_exception_access(request, e, exception_json)

        rep = ORJSONResponse(
            self._client_exception_payload(request, exception_json), status_code=500
        )
        return rep

    async def _call_generate_with_report(
        self,
        generate_call: Callable[
            [FrontendRequestMetricState], CompleteResponseAsyncGenerator
        ],
        request_metrics: FrontendRequestMetricState,
        expose_aux_info: bool,
    ):
        async def __gen_response_with_report(start_time: float, response_generator):
            last_iterate_time = current_time_ms()
            first_token = True
            iter_count = 0
            try:
                async for response in response_generator:
                    end_time = current_time_ms()
                    request_metrics.observe(response)
                    if first_token:
                        first_token = False
                        kmonitor.report(
                            GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC,
                            end_time - last_iterate_time,
                        )
                    else:
                        step_output_len = 1
                        # Internal metrics may force AuxInfo on, but the legacy
                        # iteration metric must retain the client's original
                        # aux_info=false behavior (one outward frame = one step).
                        if expose_aux_info and hasattr(response, "aux_info"):
                            if isinstance(response.aux_info, list):
                                step_output_len = 0
                                for info in response.aux_info:
                                    step_output_len += info.get("step_output_len", 1)
                                step_output_len = max(step_output_len, 1)
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
                    if not expose_aux_info:
                        self._hide_aux_info(response)
                    yield response
                kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, iter_count)
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
                request_metrics.finish()

        assert self._frontend_worker is not None
        start_time = current_time_ms()
        response_generator = generate_call(request_metrics)
        return CompleteResponseAsyncGenerator(
            __gen_response_with_report(start_time, response_generator),
            response_generator._collect_complete_response_func,
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

    async def _infer_impl(
        self,
        req: Dict[Any, Any],
        raw_request: RawRequest,
        generate_call: Callable[
            [FrontendRequestMetricState], CompleteResponseAsyncGenerator
        ],
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
        request_metrics = self._request_metrics.begin(
            rank_id=self.rank_id,
            server_id=self.server_id,
            source=str(req.get("source", "unkown")),
            streaming=is_streaming,
        )
        try:
            if await raw_request.is_disconnected():
                raise asyncio.CancelledError("client disconnects")
            res = await self._call_generate_with_report(
                generate_call,
                request_metrics,
                self._request_aux_info_enabled(req),
            )
        except BaseException:
            request_metrics.finish()
            raise

        if is_streaming:
            return StreamingResponse(
                self.stream_response(req, res), media_type="text/event-stream"
            )
        async for x in res:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await res.aclose()
                raise asyncio.CancelledError("client disconnects")

        complete_response = await self._collect_complete_response_and_record_access_log(
            req, res
        )
        return ORJSONResponse(content=complete_response)

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
