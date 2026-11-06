"""Opt-in DashSC gRPC transport for exact, precomputed prefill token IDs."""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional
from urllib.parse import urlsplit


class DashScPrefillClient:
    """Use gRPC for inference and the HTTP client for scheduler acknowledgement."""

    def __init__(
        self,
        http_client: Any,
        *,
        channel: Any = None,
        stub: Any = None,
        timeout_seconds: float = 300,
    ):
        self.http_client = http_client
        self._timeout_seconds = timeout_seconds
        self._closed = False
        parsed = urlsplit(http_client.endpoint)
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        try:
            from rtp_llm.config.py_config_modules import DASH_SC_GRPC_SERVER_PORT_OFFSET
            from rtp_llm.dash_sc.client import build_model_infer_request
            from rtp_llm.dash_sc.codec import SamplingParams

            self.endpoint = (
                f"{host}:{http_client.port + DASH_SC_GRPC_SERVER_PORT_OFFSET}"
            )
            self._build_request = build_model_infer_request
            self._sampling_type = SamplingParams
            if channel is None:
                import grpc

                from rtp_llm.dash_sc.client import dash_sc_grpc_client_channel_options

                channel = grpc.insecure_channel(
                    self.endpoint, options=dash_sc_grpc_client_channel_options()
                )
            self._channel = channel
            if stub is None:
                from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc

                stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            self._stub = stub
        except Exception as error:
            if channel is not None:
                channel.close()
            raise RuntimeError(
                f"DashSC input-ID transport dependencies unavailable: {error}"
            ) from error

    @property
    def port(self) -> int:
        return self.http_client.port

    def configure_scheduler(self, batch_size: int, mode: str = "prefill") -> None:
        self.http_client.configure_scheduler(batch_size, mode)

    def wait_ready(self, timeout_seconds: float) -> None:
        import grpc

        ready = grpc.channel_ready_future(self._channel)
        try:
            ready.result(timeout=timeout_seconds)
        except grpc.FutureTimeoutError as error:
            raise RuntimeError(
                f"DashSC gRPC endpoint {self.endpoint} unavailable after "
                f"{timeout_seconds} seconds"
            ) from error
        finally:
            ready.cancel()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._channel.close()

    def request(
        self,
        *,
        phase: str,
        case_id: str,
        request_index: int,
        run_index: Optional[int],
        prompt: str,
        input_ids: Any,
        output_length: int,
    ) -> Dict[str, Any]:
        del prompt
        request_id = f"{case_id}/{phase}/{request_index}/{run_index}"
        try:
            request = self._build_request(
                request_id=request_id,
                model_name="rtp_llm",
                input_ids=input_ids,
                sampling=self._sampling_type(
                    max_new_tokens=output_length, min_new_tokens=output_length, top_k=1
                ),
                enable_thinking=False,
                force_sp_accept=True,
            )
            values = {}
            output_len = 0
            seen_response = False
            for response in self._stub.ModelStreamInfer(
                iter((request,)),
                timeout=self._timeout_seconds,
                metadata=(("x-dashscope-inner-loggingconsent", "none"),),
            ):
                if response.error_message:
                    raise RuntimeError(response.error_message)
                if not response.HasField("infer_response"):
                    continue
                seen_response = True
                infer = response.infer_response
                parameters = infer.parameters
                error_no = (
                    parameters["error_no"].int64_param
                    if "error_no" in parameters
                    else 0
                )
                error_msg = (
                    parameters["error_msg"].string_param
                    if "error_msg" in parameters
                    else ""
                )
                status_code = (
                    parameters["status_code"].int64_param
                    if "status_code" in parameters
                    else 0
                )
                if (
                    error_no
                    or error_msg
                    or (status_code and not 200 <= status_code < 300)
                ):
                    raise RuntimeError(
                        f"DashSC business error: error_no={error_no}, "
                        f"status_code={status_code}, error_msg={error_msg}"
                    )
                if "engine_first_token_cost_time_us" in infer.parameters:
                    values["latency_ms"] = (
                        infer.parameters["engine_first_token_cost_time_us"].int64_param
                        / 1000.0
                    )
                for output, raw in zip(infer.outputs, infer.raw_output_contents):
                    if output.datatype != "INT32":
                        continue
                    if output.name == "generated_ids":
                        output_len = max(
                            output_len,
                            int(output.shape[-1]) if output.shape else len(raw) // 4,
                        )
                    elif output.name in ("prompt_token_num", "prompt_cached_token_num"):
                        values[output.name] = struct.unpack_from("<i", raw)[0]
            if not seen_response:
                raise RuntimeError("DashSC stream ended without an inference response")
            return {
                "success": True,
                "input_len": int(values.get("prompt_token_num", 0)),
                "output_len": output_len,
                "reuse_len": int(values.get("prompt_cached_token_num", 0)),
                "latency_ms": float(values.get("latency_ms", 0)),
            }
        except Exception as error:
            return {"success": False, "error": f"request {request_id} failed: {error}"}
