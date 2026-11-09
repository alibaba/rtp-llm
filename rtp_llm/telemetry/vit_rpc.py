"""Request-only ViT RPC tracing; no media, cache keys or credentials are recorded.

The SDK remains optional and is initialized by the serving process, not here.
Status/cache polling and RDMA lease maintenance deliberately stay untraced.
"""

import asyncio
from collections import namedtuple
from contextlib import contextmanager

import grpc

from rtp_llm.telemetry import attributes as attrs
from rtp_llm.telemetry import tracing

_METHODS = {
    "GetMultimodalHashes": "get_multimodal_hashes",
    "AsyncSubmitEmbedding": "async_submit_embedding",
    "RemoteMultimodalEmbedding": "remote_multimodal_embedding",
    "WaitGreenNetVerdict": "wait_greennet_verdict",
}
_SERVICE = "/MultimodalRpcService/"


def _request_attributes(request):
    result = {}
    try:
        inputs = getattr(request, "inputs", request)
        if hasattr(request, "keys") and not request.HasField("inputs"):
            inputs = None  # A cache-only probe has no request ID (not request 0).
        if hasattr(inputs, "request_id"):
            result[attrs.REQUEST_ID] = str(inputs.request_id)
        if hasattr(inputs, "multimodal_inputs"):
            result[attrs.RTP_LLM_VIT_INPUT_COUNT] = len(inputs.multimodal_inputs)
        if hasattr(request, "keys"):
            result[attrs.RTP_LLM_VIT_KEY_COUNT] = len(request.keys)
            result[attrs.RTP_LLM_VIT_CACHE_PROBE] = not request.HasField("inputs")
    except Exception:
        pass
    return result


def _finish(span, code=grpc.StatusCode.OK, error=None):
    if span is None:
        return
    try:
        if not isinstance(code, grpc.StatusCode):
            code = grpc.StatusCode.UNKNOWN
        span.set_attribute(attrs.RPC_RESPONSE_STATUS_CODE, code.name)
        span.finish(
            error=error,
            error_type=code.name if code != grpc.StatusCode.OK else "",
        )
    except Exception:
        pass  # Observability must never affect an RPC or its completion callback.


def _error_code(error):
    try:
        if isinstance(error, (asyncio.CancelledError, grpc.FutureCancelledError)):
            return grpc.StatusCode.CANCELLED
        if isinstance(error, grpc.RpcError):
            return error.code()
    except Exception:
        pass
    return grpc.StatusCode.UNKNOWN


def _start_client(method, target, request, metadata):
    span, carrier = tracing.start_client_span("rtp_llm.vit." + _METHODS[method], target)
    if span is not None:
        for key, value in _request_attributes(request).items():
            span.set_attribute(key, value)
        span.set_attribute(attrs.RPC_METHOD, "MultimodalRpcService/" + method)
    # Only replace propagation keys when there is a new child. Preserve all
    # business metadata, including binary trailers/identity headers.
    if carrier:
        metadata = [
            (k, v)
            for k, v in (metadata or ())
            if k not in ("traceparent", "tracestate")
        ]
        metadata += carrier
    return span, metadata


@contextmanager
def vit_client_span(method, target, request, metadata=None):
    """Wrap one physical async RPC, including its await/cancellation boundary."""
    span, metadata = _start_client(method, target, request, metadata)
    try:
        yield metadata
    except BaseException as error:
        _finish(span, _error_code(error), error)
        raise
    else:
        _finish(span)


class VitServerTraceInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, details):
        handler = continuation(details)
        method = details.method.removeprefix(_SERVICE)
        if (
            not details.method.startswith(_SERVICE)
            or method not in _METHODS
            or handler is None
            or handler.unary_unary is None
        ):
            return handler

        def traced(request, context):
            if not tracing.is_telemetry_active():
                return handler.unary_unary(request, context)
            # A gRPC executor reuses threads. Never leave a previous request or
            # orchestration parent installed, even after abort/cancellation.
            state_token = tracing.CURRENT_TRACE_STATE.set(None)
            internal_token = tracing.CURRENT_INTERNAL_CONTEXT.set(None)
            state = None
            error = None
            try:
                attributes = _request_attributes(request)
                attributes[attrs.RPC_SYSTEM] = "grpc"
                attributes[attrs.RPC_METHOD] = "MultimodalRpcService/" + method
                state = tracing.start_server_span(
                    "rtp_llm.vit." + _METHODS[method],
                    tracing.metadata_to_headers(details.invocation_metadata),
                    initial_attributes=attributes,
                )
                return handler.unary_unary(request, context)
            except BaseException as caught:
                error = caught
                raise
            finally:
                try:
                    code = context.code()
                    if code is None:
                        if not context.is_active():
                            remaining = context.time_remaining()
                            code = (
                                grpc.StatusCode.DEADLINE_EXCEEDED
                                if remaining is not None and remaining <= 0
                                else grpc.StatusCode.CANCELLED
                            )
                        else:
                            code = (
                                grpc.StatusCode.UNKNOWN if error else grpc.StatusCode.OK
                            )
                    _finish(state, code, error)
                except Exception:
                    _finish(state, grpc.StatusCode.UNKNOWN, error)
                finally:
                    tracing.CURRENT_INTERNAL_CONTEXT.reset(internal_token)
                    tracing.CURRENT_TRACE_STATE.reset(state_token)

        return grpc.unary_unary_rpc_method_handler(
            traced,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


class _CallDetails(
    namedtuple(
        "_CallDetails", "method timeout metadata credentials wait_for_ready compression"
    ),
    grpc.ClientCallDetails,
):
    pass


class VitClientTraceInterceptor(grpc.UnaryUnaryClientInterceptor):
    """Proxy channels: finish on Future completion, not when .future() returns."""

    def __init__(self, target):
        self.target = target

    def intercept_unary_unary(self, continuation, details, request):
        method = details.method.removeprefix(_SERVICE)
        if not details.method.startswith(_SERVICE) or method not in _METHODS:
            return continuation(details, request)
        span, metadata = _start_client(method, self.target, request, details.metadata)
        traced_details = _CallDetails(
            details.method,
            details.timeout,
            metadata,
            details.credentials,
            details.wait_for_ready,
            details.compression,
        )
        try:
            call = continuation(traced_details, request)
        except BaseException as error:
            _finish(span, _error_code(error), error)
            raise
        if span is not None:

            def finished(completed):
                try:
                    _finish(span, completed.code())
                except Exception as error:
                    _finish(span, _error_code(error), error)

            try:
                call.add_done_callback(finished)
            except Exception as error:
                _finish(span, grpc.StatusCode.UNKNOWN, error)
        return call
