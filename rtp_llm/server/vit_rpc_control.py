"""Keep status polling, inspection waits and embedding execution independent."""

import threading
from concurrent import futures

import grpc

from rtp_llm.telemetry.vit_rpc import VitServerTraceInterceptor


class _VitExecutorInterceptor(grpc.ServerInterceptor):
    def __init__(self, status_executor, inspection_executor):
        self.executors = {
            "/MultimodalRpcService/GetWorkerStatus": status_executor,
            "/MultimodalRpcService/GetGreenNetVerdict": inspection_executor,
        }

    def intercept_service(self, continuation, details):
        handler = continuation(details)
        executor = self.executors.get(details.method)
        if handler is None or handler.unary_unary is None or executor is None:
            return handler

        def dispatch(request, context):
            return handler.unary_unary(request, context)

        # gRPC Python supports a per-handler executor; keep the same port and
        # wire contract. Cache directory scans are deliberately NOT put here.
        dispatch.experimental_thread_pool = executor
        return grpc.unary_unary_rpc_method_handler(
            dispatch,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def create_vit_rpc_server(max_workers=200):
    data_executor = futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="vit-rpc"
    )
    status_executor = futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="vit-status"
    )
    inspection_executor = futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="vit-inspection"
    )
    try:
        server = grpc.server(
            data_executor,
            interceptors=[
                _VitExecutorInterceptor(status_executor, inspection_executor),
                VitServerTraceInterceptor(),
            ],
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
                ("grpc.max_concurrent_streams", -1),
                ("grpc.http2.min_ping_interval_without_data_ms", 1000),
                ("grpc.http2.max_ping_strikes", 1000),
            ],
        )
    except BaseException:
        data_executor.shutdown(wait=False, cancel_futures=True)
        status_executor.shutdown(wait=False, cancel_futures=True)
        inspection_executor.shutdown(wait=False, cancel_futures=True)
        raise
    original_stop = server.stop

    def stop(grace):
        # gRPC owns the RPC drain. Executor shutdown must not cancel handlers
        # already accepted during a graceful stop.
        stopped = original_stop(grace)

        def close_executors():
            stopped.wait()
            data_executor.shutdown(wait=False)
            status_executor.shutdown(wait=False)
            inspection_executor.shutdown(wait=False)

        threading.Thread(
            target=close_executors, name="vit-rpc-shutdown", daemon=True
        ).start()
        return stopped

    server.stop = stop
    return server
