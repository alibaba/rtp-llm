"""ViT RPC executors: lease reclamation must not queue behind embedding work."""

from concurrent import futures

import grpc

VIT_RPC_MAX_WORKERS = 1024
VIT_RELEASE_MAX_WORKERS = 16


class _ReleaseExecutorInterceptor(grpc.ServerInterceptor):
    def __init__(self, executor):
        self._executor = executor

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if (
            handler is None
            or handler_call_details.method != "/MultimodalRpcService/ReleaseRdmaLease"
        ):
            return handler

        def release(request, context):
            return handler.unary_unary(request, context)

        # grpcio selects this executor before enqueueing the unary handler. Moving
        # work to a second pool *inside* the handler would still queue behind ViT.
        release.experimental_thread_pool = self._executor
        return grpc.unary_unary_rpc_method_handler(
            release,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


def create_vit_grpc_server(max_workers=VIT_RPC_MAX_WORKERS):
    embedding_executor = futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="vit-rpc"
    )
    release_executor = futures.ThreadPoolExecutor(
        max_workers=VIT_RELEASE_MAX_WORKERS, thread_name_prefix="vit-release"
    )
    server = grpc.server(
        embedding_executor,
        interceptors=(_ReleaseExecutorInterceptor(release_executor),),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_concurrent_streams", -1),
            ("grpc.http2.min_ping_interval_without_data_ms", 1000),
            ("grpc.http2.max_ping_strikes", 1000),
        ],
    )
    server._vit_executors = (embedding_executor, release_executor)
    return server


def stop_vit_grpc_server(server):
    server.stop(grace=None).wait()
    for executor in server._vit_executors:
        executor.shutdown(wait=False, cancel_futures=True)
