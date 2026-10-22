package org.flexlb.httpserver;

import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;

import java.util.concurrent.ThreadPoolExecutor;

/** Reject new calls at the RPC boundary, never by discarding gRPC lifecycle tasks. */
final class GrpcOverloadInterceptor implements ServerInterceptor {
    private final ThreadPoolExecutor executor;
    private final ThreadPoolExecutor controlExecutor;

    GrpcOverloadInterceptor(ThreadPoolExecutor executor) {
        this(executor, executor);
    }

    GrpcOverloadInterceptor(ThreadPoolExecutor executor, ThreadPoolExecutor controlExecutor) {
        this.executor = executor;
        this.controlExecutor = controlExecutor;
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers, ServerCallHandler<ReqT, RespT> next) {
        ThreadPoolExecutor executor =
                call.getMethodDescriptor().getFullMethodName().endsWith("/Cancel")
                        ? controlExecutor
                        : this.executor;
        if (executor.isShutdown() || executor.getQueue().remainingCapacity() == 0) {
            call.close(
                    Status.RESOURCE_EXHAUSTED.withDescription("FlexLB gRPC executor saturated"),
                    new Metadata());
            // Do not request/deserialize the payload or enter the business handler.
            // Transport cancellation still runs through gRPC's normal buffer cleanup.
            return new ServerCall.Listener<>() {};
        }
        return next.startCall(call, headers);
    }
}
