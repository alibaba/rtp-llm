package org.flexlb.httpserver;

import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;

import java.util.concurrent.ThreadPoolExecutor;

/**
 * Rejects new RPCs before payload delivery when the gRPC executor has no queue capacity.
 */
final class GrpcOverloadInterceptor implements ServerInterceptor {
    private final ThreadPoolExecutor executor;

    GrpcOverloadInterceptor(ThreadPoolExecutor executor) {
        this.executor = executor;
    }

    @Override
    public <RequestT, ResponseT> ServerCall.Listener<RequestT> interceptCall(
            ServerCall<RequestT, ResponseT> call,
            Metadata headers,
            ServerCallHandler<RequestT, ResponseT> next) {
        if (executor.isShutdown() || executor.getQueue().remainingCapacity() == 0) {
            call.close(
                    Status.RESOURCE_EXHAUSTED.withDescription("FlexLB gRPC executor saturated"),
                    new Metadata());
            return new ServerCall.Listener<>() {};
        }
        return next.startCall(call, headers);
    }
}
