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
    private final ThreadPoolExecutor controlExecutor;

    GrpcOverloadInterceptor(ThreadPoolExecutor executor, ThreadPoolExecutor controlExecutor) {
        this.executor = executor;
        this.controlExecutor = controlExecutor;
    }

    @Override
    public <RequestT, ResponseT> ServerCall.Listener<RequestT> interceptCall(
            ServerCall<RequestT, ResponseT> call,
            Metadata headers,
            ServerCallHandler<RequestT, ResponseT> next) {
        var method = call.getMethodDescriptor();
        var callExecutor =
                method != null && GrpcReceivingGuard.isControlRpc(method.getFullMethodName())
                        ? controlExecutor
                        : executor;
        if (callExecutor.isShutdown() || callExecutor.getQueue().remainingCapacity() == 0) {
            call.close(
                    Status.RESOURCE_EXHAUSTED.withDescription("FlexLB gRPC executor saturated"),
                    new Metadata());
            return new ServerCall.Listener<>() {};
        }
        return next.startCall(call, headers);
    }
}
