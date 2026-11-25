package org.flexlb.httpserver;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;

import java.util.concurrent.Executor;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Rejects new RPCs before payload delivery when the gRPC executor has no queue capacity.
 */
final class GrpcOverloadInterceptor implements ServerInterceptor {
    private static final Context.Key<Executor> DIRECT_SCHEDULE_EXECUTOR =
            Context.key("flexlb-direct-schedule-executor");

    private final ThreadPoolExecutor executor;
    private final ThreadPoolExecutor controlExecutor;
    private final Executor directScheduleExecutor;

    GrpcOverloadInterceptor(ThreadPoolExecutor executor, ThreadPoolExecutor controlExecutor) {
        this.executor = executor;
        this.controlExecutor = controlExecutor;
        this.directScheduleExecutor = task -> executor.execute(new DirectScheduleTask(task));
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
        if (method != null
                && FlexlbServiceGrpc.getScheduleMethod()
                        .getFullMethodName()
                        .equals(method.getFullMethodName())) {
            Context context =
                    Context.current().withValue(DIRECT_SCHEDULE_EXECUTOR, directScheduleExecutor);
            return Contexts.interceptCall(context, call, headers, next);
        }
        return next.startCall(call, headers);
    }

    static Executor directScheduleExecutor() {
        return DIRECT_SCHEDULE_EXECUTOR.get();
    }

    /**
     * Marks new DIRECT placement work so executor overload can reject it before routing starts.
     */
    record DirectScheduleTask(Runnable route) implements Runnable {
        @Override
        public void run() {
            route.run();
        }
    }
}
