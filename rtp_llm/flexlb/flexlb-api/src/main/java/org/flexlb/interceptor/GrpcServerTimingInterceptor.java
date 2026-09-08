package org.flexlb.interceptor;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import org.springframework.stereotype.Component;

/**
 * gRPC server interceptor that records the timestamp when a request enters
 * the gRPC server pipeline (before being dispatched to the service implementation).
 *
 * <p>The {@code grpcEntryTime} is propagated via gRPC {@link Context} so that
 * the service implementation can split the total arrival delay into:
 * <ul>
 *   <li>{@code app.request.network.delay.ms} = grpcEntryTime - requestTimeMs (network transfer)</li>
 *   <li>{@code app.grpc.server.process.ms} = startTime - grpcEntryTime (server-side processing)</li>
 * </ul>
 */
@Component
public class GrpcServerTimingInterceptor implements ServerInterceptor {

    /**
     * Context key carrying the gRPC server entry timestamp (epoch millis).
     * Accessed by {@code FlexlbServiceImpl} via {@link #get()}.
     */
    public static final Context.Key<Long> GRPC_ENTRY_TIME_KEY = Context.key("grpcEntryTime");
    public static final Context.Key<Long> GRPC_ENTRY_NANOS_KEY = Context.key("grpcEntryNanos");

    /**
     * Convenience method to retrieve the current gRPC entry time from the
     * active context. Returns {@code null} if the interceptor did not set it
     * (e.g. when the call bypassed the interceptor).
     */
    public static Long get() {
        return GRPC_ENTRY_TIME_KEY.get();
    }

    public static Long getNanos() {
        return GRPC_ENTRY_NANOS_KEY.get();
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers,
            ServerCallHandler<ReqT, RespT> next) {
        long grpcEntryTime = System.currentTimeMillis();
        long grpcEntryNanos = System.nanoTime();
        Context ctx = Context.current()
                .withValue(GRPC_ENTRY_TIME_KEY, grpcEntryTime)
                .withValue(GRPC_ENTRY_NANOS_KEY, grpcEntryNanos);
        return Contexts.interceptCall(ctx, call, headers, next);
    }
}
