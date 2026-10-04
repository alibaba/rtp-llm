package org.flexlb.interceptor;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerStreamTracer;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicLong;

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

    private static final Context.Key<RequestMessageSize> REQUEST_MESSAGE_SIZE_KEY =
            Context.key("requestMessageSize");

    private final AtomicLong lastScheduleArrivalNanos = new AtomicLong(System.nanoTime());

    public long getLastScheduleArrivalNanos() {
        return lastScheduleArrivalNanos.get();
    }

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

    /**
     * Returns the received unary request's uncompressed protobuf bytes, without
     * the gRPC frame header. A call that bypasses the transport has no sample.
     */
    public static Long getRequestMessageBytes() {
        RequestMessageSize size = REQUEST_MESSAGE_SIZE_KEY.get();
        return size == null || !size.messageReceived ? null : size.bytes.get();
    }

    /**
     * Counts bytes already read by gRPC, including decompression. Reading the
     * protobuf in the service does not need another traversal for monitoring.
     */
    public ServerStreamTracer.Factory requestSizeTracerFactory() {
        return new ServerStreamTracer.Factory() {
            @Override
            public ServerStreamTracer newServerStreamTracer(String fullMethodName, Metadata headers) {
                if (!FlexlbServiceGrpc.getScheduleMethod().getFullMethodName().equals(fullMethodName)) {
                    return new ServerStreamTracer() { };
                }
                RequestMessageSize size = new RequestMessageSize();
                return new ServerStreamTracer() {
                    @Override
                    public Context filterContext(Context context) {
                        return context.withValue(REQUEST_MESSAGE_SIZE_KEY, size);
                    }

                    @Override
                    public void inboundMessageRead(int seqNo, long wireBytes, long uncompressedBytes) {
                        size.messageReceived = true;
                    }

                    @Override
                    public void inboundUncompressedSize(long bytes) {
                        size.bytes.addAndGet(bytes);
                    }
                };
            }
        };
    }

    private static class RequestMessageSize {
        private final AtomicLong bytes = new AtomicLong();
        private volatile boolean messageReceived;
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call, Metadata headers,
            ServerCallHandler<ReqT, RespT> next) {
        if (call.getMethodDescriptor().getFullMethodName()
                .equals(FlexlbServiceGrpc.getScheduleMethod().getFullMethodName())) {
            lastScheduleArrivalNanos.updateAndGet(previous -> System.nanoTime());
        }
        long grpcEntryTime = System.currentTimeMillis();
        long grpcEntryNanos = System.nanoTime();
        Context ctx = Context.current()
                .withValue(GRPC_ENTRY_TIME_KEY, grpcEntryTime)
                .withValue(GRPC_ENTRY_NANOS_KEY, grpcEntryNanos);
        return Contexts.interceptCall(ctx, call, headers, next);
    }
}
