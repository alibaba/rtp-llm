package org.flexlb.interceptor;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.propagation.TextMapGetter;
import org.flexlb.telemetry.FlexlbTrace;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

/** Extracts W3C context and owns the manual FlexLB Schedule SERVER span. */
public final class GrpcTraceInterceptor implements ServerInterceptor {

    private static final TextMapGetter<Metadata> METADATA_GETTER = new TextMapGetter<>() {
        @Override
        public Iterable<String> keys(Metadata carrier) {
            return List.of("traceparent", "tracestate");
        }

        @Override
        public String get(Metadata carrier, String key) {
            if (carrier == null || key == null) {
                return null;
            }
            try {
                return carrier.get(Metadata.Key.of(key, Metadata.ASCII_STRING_MARSHALLER));
            } catch (RuntimeException ignored) {
                return null;
            }
        }
    };

    public static final io.grpc.Context.Key<io.opentelemetry.context.Context> OTEL_CONTEXT_KEY =
            io.grpc.Context.key("flexlbOtelContext");

    public static io.opentelemetry.context.Context getOtelContext() {
        return OTEL_CONTEXT_KEY.get();
    }

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(
            ServerCall<ReqT, RespT> call,
            Metadata headers,
            ServerCallHandler<ReqT, RespT> next) {
        Context grpcCurrent = Context.current();
        io.opentelemetry.context.Context otelCurrent =
                io.opentelemetry.context.Context.current();
        io.opentelemetry.context.Context extracted = otelCurrent;
        Span serverSpan = Span.getInvalid();
        boolean ownsSpan = false;
        try {
            extracted = FlexlbTrace.extract(otelCurrent, headers, METADATA_GETTER);
            Span existing = Span.fromContext(otelCurrent);
            if (existing.getSpanContext().isValid()) {
                // An installed gRPC agent already owns the real SERVER span.
                serverSpan = existing;
            } else {
                serverSpan = FlexlbTrace.startServer(spanName(call), extracted);
                ownsSpan = true;
            }
            io.opentelemetry.context.Context serverContext =
                    FlexlbTrace.withSpan(serverSpan, extracted);
            Context grpcContext = grpcCurrent.withValue(OTEL_CONTEXT_KEY, serverContext);
            ServerCall.Listener<ReqT> listener = Contexts.interceptCall(
                    grpcContext, call, headers, next);
            return new FinishingListener<>(listener, serverSpan, ownsSpan);
        } catch (RuntimeException | Error error) {
            if (ownsSpan) {
                FlexlbTrace.finish(serverSpan, error);
            }
            throw error;
        }
    }

    private static String spanName(ServerCall<?, ?> call) {
        String fullMethodName = call.getMethodDescriptor().getFullMethodName();
        int separator = fullMethodName.lastIndexOf('/');
        String methodName = separator >= 0
                ? fullMethodName.substring(separator + 1)
                : fullMethodName;
        StringBuilder snakeName = new StringBuilder(methodName.length() + 8);
        for (int i = 0; i < methodName.length(); ++i) {
            char character = methodName.charAt(i);
            if (Character.isUpperCase(character) && i > 0) {
                snakeName.append('_');
            }
            snakeName.append(Character.toLowerCase(character));
        }
        return "rtp_llm.flexlb." + snakeName;
    }

    private static final class FinishingListener<ReqT>
            extends ForwardingServerCallListener.SimpleForwardingServerCallListener<ReqT> {
        private final Span span;
        private final boolean ownsSpan;
        private final AtomicBoolean finished = new AtomicBoolean();

        private FinishingListener(ServerCall.Listener<ReqT> delegate, Span span, boolean ownsSpan) {
            super(delegate);
            this.span = span;
            this.ownsSpan = ownsSpan;
        }

        @Override
        public void onComplete() {
            finish(null);
            super.onComplete();
        }

        @Override
        public void onCancel() {
            finish(new CancellationException("FlexLB Schedule RPC cancelled"));
            super.onCancel();
        }

        private void finish(Throwable error) {
            if (ownsSpan && finished.compareAndSet(false, true)) {
                FlexlbTrace.finish(span, error);
            }
        }
    }
}
