package org.flexlb.interceptor;

import io.grpc.Context;
import io.grpc.Contexts;
import io.grpc.ForwardingServerCall;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.propagation.TextMapGetter;
import org.flexlb.telemetry.FlexlbTrace;

import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Extracts the W3C context for a FlexLB Schedule call, reuses an upstream SERVER
 * span when one is already current, and otherwise creates and owns a manual one.
 */
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
            if (!FlexlbTrace.isEnabled()) {
                return Contexts.interceptCall(
                        grpcCurrent.withValue(OTEL_CONTEXT_KEY, extracted), call, headers, next);
            }
            Span existing = Span.fromContext(otelCurrent);
            if (existing.getSpanContext().isValid()) {
                // Something upstream already opened a span for this call -- an
                // auto-instrumentation agent, or another interceptor. Adopt it
                // instead of nesting a second SERVER span, and leave ending it
                // to whoever created it (ownsSpan stays false).
                serverSpan = existing;
            } else {
                // With tracing enabled but no provider configured, the no-op
                // SpanBuilder returns Span.wrap(parent's SpanContext), so this
                // still carries the extracted remote trace/span rather than an
                // invalid one -- storing it below preserves the traceparent for
                // the downstream forward instead of clobbering it.
                serverSpan = FlexlbTrace.startServer(spanName(call), extracted);
                ownsSpan = true;
            }
            io.opentelemetry.context.Context serverContext =
                    FlexlbTrace.withSpan(serverSpan, extracted);
            Context grpcContext = grpcCurrent.withValue(OTEL_CONTEXT_KEY, serverContext);
            // The handler's close() carries the call's real terminal status, which
            // the listener callbacks alone cannot recover: onComplete() fires for
            // an error close just as it does for a successful one.
            AtomicReference<Status> observedStatus = new AtomicReference<>();
            ServerCall.Listener<ReqT> listener = Contexts.interceptCall(
                    grpcContext, new StatusCapturingCall<>(call, observedStatus), headers, next);
            return new FinishingListener<>(listener, serverSpan, ownsSpan, observedStatus);
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

    /** Records the terminating status the handler passes to close(). */
    private static final class StatusCapturingCall<ReqT, RespT>
            extends ForwardingServerCall.SimpleForwardingServerCall<ReqT, RespT> {
        private final AtomicReference<Status> observedStatus;

        private StatusCapturingCall(ServerCall<ReqT, RespT> delegate,
                                   AtomicReference<Status> observedStatus) {
            super(delegate);
            this.observedStatus = observedStatus;
        }

        @Override
        public void close(Status status, Metadata trailers) {
            // Recorded before delegating, since the delegate drives the listener
            // callback that reads it. First close wins: gRPC rejects a second one.
            observedStatus.compareAndSet(null, status);
            super.close(status, trailers);
        }
    }

    private static final class FinishingListener<ReqT>
            extends ForwardingServerCallListener.SimpleForwardingServerCallListener<ReqT> {
        private final Span span;
        private final boolean ownsSpan;
        private final AtomicBoolean finished = new AtomicBoolean();
        private final AtomicReference<Status> observedStatus;

        private FinishingListener(ServerCall.Listener<ReqT> delegate,
                                 Span span,
                                 boolean ownsSpan,
                                 AtomicReference<Status> observedStatus) {
            super(delegate);
            this.span = span;
            this.ownsSpan = ownsSpan;
            this.observedStatus = observedStatus;
        }

        @Override
        public void onComplete() {
            settle(null);
            super.onComplete();
        }

        @Override
        public void onCancel() {
            // The synthetic CancellationException is only a fallback: when the
            // handler already closed the call, that status is the real cause.
            settle(new CancellationException("FlexLB Schedule RPC cancelled"));
            super.onCancel();
        }

        /**
         * Single terminal path shared by close, complete and cancel. Ends the span
         * exactly once, and only when this interceptor created it -- an
         * upstream-owned span is ended by its owner, so it only receives the
         * status attribute.
         */
        private void settle(Throwable fallbackError) {
            Status observed = observedStatus.get();
            if (!ownsSpan) {
                if (observed != null) {
                    FlexlbTrace.setAttribute(span, FlexlbTrace.RPC_RESPONSE_STATUS_CODE,
                            observed.getCode().name());
                }
                return;
            }
            if (!finished.compareAndSet(false, true)) {
                return;
            }
            if (observed != null) {
                FlexlbTrace.finishWithGrpcStatus(span, observed.getCode().name(),
                        observed.getCode().value(), observed.isOk());
            } else if (fallbackError instanceof CancellationException) {
                FlexlbTrace.finishWithGrpcStatus(span, "CANCELLED", Status.Code.CANCELLED.value(), false);
            } else {
                FlexlbTrace.finish(span, fallbackError);
            }
        }
    }
}
