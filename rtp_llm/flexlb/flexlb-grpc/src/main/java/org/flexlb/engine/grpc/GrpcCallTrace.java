package org.flexlb.engine.grpc;

import io.grpc.Attributes;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ClientStreamTracer;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;

import java.util.function.LongSupplier;

/** Records transport progress without inspecting messages or changing the RPC outcome. */
public final class GrpcCallTrace extends ClientStreamTracer.Factory {
    public static final CallOptions.Key<GrpcCallTrace> CALL_TRACE =
            CallOptions.Key.create("flexlb.grpc.call-trace");
    public static final ClientInterceptor INTERCEPTOR =
            new ClientInterceptor() {
                @Override
                public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
                        MethodDescriptor<ReqT, RespT> method, CallOptions options, Channel next) {
                    CallOptions tracedOptions = options;
                    try {
                        GrpcCallTrace trace = options.getOption(CALL_TRACE);
                        if (trace != null) {
                            tracedOptions = options.withStreamTracerFactory(trace);
                        }
                    } catch (RuntimeException | Error ignored) {
                        // Tracing setup must not prevent the call from starting.
                    }
                    return next.newCall(method, tracedOptions);
                }
            };

    private static final long UNSET = Long.MIN_VALUE;
    private static final ClientStreamTracer NOOP = new ClientStreamTracer() {};

    private final GrpcTransportObserver observer;
    private final LongSupplier clock;
    private final long startedNanos;
    private volatile TraceStream latestStream;
    private volatile long callbackStartedNanos = UNSET;

    public GrpcCallTrace(GrpcTransportObserver observer) {
        this(observer, System::nanoTime);
    }

    GrpcCallTrace(GrpcTransportObserver observer, LongSupplier clock) {
        this.observer = observer;
        this.clock = clock;
        this.startedNanos = nanoTime();
    }

    public enum Stage {
        WAITING_TRANSPORT,
        WAITING_HEADERS_WRITE,
        WAITING_RESPONSE,
        RECEIVING_RESPONSE,
        RESPONSE_TRAILERS_RECEIVED
    }

    public record Snapshot(
            Stage stage,
            String transportEvent,
            GrpcFailure.Kind transportCauseKind,
            double callbackDelayMs,
            double responseDeliveryMs,
            double streamCreatedMs,
            double outboundHeadersMs,
            double inboundHeadersMs,
            double inboundTrailersMs,
            double streamClosedMs,
            double callbackStartedMs,
            double transportEventMs) {
        public static final Snapshot EMPTY =
                new Snapshot(
                        Stage.WAITING_TRANSPORT, "NONE", null, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        public String failureOrigin(GrpcFailure.Analysis failure) {
            return (failure != null && failure.confirmedLocalFailure())
                            || transportCauseKind == GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED
                            || transportCauseKind == GrpcFailure.Kind.LOCAL_EXECUTOR_REJECTED
                            || transportCauseKind == GrpcFailure.Kind.LOCAL_BIND_FAILED
                    ? "MASTER"
                    : "UNCONFIRMED";
        }
    }

    @Override
    public ClientStreamTracer newClientStreamTracer(
            ClientStreamTracer.StreamInfo info, Metadata headers) {
        try {
            if (callbackStartedNanos != UNSET) {
                return NOOP;
            }
            latestStream = null;
            TraceStream stream = new TraceStream();
            latestStream = stream;
            return stream;
        } catch (RuntimeException | Error ignored) {
            return NOOP;
        }
    }

    public void markCallbackStarted() {
        if (callbackStartedNanos == UNSET) {
            callbackStartedNanos = nanoTime();
        }
    }

    public Snapshot snapshot() {
        try {
            TraceStream stream = latestStream;
            long callback = callbackStartedNanos;
            if (stream == null) {
                return new Snapshot(
                        Stage.WAITING_TRANSPORT,
                        "NONE",
                        null,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        -1,
                        elapsedMs(startedNanos, callback),
                        -1);
            }
            long closed = beforeEnd(stream.closedNanos, callback);
            long end = closed == UNSET ? callback : closed;
            long created = beforeEnd(stream.createdNanos, end);
            long outbound = beforeEnd(stream.outboundHeadersNanos, end);
            long headers = beforeEnd(stream.inboundHeadersNanos, end);
            long trailers = beforeEnd(stream.inboundTrailersNanos, end);
            Stage stage =
                    trailers != UNSET
                            ? Stage.RESPONSE_TRAILERS_RECEIVED
                            : headers != UNSET
                                    ? Stage.RECEIVING_RESPONSE
                                    : outbound != UNSET
                                            ? Stage.WAITING_RESPONSE
                                            : created != UNSET
                                                    ? Stage.WAITING_HEADERS_WRITE
                                                    : Stage.WAITING_TRANSPORT;

            GrpcTransportObserver.SocketState socket = stream.socket;
            GrpcTransportObserver.Observation event = socket == null ? null : socket.observation();
            long socketEnd = trailers == UNSET ? end : trailers;
            if (event != null
                    && (created == UNSET
                            || event.nanoTime() - created < 0
                            || beforeEnd(event.nanoTime(), socketEnd) == UNSET)) {
                event = null;
            }
            return new Snapshot(
                    stage,
                    event == null ? "NONE" : event.event().name(),
                    event == null ? null : event.causeKind(),
                    elapsedMs(closed, callback),
                    elapsedMs(trailers, callback),
                    elapsedMs(startedNanos, created),
                    elapsedMs(startedNanos, outbound),
                    elapsedMs(startedNanos, headers),
                    elapsedMs(startedNanos, trailers),
                    elapsedMs(startedNanos, closed),
                    elapsedMs(startedNanos, callback),
                    event == null ? -1 : elapsedMs(startedNanos, event.nanoTime()));
        } catch (RuntimeException | Error ignored) {
            return Snapshot.EMPTY;
        }
    }

    private long nanoTime() {
        try {
            return clock.getAsLong();
        } catch (RuntimeException | Error ignored) {
            return UNSET;
        }
    }

    private static long beforeEnd(long time, long end) {
        return time != UNSET && (end == UNSET || time - end <= 0) ? time : UNSET;
    }

    private static double elapsedMs(long start, long end) {
        return start == UNSET || end == UNSET || end - start < 0 ? -1 : (end - start) / 1_000_000.0;
    }

    private final class TraceStream extends ClientStreamTracer {
        private volatile long createdNanos = UNSET;
        private volatile long outboundHeadersNanos = UNSET;
        private volatile long inboundHeadersNanos = UNSET;
        private volatile long inboundTrailersNanos = UNSET;
        private volatile long closedNanos = UNSET;
        private volatile GrpcTransportObserver.SocketState socket;

        @Override
        public void streamCreated(Attributes attributes, Metadata headers) {
            try {
                if (createdNanos == UNSET) {
                    createdNanos = nanoTime();
                    socket = observer == null ? null : observer.find(attributes);
                }
            } catch (RuntimeException | Error ignored) {
                // Missing diagnostics must not prevent stream creation.
            }
        }

        @Override
        public void outboundHeaders() {
            if (outboundHeadersNanos == UNSET) {
                outboundHeadersNanos = nanoTime();
            }
        }

        @Override
        public void inboundHeaders(Metadata headers) {
            if (inboundHeadersNanos == UNSET) {
                inboundHeadersNanos = nanoTime();
            }
        }

        @Override
        public void inboundTrailers(Metadata trailers) {
            if (inboundTrailersNanos == UNSET) {
                inboundTrailersNanos = nanoTime();
            }
        }

        @Override
        public void streamClosed(Status status) {
            if (closedNanos == UNSET) {
                closedNanos = nanoTime();
            }
        }
    }
}
