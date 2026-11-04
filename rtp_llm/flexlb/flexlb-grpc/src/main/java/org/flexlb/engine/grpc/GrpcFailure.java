package org.flexlb.engine.grpc;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;
import io.netty.channel.ConnectTimeoutException;
import io.netty.handler.codec.http2.Http2Exception;

import java.io.EOFException;
import java.io.IOException;
import java.net.BindException;
import java.net.ConnectException;
import java.net.NoRouteToHostException;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.ClosedChannelException;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.RejectedExecutionException;

/** Classifies actual causes for diagnostics and retry. RPC failure alone does not establish worker health. */
public final class GrpcFailure {
    // Stronger local evidence wins over transport wrappers.
    // This order is independent of both the cause-chain order and Kind's declaration order.
    private static final List<Kind> CAUSE_PRECEDENCE =
            List.of(
                    Kind.LOCAL_MEMORY_EXHAUSTED,
                    Kind.LOCAL_EXECUTOR_REJECTED,
                    Kind.LOCAL_BIND_FAILED,
                    Kind.PROTOBUF_FAILURE,
                    Kind.CONNECT_TIMEOUT,
                    Kind.CONNECT_REFUSED,
                    Kind.NO_ROUTE,
                    Kind.CONNECTION_RESET,
                    Kind.BROKEN_PIPE,
                    Kind.IO_INTERRUPTED,
                    Kind.ASYNC_CHANNEL_CLOSED,
                    Kind.CONNECT_FAILED,
                    Kind.UNEXPECTED_EOF,
                    Kind.IO_TIMEOUT,
                    Kind.SOCKET_FAILURE,
                    Kind.CHANNEL_CLOSED,
                    Kind.HTTP2_FAILURE,
                    Kind.IO_FAILURE);

    private GrpcFailure() {}

    public enum Kind {
        LOCAL_MEMORY_EXHAUSTED,
        LOCAL_EXECUTOR_REJECTED,
        LOCAL_BIND_FAILED,
        CONNECT_TIMEOUT,
        CONNECT_REFUSED,
        CONNECT_FAILED,
        NO_ROUTE,
        CONNECTION_RESET,
        BROKEN_PIPE,
        IO_INTERRUPTED,
        ASYNC_CHANNEL_CLOSED,
        CHANNEL_CLOSED,
        UNEXPECTED_EOF,
        IO_TIMEOUT,
        SOCKET_FAILURE,
        PROTOBUF_FAILURE,
        IO_FAILURE,
        HTTP2_FAILURE,
        DEADLINE_EXCEEDED,
        CANCELLED,
        RPC_FAILURE
    }

    public record Analysis(Kind kind, Status.Code grpcCode, Throwable rootCause) {
        public boolean confirmedLocalFailure() {
            return kind == Kind.LOCAL_MEMORY_EXHAUSTED
                    || kind == Kind.LOCAL_EXECUTOR_REJECTED
                    || kind == Kind.LOCAL_BIND_FAILED;
        }

        /** Caller also enforces operation idempotency and the single retry limit. */
        public boolean retryableTransportFailure() {
            // Cancellation/deadline remains terminal even when cancellation exposes a closed
            // transport.
            if (grpcCode == Status.Code.CANCELLED || grpcCode == Status.Code.DEADLINE_EXCEEDED) {
                return false;
            }
            return switch (kind) {
                case CONNECT_TIMEOUT,
                        CONNECT_REFUSED,
                        CONNECT_FAILED,
                        NO_ROUTE,
                        CONNECTION_RESET,
                        BROKEN_PIPE,
                        IO_INTERRUPTED,
                        ASYNC_CHANNEL_CLOSED,
                        CHANNEL_CLOSED,
                        UNEXPECTED_EOF,
                        IO_TIMEOUT,
                        SOCKET_FAILURE,
                        HTTP2_FAILURE ->
                        true;
                default -> false;
            };
        }
    }

    public static Kind classify(Throwable failure) {
        return analyze(failure).kind();
    }

    public static Analysis analyze(Throwable failure) {
        if (failure == null) {
            return new Analysis(Kind.RPC_FAILURE, Status.Code.UNKNOWN, null);
        }
        if (failure.getCause() == null) {
            Status status = grpcStatus(failure);
            Status.Code code = status == null ? Status.Code.UNKNOWN : status.getCode();
            Kind kind = classifyCause(failure);
            return new Analysis(kind == null ? classifyStatusCode(code) : kind, code, failure);
        }

        Status.Code code = Status.Code.UNKNOWN;
        boolean statusFound = false;
        Throwable root = failure;
        EnumSet<Kind> observedKinds = EnumSet.noneOf(Kind.class);
        IdentityHashMap<Throwable, Boolean> visited = new IdentityHashMap<>();
        // Suppressed errors are earlier attempts/diagnostic failures, not this attempt's cause.
        for (Throwable cause = failure;
                cause != null && visited.put(cause, Boolean.TRUE) == null;
                cause = cause.getCause()) {
            root = cause;
            // The first gRPC exception owns this attempt's code, including UNKNOWN.
            if (!statusFound) {
                Status status = grpcStatus(cause);
                if (status != null) {
                    code = status.getCode();
                    statusFound = true;
                }
            }
            Kind kind = classifyCause(cause);
            if (kind != null) {
                observedKinds.add(kind);
            }
        }
        return new Analysis(resolveKind(observedKinds, code), code, root);
    }

    private static Kind classifyCause(Throwable cause) {
        // Subclasses precede their broader exception types.
        return switch (cause) {
            case OutOfMemoryError ignored -> Kind.LOCAL_MEMORY_EXHAUSTED;
            case RejectedExecutionException ignored -> Kind.LOCAL_EXECUTOR_REJECTED;
            case BindException ignored -> Kind.LOCAL_BIND_FAILED;
            case InvalidProtocolBufferException ignored -> Kind.PROTOBUF_FAILURE;
            case ConnectTimeoutException ignored -> Kind.CONNECT_TIMEOUT;
            case ConnectException connection ->
                    "Connection refused".equals(connection.getMessage())
                            ? Kind.CONNECT_REFUSED
                            : Kind.CONNECT_FAILED;
            case NoRouteToHostException ignored -> Kind.NO_ROUTE;
            case ClosedByInterruptException ignored -> Kind.IO_INTERRUPTED;
            case AsynchronousCloseException ignored -> Kind.ASYNC_CHANNEL_CLOSED;
            case ClosedChannelException ignored -> Kind.CHANNEL_CLOSED;
            case EOFException ignored -> Kind.UNEXPECTED_EOF;
            case SocketTimeoutException ignored -> Kind.IO_TIMEOUT;
            case SocketException socket -> classifySocketMessage(socket.getMessage());
            case Http2Exception ignored -> Kind.HTTP2_FAILURE;
            case IOException ignored -> Kind.IO_FAILURE;
            default -> null;
        };
    }

    private static Kind classifySocketMessage(String message) {
        // NIO exposes these observations as exception text, not structured errno or worker-fault
        // evidence.
        if ("Connection reset".equals(message) || "Connection reset by peer".equals(message)) {
            return Kind.CONNECTION_RESET;
        }
        return "Broken pipe".equals(message) ? Kind.BROKEN_PIPE : Kind.SOCKET_FAILURE;
    }

    private static Kind resolveKind(EnumSet<Kind> observedKinds, Status.Code code) {
        for (Kind kind : CAUSE_PRECEDENCE) {
            if (observedKinds.contains(kind)) {
                return kind;
            }
        }
        return classifyStatusCode(code);
    }

    private static Kind classifyStatusCode(Status.Code code) {
        return switch (code) {
            case DEADLINE_EXCEEDED -> Kind.DEADLINE_EXCEEDED;
            case CANCELLED -> Kind.CANCELLED;
            default -> Kind.RPC_FAILURE;
        };
    }

    private static Status grpcStatus(Throwable failure) {
        return switch (failure) {
            case StatusRuntimeException rpcFailure -> rpcFailure.getStatus();
            case StatusException rpcFailure -> rpcFailure.getStatus();
            default -> null;
        };
    }

    public static boolean isRetryableTransportFailure(Throwable failure) {
        return analyze(failure).retryableTransportFailure();
    }
}
