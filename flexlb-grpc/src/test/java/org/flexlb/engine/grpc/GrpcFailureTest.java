package org.flexlb.engine.grpc;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import io.netty.channel.ConnectTimeoutException;
import io.netty.handler.codec.http2.Http2Error;
import io.netty.handler.codec.http2.Http2Exception;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

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
import java.time.Duration;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GrpcFailureTest {
    @Test
    void absentFailureProvidesNoFaultEvidence() {
        GrpcFailure.Analysis analysis = GrpcFailure.analyze(null);
        assertEquals(GrpcFailure.Kind.RPC_FAILURE, analysis.kind());
        assertEquals(Status.Code.UNKNOWN, analysis.grpcCode());
        assertNull(analysis.rootCause());
        assertFalse(GrpcFailure.isRetryableTransportFailure(null));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void bareGrpcStatusCannotIdentifyTheFaultOrigin(boolean checkedException) {
        for (Status status :
                new Status[] {
                    Status.INTERNAL, Status.RESOURCE_EXHAUSTED, Status.UNAVAILABLE, Status.UNKNOWN
                }) {
            Throwable rpcFailure = statusFailure(status, checkedException);
            var wrapped = new CompletionException(new ExecutionException(rpcFailure));
            for (Throwable failure : new Throwable[] {rpcFailure, wrapped}) {
                var analysis = GrpcFailure.analyze(failure);
                assertEquals(GrpcFailure.Kind.RPC_FAILURE, analysis.kind());
                assertEquals(status.getCode(), analysis.grpcCode());
                assertSame(rpcFailure, analysis.rootCause());
                assertFalse(analysis.confirmedLocalFailure());
                assertFalse(GrpcFailure.isRetryableTransportFailure(failure));
            }
        }
    }

    @Test
    void actualLocalCauseTakesPrecedenceOverGrpcStatusAndTransportWrapper() {
        var oom = new OutOfMemoryError("arbitrary text");
        var error =
                new ExecutionException(
                        new CompletionException(
                                Status.INTERNAL
                                        .withCause(
                                                new Http2Exception(
                                                        Http2Error.INTERNAL_ERROR,
                                                        "protocol wrapper",
                                                        oom))
                                        .asRuntimeException()));
        var analysis = GrpcFailure.analyze(error);
        assertEquals(GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED, analysis.kind());
        assertEquals(Status.Code.INTERNAL, analysis.grpcCode());
        assertSame(oom, analysis.rootCause());
        assertTrue(analysis.confirmedLocalFailure());
        assertFalse(GrpcFailure.isRetryableTransportFailure(error));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("competingCauses")
    void classificationPriorityDoesNotDependOnCauseOrder(
            String description,
            GrpcFailure.Kind expectedKind,
            boolean retryable,
            Function<Throwable, Throwable> preferred,
            Function<Throwable, Throwable> other) {
        Throwable preferredRoot = preferred.apply(null);
        Throwable otherWrapper = other.apply(preferredRoot);
        Throwable otherRoot = other.apply(null);
        Throwable preferredWrapper = preferred.apply(otherRoot);

        assertEquals(expectedKind, GrpcFailure.classify(otherWrapper));
        assertEquals(expectedKind, GrpcFailure.classify(preferredWrapper));
        assertEquals(retryable, GrpcFailure.isRetryableTransportFailure(otherWrapper));
        assertEquals(retryable, GrpcFailure.isRetryableTransportFailure(preferredWrapper));
        assertSame(preferredRoot, GrpcFailure.analyze(otherWrapper).rootCause());
        assertSame(otherRoot, GrpcFailure.analyze(preferredWrapper).rootCause());
    }

    private static Stream<Arguments> competingCauses() {
        Function<Throwable, Throwable> memory = cause -> withCause(new OutOfMemoryError(), cause);
        Function<Throwable, Throwable> rejected = RejectedExecutionException::new;
        Function<Throwable, Throwable> parsing =
                cause -> withCause(new InvalidProtocolBufferException("invalid response"), cause);
        Function<Throwable, Throwable> connectTimeout =
                cause -> withCause(new ConnectTimeoutException(), cause);
        Function<Throwable, Throwable> connect = cause -> withCause(new ConnectException(), cause);
        Function<Throwable, Throwable> refused =
                cause -> withCause(new ConnectException("Connection refused"), cause);
        Function<Throwable, Throwable> bind = cause -> withCause(new BindException(), cause);
        Function<Throwable, Throwable> noRoute =
                cause -> withCause(new NoRouteToHostException(), cause);
        Function<Throwable, Throwable> reset =
                cause -> withCause(new SocketException("Connection reset"), cause);
        Function<Throwable, Throwable> brokenPipe =
                cause -> withCause(new SocketException("Broken pipe"), cause);
        Function<Throwable, Throwable> interrupted =
                cause -> withCause(new ClosedByInterruptException(), cause);
        Function<Throwable, Throwable> asyncClosed =
                cause -> withCause(new AsynchronousCloseException(), cause);
        Function<Throwable, Throwable> closed =
                cause -> withCause(new ClosedChannelException(), cause);
        Function<Throwable, Throwable> eof = cause -> withCause(new EOFException(), cause);
        Function<Throwable, Throwable> readTimeout =
                cause -> withCause(new SocketTimeoutException(), cause);
        Function<Throwable, Throwable> socket = cause -> withCause(new SocketException(), cause);
        Function<Throwable, Throwable> http2 =
                cause -> new Http2Exception(Http2Error.INTERNAL_ERROR, "protocol failure", cause);
        Function<Throwable, Throwable> io = IOException::new;
        return Stream.of(
                Arguments.of(
                        "memory failure and executor rejection",
                        GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED,
                        false,
                        memory,
                        rejected),
                Arguments.of(
                        "executor rejection and response decoding",
                        GrpcFailure.Kind.LOCAL_EXECUTOR_REJECTED,
                        false,
                        rejected,
                        parsing),
                Arguments.of(
                        "local binding failure and response decoding",
                        GrpcFailure.Kind.LOCAL_BIND_FAILED,
                        false,
                        bind,
                        parsing),
                Arguments.of(
                        "response decoding and connection closure",
                        GrpcFailure.Kind.PROTOBUF_FAILURE,
                        false,
                        parsing,
                        closed),
                Arguments.of(
                        "connect timeout and generic connect failure",
                        GrpcFailure.Kind.CONNECT_TIMEOUT,
                        true,
                        connectTimeout,
                        connect),
                Arguments.of(
                        "connect failure and read timeout",
                        GrpcFailure.Kind.CONNECT_FAILED,
                        true,
                        connect,
                        readTimeout),
                Arguments.of(
                        "refused connection and generic connect wrapper",
                        GrpcFailure.Kind.CONNECT_REFUSED,
                        true,
                        refused,
                        connect),
                Arguments.of(
                        "no route and generic socket wrapper",
                        GrpcFailure.Kind.NO_ROUTE,
                        true,
                        noRoute,
                        socket),
                Arguments.of(
                        "reset connection and closed channel wrapper",
                        GrpcFailure.Kind.CONNECTION_RESET,
                        true,
                        reset,
                        closed),
                Arguments.of(
                        "broken pipe and closed channel wrapper",
                        GrpcFailure.Kind.BROKEN_PIPE,
                        true,
                        brokenPipe,
                        closed),
                Arguments.of(
                        "interrupted I/O and asynchronous close wrapper",
                        GrpcFailure.Kind.IO_INTERRUPTED,
                        true,
                        interrupted,
                        asyncClosed),
                Arguments.of(
                        "asynchronous close and closed channel wrapper",
                        GrpcFailure.Kind.ASYNC_CHANNEL_CLOSED,
                        true,
                        asyncClosed,
                        closed),
                Arguments.of(
                        "unexpected EOF and closed channel wrapper",
                        GrpcFailure.Kind.UNEXPECTED_EOF,
                        true,
                        eof,
                        closed),
                Arguments.of(
                        "read timeout and closed channel wrapper",
                        GrpcFailure.Kind.IO_TIMEOUT,
                        true,
                        readTimeout,
                        closed),
                Arguments.of(
                        "socket failure and closed channel wrapper",
                        GrpcFailure.Kind.SOCKET_FAILURE,
                        true,
                        socket,
                        closed),
                Arguments.of(
                        "read timeout and generic socket failure",
                        GrpcFailure.Kind.IO_TIMEOUT,
                        true,
                        readTimeout,
                        socket),
                Arguments.of(
                        "socket failure and HTTP/2 wrapper",
                        GrpcFailure.Kind.SOCKET_FAILURE,
                        true,
                        socket,
                        http2),
                Arguments.of(
                        "HTTP/2 failure and generic I/O wrapper",
                        GrpcFailure.Kind.HTTP2_FAILURE,
                        true,
                        http2,
                        io));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void outermostGrpcStatusWinsEvenWhenItsCodeIsUnknown(boolean checkedException) {
        Throwable inner = statusFailure(Status.DEADLINE_EXCEEDED, !checkedException);
        Throwable outer = statusFailure(Status.UNKNOWN.withCause(inner), checkedException);

        var analysis = GrpcFailure.analyze(new CompletionException(outer));

        assertEquals(Status.Code.UNKNOWN, analysis.grpcCode());
        assertEquals(GrpcFailure.Kind.RPC_FAILURE, analysis.kind());
        assertSame(inner, analysis.rootCause());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void outermostGrpcStatusControlsRetryWhenNestedStatusesDisagree(boolean checkedException) {
        for (Status terminal : new Status[] {Status.CANCELLED, Status.DEADLINE_EXCEEDED}) {
            Throwable brokenConnection =
                    statusFailure(
                            Status.UNAVAILABLE.withCause(new ClosedChannelException()),
                            !checkedException);
            Throwable cancelled =
                    statusFailure(terminal.withCause(brokenConnection), checkedException);
            assertEquals(terminal.getCode(), GrpcFailure.analyze(cancelled).grpcCode());
            assertFalse(GrpcFailure.isRetryableTransportFailure(cancelled));

            Throwable innerCancellation =
                    statusFailure(
                            terminal.withCause(new ClosedChannelException()), !checkedException);
            Throwable unavailable =
                    statusFailure(
                            Status.UNAVAILABLE.withCause(innerCancellation), checkedException);
            assertEquals(Status.Code.UNAVAILABLE, GrpcFailure.analyze(unavailable).grpcCode());
            assertTrue(GrpcFailure.isRetryableTransportFailure(unavailable));
        }
    }

    @Test
    void textCannotInventAnOomDeadlineOrBrokenConnection() {
        var failure =
                new IllegalStateException(
                        "OutOfMemoryError DEADLINE_EXCEEDED Connection reset http2 exception");
        assertEquals(GrpcFailure.Kind.RPC_FAILURE, GrpcFailure.classify(failure));
        assertFalse(GrpcFailure.isRetryableTransportFailure(failure));
        assertFalse(GrpcFailure.analyze(failure).confirmedLocalFailure());
    }

    @Test
    void strongLocalCauseClassificationDoesNotReadExceptionMessages() {
        var failure =
                new RejectedExecutionException() {
                    @Override
                    public String getMessage() {
                        throw new AssertionError("message must not be used");
                    }
                };
        assertEquals(GrpcFailure.Kind.LOCAL_EXECUTOR_REJECTED, GrpcFailure.classify(failure));
        assertFalse(GrpcFailure.isRetryableTransportFailure(failure));
    }

    @Test
    void statusCodeWorksWithoutDescriptionAndDoesNotClaimWorkerFault() {
        var deadline = Status.DEADLINE_EXCEEDED.asException();
        assertEquals(GrpcFailure.Kind.DEADLINE_EXCEEDED, GrpcFailure.classify(deadline));
        assertFalse(GrpcFailure.analyze(deadline).confirmedLocalFailure());
        assertFalse(GrpcFailure.isRetryableTransportFailure(deadline));
        assertEquals(
                GrpcFailure.Kind.RPC_FAILURE,
                GrpcFailure.classify(Status.RESOURCE_EXHAUSTED.asRuntimeException()));
    }

    @Test
    void typedConnectionErrorsArePreservedWithoutAttributingThemToWorker() {
        assertEquals(
                GrpcFailure.Kind.CONNECT_TIMEOUT,
                GrpcFailure.classify(
                        Status.UNAVAILABLE.withCause(new ConnectTimeoutException()).asException()));
        assertEquals(GrpcFailure.Kind.CONNECT_FAILED, GrpcFailure.classify(new ConnectException()));
        var closed = Status.UNKNOWN.withCause(new ClosedChannelException()).asRuntimeException();
        assertEquals(GrpcFailure.Kind.CHANNEL_CLOSED, GrpcFailure.classify(closed));
        assertTrue(GrpcFailure.isRetryableTransportFailure(closed));
        assertFalse(GrpcFailure.analyze(closed).confirmedLocalFailure());
    }

    @ParameterizedTest(name = "{1}")
    @MethodSource("refinedTransportCauses")
    void refinedTransportCausesRetainTheirLocalEvidenceAndRetryPolicy(
            Throwable cause, GrpcFailure.Kind kind, boolean confirmedLocal, boolean retryable) {
        var failure = Status.UNAVAILABLE.withCause(cause).asRuntimeException();
        var analysis = GrpcFailure.analyze(failure);

        assertEquals(kind, analysis.kind());
        assertEquals(Status.Code.UNAVAILABLE, analysis.grpcCode());
        assertSame(cause, analysis.rootCause());
        assertEquals(confirmedLocal, analysis.confirmedLocalFailure());
        assertEquals(retryable, analysis.retryableTransportFailure());
        assertEquals(retryable, GrpcFailure.isRetryableTransportFailure(failure));
        for (Status terminal : new Status[] {Status.CANCELLED, Status.DEADLINE_EXCEEDED}) {
            assertFalse(
                    GrpcFailure.analyze(terminal.withCause(cause).asException())
                            .retryableTransportFailure());
        }
    }

    private static Stream<Arguments> refinedTransportCauses() {
        return Stream.of(
                Arguments.of(new BindException(), GrpcFailure.Kind.LOCAL_BIND_FAILED, true, false),
                Arguments.of(new NoRouteToHostException(), GrpcFailure.Kind.NO_ROUTE, false, true),
                Arguments.of(
                        new ConnectException("Connection refused"),
                        GrpcFailure.Kind.CONNECT_REFUSED,
                        false,
                        true),
                Arguments.of(
                        new SocketException("Connection reset"),
                        GrpcFailure.Kind.CONNECTION_RESET,
                        false,
                        true),
                Arguments.of(
                        new SocketException("Connection reset by peer"),
                        GrpcFailure.Kind.CONNECTION_RESET,
                        false,
                        true),
                Arguments.of(
                        new SocketException("Broken pipe"),
                        GrpcFailure.Kind.BROKEN_PIPE,
                        false,
                        true),
                Arguments.of(
                        new ClosedByInterruptException(),
                        GrpcFailure.Kind.IO_INTERRUPTED,
                        false,
                        true),
                Arguments.of(
                        new AsynchronousCloseException(),
                        GrpcFailure.Kind.ASYNC_CHANNEL_CLOSED,
                        false,
                        true),
                Arguments.of(
                        new ClosedChannelException(), GrpcFailure.Kind.CHANNEL_CLOSED, false, true),
                Arguments.of(new EOFException(), GrpcFailure.Kind.UNEXPECTED_EOF, false, true));
    }

    @Test
    void exactSocketObservationsSurviveAddressAnnotationsThroughTheirOriginalCause() {
        var original = new ConnectException("Connection refused");
        var annotated =
                withCause(new ConnectException("Connection refused: /127.0.0.1:8080"), original);
        var analysis =
                GrpcFailure.analyze(Status.UNAVAILABLE.withCause(annotated).asRuntimeException());

        assertEquals(GrpcFailure.Kind.CONNECT_REFUSED, analysis.kind());
        assertSame(original, analysis.rootCause());
        assertFalse(analysis.confirmedLocalFailure());
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "Connection refused",
                "Connection reset",
                "Connection reset by peer",
                "Broken pipe"
            })
    void socketTextOutsideItsExpectedExceptionTypeCannotInventATransportObservation(String text) {
        assertEquals(
                GrpcFailure.Kind.RPC_FAILURE,
                GrpcFailure.classify(
                        Status.UNAVAILABLE.withDescription(text).asRuntimeException()));
        assertEquals(
                GrpcFailure.Kind.RPC_FAILURE,
                GrpcFailure.classify(new IllegalStateException(text)));
        assertEquals(GrpcFailure.Kind.IO_FAILURE, GrpcFailure.classify(new IOException(text)));
        assertEquals(
                GrpcFailure.Kind.SOCKET_FAILURE,
                GrpcFailure.classify(new SocketException("prefix " + text)));
        assertEquals(
                GrpcFailure.Kind.CONNECT_FAILED,
                GrpcFailure.classify(new ConnectException("prefix " + text)));
    }

    @Test
    void socketTextMatchingDoesNotGuessAcrossTypesOrStripMessageSuffixes() {
        assertEquals(
                GrpcFailure.Kind.SOCKET_FAILURE,
                GrpcFailure.classify(new SocketException("Connection refused")));
        assertEquals(
                GrpcFailure.Kind.CONNECT_FAILED,
                GrpcFailure.classify(new ConnectException("Connection reset")));
        assertEquals(
                GrpcFailure.Kind.CONNECT_FAILED,
                GrpcFailure.classify(new ConnectException("Broken pipe")));
        assertEquals(
                GrpcFailure.Kind.CONNECT_FAILED,
                GrpcFailure.classify(new ConnectException("Connection refused: /127.0.0.1:8080")));
        assertEquals(
                GrpcFailure.Kind.SOCKET_FAILURE,
                GrpcFailure.classify(new SocketException("Connection reset: /127.0.0.1:8080")));
        assertEquals(
                GrpcFailure.Kind.SOCKET_FAILURE,
                GrpcFailure.classify(new SocketException("Broken pipe\n")));
    }

    @Test
    void earlierAttemptsDoNotReplaceTheFinalAttemptCause() {
        var deadline = Status.DEADLINE_EXCEEDED.asRuntimeException();
        deadline.addSuppressed(new OutOfMemoryError());
        assertEquals(GrpcFailure.Kind.DEADLINE_EXCEEDED, GrpcFailure.classify(deadline));
    }

    @Test
    void cancellationAndDeadlineCannotRetryEvenWithAnIoCause() {
        for (Status status : new Status[] {Status.CANCELLED, Status.DEADLINE_EXCEEDED}) {
            assertFalse(
                    GrpcFailure.isRetryableTransportFailure(
                            status.withCause(new ClosedChannelException()).asRuntimeException()));
        }
    }

    @Test
    void responseParsingAndUnspecifiedIoFailureDoNotRecreateConnections() {
        var malformed =
                withCause(
                        new InvalidProtocolBufferException("Connection reset"), new EOFException());
        var parsing = Status.INTERNAL.withCause(malformed).asRuntimeException();
        assertEquals(GrpcFailure.Kind.PROTOBUF_FAILURE, GrpcFailure.classify(parsing));
        assertFalse(GrpcFailure.isRetryableTransportFailure(parsing));
        var io = Status.UNKNOWN.withCause(new IOException()).asException();
        assertEquals(GrpcFailure.Kind.IO_FAILURE, GrpcFailure.classify(io));
        assertFalse(GrpcFailure.isRetryableTransportFailure(io));
    }

    @Test
    void cyclicCauseChainWithoutRecognizedTypesCannotHangFailureHandling() {
        var first = new Exception();
        var second = new Exception(first);
        first.initCause(second);
        assertTimeoutPreemptively(
                Duration.ofSeconds(1),
                () -> {
                    assertEquals(GrpcFailure.Kind.RPC_FAILURE, GrpcFailure.classify(first));
                    assertFalse(GrpcFailure.isRetryableTransportFailure(first));
                });
    }

    @Test
    void cyclicCauseChainRetainsAlreadyRecognizedFailureEvidence() {
        for (Throwable recognized :
                new Throwable[] {new ConnectException(), new OutOfMemoryError()}) {
            var outer = new Exception(recognized);
            recognized.initCause(outer);
            assertTimeoutPreemptively(
                    Duration.ofSeconds(1),
                    () -> {
                        var analysis = GrpcFailure.analyze(outer);
                        boolean localMemoryFailure = recognized instanceof OutOfMemoryError;
                        assertEquals(
                                localMemoryFailure
                                        ? GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED
                                        : GrpcFailure.Kind.CONNECT_FAILED,
                                analysis.kind());
                        assertEquals(localMemoryFailure, analysis.confirmedLocalFailure());
                        assertSame(recognized, analysis.rootCause());
                        assertEquals(
                                !localMemoryFailure,
                                GrpcFailure.isRetryableTransportFailure(outer));
                    });
        }
    }

    private static Throwable statusFailure(Status status, boolean checkedException) {
        return checkedException ? status.asException() : status.asRuntimeException();
    }

    private static <T extends Throwable> T withCause(T failure, Throwable cause) {
        failure.initCause(cause);
        return failure;
    }
}
