package org.flexlb.engine.grpc;

import io.grpc.Attributes;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientStreamTracer;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.Status;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.channels.ClosedChannelException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.same;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcCallTraceTest {
    private final AtomicLong clock = new AtomicLong();
    private final GrpcTransportObserver observer = mock(GrpcTransportObserver.class);
    private final GrpcTransportObserver.SocketState socket =
            mock(GrpcTransportObserver.SocketState.class);
    private final GrpcCallTrace trace = new GrpcCallTrace(observer, clock::get);

    @Test
    @SuppressWarnings("unchecked")
    void interceptorAttachesTheFactoryWithoutReplacingExistingCallOptions() {
        Channel channel = mock(Channel.class);
        MethodDescriptor<Object, Object> method = mock(MethodDescriptor.class);
        ClientStreamTracer.Factory existing = mock(ClientStreamTracer.Factory.class);
        CallOptions options =
                CallOptions.DEFAULT
                        .withDeadlineAfter(5, TimeUnit.SECONDS)
                        .withWaitForReady()
                        .withStreamTracerFactory(existing)
                        .withOption(GrpcCallTrace.CALL_TRACE, trace);
        var captured = org.mockito.ArgumentCaptor.forClass(CallOptions.class);
        GrpcCallTrace.INTERCEPTOR.interceptCall(method, options, channel);
        verify(channel).newCall(same(method), captured.capture());
        CallOptions installed = captured.getValue();
        assertSame(options.getDeadline(), installed.getDeadline());
        assertEquals(options.isWaitForReady(), installed.isWaitForReady());
        assertSame(trace, installed.getOption(GrpcCallTrace.CALL_TRACE));
        assertEquals(java.util.List.of(existing, trace), installed.getStreamTracerFactories());
    }

    @Test
    @SuppressWarnings("unchecked")
    void interceptorPassesUntracedCallsThroughUnchanged() {
        Channel channel = mock(Channel.class);
        MethodDescriptor<Object, Object> method = mock(MethodDescriptor.class);
        ClientCall<Object, Object> call = mock(ClientCall.class);
        CallOptions options = CallOptions.DEFAULT.withWaitForReady();
        when(channel.newCall(method, options)).thenReturn(call);
        assertSame(call, GrpcCallTrace.INTERCEPTOR.interceptCall(method, options, channel));
        verify(channel).newCall(same(method), same(options));
    }

    @Test
    @SuppressWarnings("unchecked")
    void interceptorDiagnosticFailureStillStartsTheOriginalCall() {
        Channel channel = mock(Channel.class);
        MethodDescriptor<Object, Object> method = mock(MethodDescriptor.class);
        CallOptions options = mock(CallOptions.class);
        when(options.getOption(GrpcCallTrace.CALL_TRACE)).thenReturn(trace);
        when(options.withStreamTracerFactory(trace)).thenThrow(new OutOfMemoryError("diagnostic"));
        assertDoesNotThrow(() -> GrpcCallTrace.INTERCEPTOR.interceptCall(method, options, channel));
        verify(channel).newCall(same(method), same(options));
    }

    @Test
    void stagesOnlyAdvanceOnObservedTransportEvents() {
        assertEquals(GrpcCallTrace.Stage.WAITING_TRANSPORT, trace.snapshot().stage());
        ClientStreamTracer stream = stream(trace);
        assertEquals(GrpcCallTrace.Stage.WAITING_TRANSPORT, trace.snapshot().stage());
        clock.set(1_000_000);
        stream.streamCreated(Attributes.EMPTY, null);
        assertEquals(GrpcCallTrace.Stage.WAITING_HEADERS_WRITE, trace.snapshot().stage());
        clock.set(2_000_000);
        stream.outboundHeaders();
        assertEquals(GrpcCallTrace.Stage.WAITING_RESPONSE, trace.snapshot().stage());
        clock.set(3_000_000);
        stream.inboundHeaders(null);
        assertEquals(GrpcCallTrace.Stage.RECEIVING_RESPONSE, trace.snapshot().stage());
        clock.set(4_000_000);
        stream.inboundTrailers(null);
        assertEquals(GrpcCallTrace.Stage.RESPONSE_TRAILERS_RECEIVED, trace.snapshot().stage());
        assertEquals(-1, trace.snapshot().callbackDelayMs());
        assertEquals(-1, trace.snapshot().responseDeliveryMs());
        clock.set(5_000_000);
        stream.streamClosed(Status.OK);
        clock.set(8_000_000);
        trace.markCallbackStarted();
        var snapshot = trace.snapshot();
        assertEquals(3.0, snapshot.callbackDelayMs());
        assertEquals(4.0, snapshot.responseDeliveryMs());
        assertEquals(1.0, snapshot.streamCreatedMs());
        assertEquals(2.0, snapshot.outboundHeadersMs());
        assertEquals(3.0, snapshot.inboundHeadersMs());
        assertEquals(4.0, snapshot.inboundTrailersMs());
        assertEquals(5.0, snapshot.streamClosedMs());
        assertEquals(8.0, snapshot.callbackStartedMs());
        assertEquals("NONE", snapshot.transportEvent());
        assertNull(snapshot.transportCauseKind());
    }

    @Test
    void trailersOnlyResponseDoesNotRequireInitialHeaders() {
        ClientStreamTracer stream = stream(trace);
        stream.streamCreated(Attributes.EMPTY, null);
        clock.set(1_000_000);
        stream.inboundTrailers(null);
        stream.streamClosed(Status.UNAVAILABLE);
        trace.markCallbackStarted();
        assertEquals(GrpcCallTrace.Stage.RESPONSE_TRAILERS_RECEIVED, trace.snapshot().stage());
        assertEquals(-1, trace.snapshot().inboundHeadersMs());
        assertEquals(0, trace.snapshot().responseDeliveryMs());
    }

    @Test
    void earlyLocalFailureHasNoTransportTiming() {
        clock.set(1_000_000);
        trace.markCallbackStarted();
        var snapshot = trace.snapshot();
        assertEquals(GrpcCallTrace.Stage.WAITING_TRANSPORT, snapshot.stage());
        assertEquals(1, snapshot.callbackStartedMs());
        assertEquals(-1, snapshot.streamCreatedMs());
        assertEquals(-1, snapshot.callbackDelayMs());
        assertEquals(-1, snapshot.responseDeliveryMs());
    }

    @Test
    void socketEvidenceAndCauseAreKeptWhenObservedBeforeTerminalStatus() {
        when(observer.find(Attributes.EMPTY)).thenReturn(socket);
        ClientStreamTracer stream = stream(trace);
        clock.set(1_000_000);
        stream.streamCreated(Attributes.EMPTY, null);
        var event =
                new GrpcTransportObserver.Observation(
                        GrpcTransportObserver.Event.READ_FAILURE,
                        3_000_000,
                        GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED);
        when(socket.observation()).thenReturn(event);
        clock.set(4_000_000);
        stream.streamClosed(Status.UNAVAILABLE);
        clock.set(5_000_000);
        trace.markCallbackStarted();
        var snapshot = trace.snapshot();
        assertEquals("READ_FAILURE", snapshot.transportEvent());
        assertEquals(GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED, snapshot.transportCauseKind());
        assertEquals(3, snapshot.transportEventMs());
        assertEquals(
                "MASTER",
                snapshot.failureOrigin(GrpcFailure.analyze(new ClosedChannelException())));
    }

    @Test
    void peerClosureIsEvidenceOfAnActionWithoutWorkerFaultAttribution() {
        when(observer.find(Attributes.EMPTY)).thenReturn(socket);
        ClientStreamTracer stream = stream(trace);
        stream.streamCreated(Attributes.EMPTY, null);
        when(socket.observation())
                .thenReturn(
                        new GrpcTransportObserver.Observation(
                                GrpcTransportObserver.Event.PEER_EOF,
                                0,
                                GrpcFailure.Kind.UNEXPECTED_EOF));
        stream.streamClosed(Status.UNAVAILABLE);
        trace.markCallbackStarted();
        assertEquals("PEER_EOF", trace.snapshot().transportEvent());
        assertEquals(
                "UNCONFIRMED",
                trace.snapshot().failureOrigin(GrpcFailure.analyze(new ClosedChannelException())));
        assertEquals(
                "MASTER",
                trace.snapshot().failureOrigin(GrpcFailure.analyze(new OutOfMemoryError())));
        assertEquals("UNCONFIRMED", GrpcCallTrace.Snapshot.EMPTY.failureOrigin(null));
    }

    @ParameterizedTest
    @ValueSource(longs = {500_000, 3_000_001, 5_000_000})
    void socketEventsOutsideTheActiveStreamDoNotContaminateItsResult(long eventNanos) {
        when(observer.find(Attributes.EMPTY)).thenReturn(socket);
        ClientStreamTracer stream = stream(trace);
        clock.set(1_000_000);
        stream.streamCreated(Attributes.EMPTY, null);
        clock.set(3_000_000);
        stream.streamClosed(Status.OK);
        clock.set(4_000_000);
        trace.markCallbackStarted();
        when(socket.observation())
                .thenReturn(
                        new GrpcTransportObserver.Observation(
                                GrpcTransportObserver.Event.PEER_EOF,
                                eventNanos,
                                GrpcFailure.Kind.UNEXPECTED_EOF));
        assertEquals("NONE", trace.snapshot().transportEvent());
        assertNull(trace.snapshot().transportCauseKind());
        assertEquals(-1, trace.snapshot().transportEventMs());
    }

    @Test
    void callbackWithoutStreamClosedStillCutsOffLaterSocketEvents() {
        when(observer.find(Attributes.EMPTY)).thenReturn(socket);
        ClientStreamTracer stream = stream(trace);
        stream.streamCreated(Attributes.EMPTY, null);
        clock.set(2_000_000);
        trace.markCallbackStarted();
        when(socket.observation())
                .thenReturn(
                        new GrpcTransportObserver.Observation(
                                GrpcTransportObserver.Event.WRITE_BROKEN_PIPE,
                                3_000_000,
                                GrpcFailure.Kind.BROKEN_PIPE));
        assertEquals("NONE", trace.snapshot().transportEvent());
        assertEquals(-1, trace.snapshot().callbackDelayMs());
    }

    @Test
    void socketFailureAfterResponseTrailersDoesNotBlameTheCompletedResponse() {
        when(observer.find(Attributes.EMPTY)).thenReturn(socket);
        ClientStreamTracer stream = stream(trace);
        stream.streamCreated(Attributes.EMPTY, null);
        clock.set(1_000_000);
        stream.inboundTrailers(null);
        when(socket.observation())
                .thenReturn(
                        new GrpcTransportObserver.Observation(
                                GrpcTransportObserver.Event.READ_FAILURE,
                                2_000_000,
                                GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED));
        clock.set(3_000_000);
        stream.streamClosed(Status.UNKNOWN);
        clock.set(4_000_000);
        trace.markCallbackStarted();
        var snapshot = trace.snapshot();
        assertEquals("NONE", snapshot.transportEvent());
        assertNull(snapshot.transportCauseKind());
        assertEquals(
                "UNCONFIRMED",
                snapshot.failureOrigin(GrpcFailure.analyze(new ClosedChannelException())));
        assertEquals(1, snapshot.callbackDelayMs());
        assertEquals(3, snapshot.responseDeliveryMs());
    }

    @Test
    void lateTransportCallbacksDoNotRewriteTheCompletedStage() {
        ClientStreamTracer stream = stream(trace);
        stream.streamCreated(Attributes.EMPTY, null);
        clock.set(1_000_000);
        stream.outboundHeaders();
        clock.set(2_000_000);
        stream.streamClosed(Status.DEADLINE_EXCEEDED);
        clock.set(3_000_000);
        trace.markCallbackStarted();
        clock.set(4_000_000);
        stream.inboundHeaders(null);
        stream.inboundTrailers(null);
        trace.markCallbackStarted();
        assertEquals(GrpcCallTrace.Stage.WAITING_RESPONSE, trace.snapshot().stage());
        assertEquals(3, trace.snapshot().callbackStartedMs());
        assertEquals(-1, trace.snapshot().responseDeliveryMs());
    }

    @Test
    void anOlderStreamCannotMixItsEventsIntoTheLatestStream() {
        ClientStreamTracer previous = stream(trace);
        previous.streamCreated(Attributes.EMPTY, null);
        previous.outboundHeaders();
        clock.set(1_000_000);
        ClientStreamTracer latest = stream(trace);
        latest.streamCreated(Attributes.EMPTY, null);
        clock.set(2_000_000);
        previous.inboundTrailers(null);
        previous.streamClosed(Status.OK);
        assertEquals(GrpcCallTrace.Stage.WAITING_HEADERS_WRITE, trace.snapshot().stage());
        assertEquals(-1, trace.snapshot().outboundHeadersMs());
        assertEquals(-1, trace.snapshot().inboundTrailersMs());
        latest.streamClosed(Status.UNAVAILABLE);
        trace.markCallbackStarted();
        stream(trace).inboundTrailers(null);
        assertEquals(GrpcCallTrace.Stage.WAITING_HEADERS_WRITE, trace.snapshot().stage());
    }

    @ParameterizedTest
    @ValueSource(longs = {0, -10_000_000, Long.MAX_VALUE - 1_000_000})
    void timestampsSupportZeroNegativeAndNanoTimeWraparound(long start) {
        clock.set(start);
        GrpcCallTrace timed = new GrpcCallTrace(null, clock::get);
        ClientStreamTracer stream = stream(timed);
        stream.streamCreated(Attributes.EMPTY, null);
        clock.set(start + 2_000_000);
        stream.inboundTrailers(null);
        stream.streamClosed(Status.OK);
        clock.set(start + 3_000_000);
        timed.markCallbackStarted();
        assertEquals(0, timed.snapshot().streamCreatedMs());
        assertEquals(2, timed.snapshot().inboundTrailersMs());
        assertEquals(1, timed.snapshot().callbackDelayMs());
        assertEquals(1, timed.snapshot().responseDeliveryMs());
    }

    @Test
    void observerFailuresDoNotEscapeTheTransportCallback() {
        when(observer.find(Attributes.EMPTY)).thenThrow(new OutOfMemoryError("diagnostic"));
        ClientStreamTracer stream = stream(trace);
        assertDoesNotThrow(() -> stream.streamCreated(Attributes.EMPTY, null));
        assertEquals(GrpcCallTrace.Stage.WAITING_HEADERS_WRITE, trace.snapshot().stage());
    }

    @Test
    void snapshotObservationFailureReturnsSafeEmptyEvidence() {
        when(observer.find(Attributes.EMPTY)).thenReturn(socket);
        ClientStreamTracer stream = stream(trace);
        stream.streamCreated(Attributes.EMPTY, null);
        when(socket.observation()).thenThrow(new IllegalStateException("diagnostic"));
        assertSame(GrpcCallTrace.Snapshot.EMPTY, trace.snapshot());
    }

    @Test
    void clockFailuresNeverEscapeDiagnosticCallbacks() {
        GrpcCallTrace broken =
                assertDoesNotThrow(
                        () ->
                                new GrpcCallTrace(
                                        null,
                                        () -> {
                                            throw new OutOfMemoryError("diagnostic");
                                        }));
        ClientStreamTracer stream = stream(broken);
        assertDoesNotThrow(() -> stream.streamCreated(Attributes.EMPTY, null));
        assertDoesNotThrow(stream::outboundHeaders);
        assertDoesNotThrow(() -> stream.inboundHeaders(null));
        assertDoesNotThrow(() -> stream.inboundTrailers(null));
        assertDoesNotThrow(() -> stream.streamClosed(Status.OK));
        assertDoesNotThrow(broken::markCallbackStarted);
        assertEquals(GrpcCallTrace.Stage.WAITING_TRANSPORT, broken.snapshot().stage());
        assertEquals(-1, broken.snapshot().callbackDelayMs());
    }

    private static ClientStreamTracer stream(GrpcCallTrace trace) {
        return trace.newClientStreamTracer(
                ClientStreamTracer.StreamInfo.newBuilder().build(), new Metadata());
    }
}
