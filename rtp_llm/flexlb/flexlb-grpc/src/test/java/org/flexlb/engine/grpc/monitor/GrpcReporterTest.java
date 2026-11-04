package org.flexlb.engine.grpc.monitor;

import io.grpc.Status;
import org.flexlb.engine.grpc.GrpcCallTrace;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;

import java.net.BindException;
import java.net.ConnectException;
import java.net.SocketException;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.ClosedChannelException;
import java.util.Map;
import java.util.concurrent.RejectedExecutionException;
import java.util.stream.Stream;

import static org.flexlb.constant.MetricConstant.GRPC_CALL_FAILURE;
import static org.flexlb.constant.MetricConstant.GRPC_DELIVERY_DELAY;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

class GrpcReporterTest {
    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private final GrpcReporter reporter = new GrpcReporter(monitor);

    @Test
    void registersFinalCallFailuresAtPrecisePriority() {
        reporter.init();

        assertEquals("app.grpc.call.failure", GRPC_CALL_FAILURE);
        verify(monitor).register(GRPC_CALL_FAILURE, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        verify(monitor)
                .register(GRPC_DELIVERY_DELAY, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @ParameterizedTest
    @MethodSource("failureOrigins")
    void reportsOneFinalFailureWithOnlyFixedTagsAndConfirmedOrigin(
            Throwable cause, String reason, String origin) {
        var failure = GrpcFailure.analyze(Status.UNAVAILABLE.withCause(cause).asRuntimeException());

        reporter.reportCallFailure(
                "worker", false, "CALL_RESULT", failure, GrpcCallTrace.Snapshot.EMPTY);

        var tags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(GRPC_CALL_FAILURE), tags.capture(), eq(1.0));
        assertEquals(
                Map.of(
                        "service", "worker",
                        "retry", "false",
                        "phase", "CALL_RESULT",
                        "reason", reason,
                        "origin", origin,
                        "grpc_code", "UNAVAILABLE",
                        "transport_stage", "WAITING_TRANSPORT",
                        "transport_cause", "NONE",
                        "transport_event",
                                "CONNECT_REFUSED".equals(reason) ? "CONNECT_REFUSED" : "NONE"),
                tags.getValue().getTags());
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void unknownDeliveryTimesDoNotReportZeroSamples() {
        reporter.reportDeliveryDelay("worker", false, GrpcCallTrace.Snapshot.EMPTY);
        verifyNoMoreInteractions(monitor);
    }

    @Test
    void originalSocketOomIsVisibleEvenWhenGrpcOnlyReportsChannelClosed() {
        var transport =
                new GrpcCallTrace.Snapshot(
                        GrpcCallTrace.Stage.WAITING_RESPONSE,
                        "READ_FAILURE",
                        GrpcFailure.Kind.LOCAL_MEMORY_EXHAUSTED,
                        2,
                        -1,
                        0,
                        1,
                        -1,
                        -1,
                        10,
                        12,
                        9);
        reporter.reportCallFailure(
                "worker",
                false,
                "CALL_RESULT",
                GrpcFailure.analyze(
                        Status.UNAVAILABLE
                                .withCause(new ClosedChannelException())
                                .asRuntimeException()),
                transport);

        var tags = ArgumentCaptor.forClass(FlexMetricTags.class);
        verify(monitor).report(eq(GRPC_CALL_FAILURE), tags.capture(), eq(1.0));
        assertEquals("CHANNEL_CLOSED", tags.getValue().getTags().get("reason"));
        assertEquals("MASTER", tags.getValue().getTags().get("origin"));
        assertEquals("READ_FAILURE", tags.getValue().getTags().get("transport_event"));
        assertEquals("LOCAL_MEMORY_EXHAUSTED", tags.getValue().getTags().get("transport_cause"));
    }

    @Test
    void localDeliveryIntervalsUseMillisecondsAndSeparateStageLabels() {
        reporter.reportDeliveryDelay(
                "worker",
                false,
                new GrpcCallTrace.Snapshot(
                        GrpcCallTrace.Stage.RESPONSE_TRAILERS_RECEIVED,
                        "NONE",
                        null,
                        2.5,
                        6.0,
                        0,
                        1,
                        2,
                        3,
                        6.5,
                        9,
                        -1));

        verify(monitor)
                .report(
                        GRPC_DELIVERY_DELAY,
                        FlexMetricTags.of(
                                "service",
                                "worker",
                                "retry",
                                "false",
                                "stage",
                                "STREAM_CLOSED_TO_CALLBACK"),
                        2.5);
        verify(monitor)
                .report(
                        GRPC_DELIVERY_DELAY,
                        FlexMetricTags.of(
                                "service",
                                "worker",
                                "retry",
                                "false",
                                "stage",
                                "TRAILERS_TO_CALLBACK"),
                        6.0);
        verifyNoMoreInteractions(monitor);
    }

    static Stream<Arguments> failureOrigins() {
        return Stream.of(
                Arguments.of(
                        new ConnectException("Connection refused"),
                        "CONNECT_REFUSED",
                        "UNCONFIRMED"),
                Arguments.of(
                        new BindException("Cannot assign requested address"),
                        "LOCAL_BIND_FAILED",
                        "MASTER"),
                Arguments.of(
                        new RejectedExecutionException("full"),
                        "LOCAL_EXECUTOR_REJECTED",
                        "MASTER"),
                Arguments.of(
                        new OutOfMemoryError("direct buffer memory"),
                        "LOCAL_MEMORY_EXHAUSTED",
                        "MASTER"),
                Arguments.of(
                        new SocketException("Connection reset"), "CONNECTION_RESET", "UNCONFIRMED"),
                Arguments.of(new ClosedByInterruptException(), "IO_INTERRUPTED", "UNCONFIRMED"),
                Arguments.of(
                        new AsynchronousCloseException(), "ASYNC_CHANNEL_CLOSED", "UNCONFIRMED"));
    }

    @Test
    void retrySetupFailurePreservesServicePhaseAndDoesNotInferOriginFromGrpcCode() {
        reporter.reportCallFailure(
                "multimodal_worker",
                true,
                "CHANNEL_SETUP",
                GrpcFailure.analyze(Status.RESOURCE_EXHAUSTED.asRuntimeException()),
                GrpcCallTrace.Snapshot.EMPTY);

        verify(monitor)
                .report(
                        GRPC_CALL_FAILURE,
                        FlexMetricTags.of(
                                "service", "multimodal_worker",
                                "retry", "true",
                                "phase", "CHANNEL_SETUP",
                                "reason", "RPC_FAILURE",
                                "origin", "UNCONFIRMED",
                                "grpc_code", "RESOURCE_EXHAUSTED",
                                "transport_stage", "WAITING_TRANSPORT",
                                "transport_cause", "NONE",
                                "transport_event", "NONE"),
                        1.0);
        verifyNoMoreInteractions(monitor);
    }
}
