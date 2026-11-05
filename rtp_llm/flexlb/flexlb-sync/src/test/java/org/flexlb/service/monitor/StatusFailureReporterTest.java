package org.flexlb.service.monitor;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.flexlb.config.WhitelistMetricsFilterConfig;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.EOFException;
import java.net.BindException;
import java.net.ConnectException;
import java.net.NoRouteToHostException;
import java.net.SocketException;
import java.nio.channels.AsynchronousCloseException;
import java.nio.channels.ClosedByInterruptException;
import java.nio.channels.ClosedChannelException;
import java.util.concurrent.RejectedExecutionException;
import java.util.stream.Stream;

import static org.flexlb.constant.MetricConstant.STATUS_FAILURE_DETAIL;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class StatusFailureReporterTest {
    private FlexMonitor monitor;
    private StatusFailureReporter reporter;

    @BeforeEach
    void setUp() {
        monitor = mock(FlexMonitor.class);
        reporter = new StatusFailureReporter(monitor);
    }

    @Test
    void registersFailureDetailAtPrecisePriority() {
        reporter.init();

        verify(monitor)
                .register(STATUS_FAILURE_DETAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @ParameterizedTest
    @MethodSource("failureOrigins")
    void reportsOnlyBoundedTagsAndConfirmedOrigin(Throwable cause, String origin) {
        var failure = GrpcFailure.analyze(Status.INTERNAL.withCause(cause).asRuntimeException());

        reporter.report(
                RoleType.DECODE,
                "worker_status",
                Stage.STATUS_RESPONSE_FAILED,
                "CONVERSION_FAILED",
                failure);

        verify(monitor)
                .report(
                        STATUS_FAILURE_DETAIL,
                        FlexMetricTags.of(
                                "role", RoleType.DECODE.getCode(),
                                "operation", "worker_status",
                                "stage", "STATUS_RESPONSE_FAILED",
                                "reason", "CONVERSION_FAILED",
                                "origin", origin,
                                "grpc_code", "INTERNAL"),
                        1.0);
    }

    static Stream<Arguments> failureOrigins() {
        return Stream.of(
                Arguments.of(new OutOfMemoryError("direct buffer memory"), "MASTER"),
                Arguments.of(new RejectedExecutionException("full"), "MASTER"),
                Arguments.of(new BindException(), "MASTER"),
                Arguments.of(new ConnectException("Connection refused"), "UNCONFIRMED"),
                Arguments.of(new NoRouteToHostException(), "UNCONFIRMED"),
                Arguments.of(new SocketException("Connection reset"), "UNCONFIRMED"),
                Arguments.of(new SocketException("Connection reset by peer"), "UNCONFIRMED"),
                Arguments.of(new SocketException("Broken pipe"), "UNCONFIRMED"),
                Arguments.of(new ClosedByInterruptException(), "UNCONFIRMED"),
                Arguments.of(new AsynchronousCloseException(), "UNCONFIRMED"),
                Arguments.of(new ClosedChannelException(), "UNCONFIRMED"),
                Arguments.of(new EOFException(), "UNCONFIRMED"),
                Arguments.of(
                        new InvalidProtocolBufferException("malformed response"), "UNCONFIRMED"),
                Arguments.of(new IllegalStateException("role mismatch"), "UNCONFIRMED"));
    }

    @Test
    void unsuccessfulApplicationResultDoesNotRequireAnException() {
        reporter.report(
                RoleType.PREFILL,
                "cache_status",
                Stage.CACHE_INDEX_UPDATE_FAILED,
                "NULL_RESULT",
                null);

        verify(monitor)
                .report(
                        STATUS_FAILURE_DETAIL,
                        FlexMetricTags.of(
                                "role", RoleType.PREFILL.getCode(),
                                "operation", "cache_status",
                                "stage", "CACHE_INDEX_UPDATE_FAILED",
                                "reason", "NULL_RESULT",
                                "origin", "UNCONFIRMED",
                                "grpc_code", "NONE"),
                        1.0);
    }

    @Test
    void prometheusExposesDetailsWhenTheirPrefixIsAllowed() {
        var whitelist = new WhitelistMetricsFilterConfig("flexlb_app_status_failure_detail");
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            whitelist.init();
            registry.config().meterFilter(whitelist.whitelistMetricsFilter());
            var exported = new StatusFailureReporter(new MicrometerFlexMonitor(registry));
            exported.init();
            exported.report(
                    RoleType.DECODE,
                    "worker_status",
                    Stage.RPC_COMPLETED,
                    "DEADLINE_EXCEEDED",
                    GrpcFailure.analyze(Status.DEADLINE_EXCEEDED.asRuntimeException()));

            String scrape = registry.scrape();
            assertTrue(scrape.contains("flexlb_app_status_failure_detail_total{"));
            assertTrue(scrape.contains("reason=\"DEADLINE_EXCEEDED\""));
            assertTrue(scrape.contains("origin=\"UNCONFIRMED\""));
        } finally {
            registry.close();
            MicrometerFlexMonitor.setAllowedMetrics(null);
        }
    }

    @ParameterizedTest
    @MethodSource("diagnosticFailures")
    void monitorFailureDoesNotEscape(Throwable diagnosticFailure) {
        doThrow(diagnosticFailure)
                .when(monitor)
                .report(eq(STATUS_FAILURE_DETAIL), any(FlexMetricTags.class), eq(1.0));

        assertDoesNotThrow(
                () ->
                        reporter.report(
                                RoleType.PREFILL,
                                "cache_status",
                                Stage.STATUS_RESPONSE_FAILED,
                                "DEADLINE_EXCEEDED",
                                GrpcFailure.analyze(
                                        Status.DEADLINE_EXCEEDED.asRuntimeException())));
    }

    static Stream<Arguments> diagnosticFailures() {
        return Stream.of(
                Arguments.of(new IllegalStateException("monitor failed")),
                Arguments.of(new OutOfMemoryError("metrics allocation")));
    }
}
