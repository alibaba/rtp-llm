package org.flexlb.balance.schedule;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.trace.FlexlbTrace;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks submission ownership, input validation, and tracing and reporting observations across scheduler modes.
 */
class RequestLifecycleSubmissionTest {
    @BeforeEach
    void configureTrace() {
        FlexlbTrace.configure(io.opentelemetry.api.OpenTelemetry.noop(), "");
    }

    @AfterEach
    void clearTrace() {
        FlexlbTrace.configure(null, "");
    }

    @ParameterizedTest
    @ValueSource(strings = {"DIRECT", "QUEUE"})
    void modesKeepOriginalFutureOwnershipAndCompatibleTraceLabels(String mode) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        if (mode.equals("DIRECT")) {
            config.setScheduler(ConfigTestSupport.direct());
        }
        var router = mock(RequestRouteSelector.class);
        var owner = owner(config, mock(TheoryCacheHitReporter.class), router);
        var context = RequestProtocolTestSupport.context(config, 700L);
        Span span = mock(Span.class);
        when(span.storeInContext(any(Context.class))).thenCallRealMethod();
        context.setTraceContext(Context.root().with(span));
        when(router.select(context, null))
                .thenReturn(
                        RequestResult.rejected(
                                Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
        try {
            var result =
                    SchedulerTestServices.entry(
                                    (AbstractRequestScheduler)
                                            org.flexlb.balance.schedule.SchedulerTestServices
                                                    .scheduler(owner))
                            .submit(context);
            assertSame(context.getFuture(), result);
            assertEquals(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    result.get(3, TimeUnit.SECONDS).getCode());
            verify(span).setAttribute(FlexlbTrace.SCHEDULE_MODE, mode);
        } finally {
            SchedulerTestSupport.shutdown(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(owner));
        }
    }

    @org.junit.jupiter.api.Test
    void observationsReadOnlyPublishedResultsWithoutWaitingForCompletionSideEffects() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 703L);
        assertNull(context.getResponse());
        var pending = new CompletableFuture<Response>();
        context.setFuture(pending);
        assertNull(context.getResponse());
        var response = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
        var observed = pending.thenApply(ignored -> context.getResponse());
        pending.complete(response);
        assertSame(response, observed.join());
        assertSame(response, context.getResponse());
        context.setFuture(CompletableFuture.failedFuture(new IllegalStateException("failed")));
        assertNull(context.getResponse());
        var cancelled = new CompletableFuture<Response>();
        context.setFuture(cancelled);
        cancelled.cancel(false);
        assertNull(context.getResponse());
    }

    private static AbstractRequestScheduler owner(
            FlexlbConfig config, TheoryCacheHitReporter trace, RequestRouteSelector router) {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        return SchedulerTestSupport.create(
                service,
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                trace,
                router,
                mock(org.flexlb.balance.schedule.DecodePreemption.class),
                new CapacityNotifier());
    }

    private static AbstractRequestScheduler owner(
            FlexlbConfig config, TheoryCacheHitReporter trace) {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        return SchedulerTestSupport.create(
                service,
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                trace);
    }
}
