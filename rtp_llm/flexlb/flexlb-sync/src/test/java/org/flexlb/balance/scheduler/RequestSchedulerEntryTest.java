package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;

class RequestSchedulerEntryTest {
    @BeforeEach void configureTrace() { FlexlbTrace.configure(io.opentelemetry.api.OpenTelemetry.noop(), ""); }
    @AfterEach void clearTrace() { FlexlbTrace.configure(null, ""); }

    @ParameterizedTest @ValueSource(strings = {"DIRECT", "QUEUE", "BATCH"})
    void modesKeepOriginalFutureOwnershipAndCompatibleTraceLabels(String mode) throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        if (mode.equals("DIRECT")) { config.setScheduler(SchedulerConfig.direct()); }
        if (!mode.equals("BATCH")) { SchedulingTestConfig.useNonBatchDispatcher(config); }
        var owner = owner(config, mock(RecentCacheKeyTraceReporter.class));
        var context = RequestProtocolTestSupport.context(config, 700L);
        Span span = mock(Span.class);
        when(span.storeInContext(any(Context.class))).thenCallRealMethod();
        context.setTraceContext(Context.root().with(span));
        var router = mock(RequestWorkerSelector.class);
        when(router.select(context, null)).thenReturn(PlacementResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
        SchedulerTestSupport.configure(owner, config, router, mock(DeliveryMetricsReporter.class),
                mock(org.flexlb.balance.eviction.DecodeCapacityAcquirer.class), new PlacementAvailability());
        try {
            var result = owner.submit(context);
            assertSame(context.getFuture(), result);
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), result.get(3, TimeUnit.SECONDS).getCode());
            verify(span).setAttribute(FlexlbTrace.SCHEDULE_MODE, mode);
        } finally { owner.runtime.shutdown(); }
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void completionObservationPreservesResponseEvenWhenReporterFails(boolean reporterFails) {
        var config = SchedulingTestConfig.batchConfig();
        var reporter = mock(RecentCacheKeyTraceReporter.class);
        var owner = owner(config, reporter);
        var context = RequestProtocolTestSupport.context(config, 702L);
        if (reporterFails) { doThrow(new IllegalStateException("reporter unavailable")).when(reporter).report(context); }
        var pending = new CompletableFuture<Response>();
        context.setFuture(pending);
        try {
            owner.registerResponseCallback(context, pending);
            var response = new Response();
            response.setSuccess(true);
            pending.complete(response);
            assertSame(response, pending.join());
            assertSame(response, context.getResponse());
            verify(reporter).report(context);
        } finally { owner.runtime.shutdown(); }
    }

    @org.junit.jupiter.api.Test
    void observationsReadOnlyPublishedResultsWithoutWaitingForCompletionSideEffects() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 703L);
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

    @org.junit.jupiter.api.Test void missingBatchInputRejectsBeforeRegistration() {
        var config = SchedulingTestConfig.batchConfig();
        var owner = owner(config, mock(RecentCacheKeyTraceReporter.class));
        var context = RequestProtocolTestSupport.context(config, 701L);
        context.setGenerateInputPb(null);
        try {
            assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), owner.submit(context).join().getCode());
            assertFalse(owner.requests.retainsIdentity(701L));
        } finally { owner.runtime.shutdown(); }
    }

    private static AbstractRequestScheduler owner(FlexlbConfig config, RecentCacheKeyTraceReporter trace) {
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        return SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), trace);
    }
}
