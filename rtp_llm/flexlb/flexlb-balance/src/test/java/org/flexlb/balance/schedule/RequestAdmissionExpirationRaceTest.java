package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks shutdown preserves committed responses and retries failed resource cleanup.
 */
class RequestAdmissionExpirationRaceTest {
    @Test
    void shutdownPreservesCommittedResponsesDespiteTheFirstCleanupFailure() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var contexts = new ArrayList<RequestContext>();
        var futures = new ArrayList<CompletableFuture<Response>>();
        var prefill = mock(PrefillEndpoint.class);
        var cleanupFailure = new IllegalStateException("first Prefill cleanup failed");
        var releases = new AtomicInteger();
        when(prefill.releaseRequest(any()))
                .thenAnswer(
                        call -> {
                            if (releases.incrementAndGet() == 1) {
                                throw cleanupFailure;
                            }
                            return true;
                        });
        try {
            for (long requestId = 310L; requestId < 313L; requestId++) {
                var context = RequestProtocolTestSupport.context(config, requestId);
                var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
                contexts.add(context);
                futures.add(future);
                var item =
                        org.flexlb.balance.schedule.SchedulingTestConfig.createRoute(
                                context,
                                new Response(),
                                null,
                                null,
                                prefill,
                                null,
                                null,
                                context.snapshot().createdAtMs());
                var admission =
                        SchedulerTestSupport.claim(registry, String.valueOf(requestId), future);
                try (var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                    assertNotNull(admission);
                    assertEquals(
                            org.flexlb.balance.strategy.RoutingResult.Status.SUCCESS,
                            RequestProtocolTestSupport.commitRoute(
                                    registry,
                                    item,
                                    RequestProtocolTestSupport.publication(() -> true)));
                }
            }
            assertTrue(RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry));
            assertSame(
                    cleanupFailure,
                    assertThrows(
                            IllegalStateException.class, registry::closeOutstandingAndTerminalize));
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry));
            assertSame(cleanupFailure, SchedulerTestSupport.failure(registry));
            assertEquals(
                    4,
                    releases.get(),
                    "one failed release retries without skipping another terminal owner");
            for (var future : futures) {
                assertTrue(future.get(2L, TimeUnit.SECONDS).isSuccess());
            }
            assertEquals(
                    0,
                    SchedulerTestSupport.repository(registry).liveRequestCount(),
                    "transient local release failure must recover without another notification");
            for (var context : contexts) {
                registry.closeRequest(context, null);
            }
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry));
            assertEquals(
                    4,
                    releases.get(),
                    "duplicate cleanup must not release an archived request again");
            assertEquals(0, SchedulerTestSupport.repository(registry).liveRequestCount());
            CompletableFuture.runAsync(
                            SchedulerTestSupport.responseCompletions(
                                            org.flexlb.balance.schedule.SchedulerTestSupport
                                                    .runtime(registry))
                                    ::close)
                    .get(2L, TimeUnit.SECONDS);
        } finally {
            SchedulerTestSupport.timer(
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(registry));
        }
    }
}
