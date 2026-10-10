package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.Duration;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** A selected failure must not consume the cleanup deadline while admission still owns resources. */
class RequestAdmissionExpirationRaceTest {
    @Test
    void shutdownPublishesEveryClaimedResponseDespiteTheFirstCleanupFailure() throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        var contexts = new ArrayList<RequestContext>();
        var futures = new ArrayList<CompletableFuture<Response>>();
        var prefill = mock(PrefillEndpoint.class);
        var cleanupFailure = new IllegalStateException("first Prefill cleanup failed");
        var releases = new AtomicInteger();
        when(prefill.releaseRequest(any())).thenAnswer(call -> {
            if (releases.incrementAndGet() == 1) { throw cleanupFailure; }
            return true;
        });
        try {
            for (long requestId = 310L; requestId < 313L; requestId++) {
                var context = RequestProtocolTestSupport.context(config, requestId);
                var future = RequestProtocolTestSupport.register(registry, context);
                contexts.add(context);
                futures.add(future);
                var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                        prefill, null, null, context.createdAtMs());
                try (var admission = registry.claimAdmissionHandle(requestId, future);
                     var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                    assertNotNull(admission);
                    assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                            registry.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
                }
            }
            assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry));
            registry.closeOutstandingAndTerminalize();
            registry.runtime.continuations().awaitIdle();
            assertSame(cleanupFailure, SchedulerTestSupport.failure(registry));
            assertEquals(3, releases.get(), "one cleanup failure must not skip another claimed terminal owner");
            for (var future : futures) { assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess()); }
            assertEquals(1, SchedulerTestSupport.repository(registry).liveRequestCount(),
                    "only the failed cleanup stays tracked");
            for (var context : contexts) { registry.resumeCleanup(context); }
            assertEquals(4, releases.get(), "retry must release only the failed request");
            assertEquals(0, SchedulerTestSupport.repository(registry).liveRequestCount());
            CompletableFuture.runAsync(registry.runtime.responseCompletions()::close).get(2L, TimeUnit.SECONDS);
        } finally {
            // Keep the regression bounded when run against an implementation that strands claimed permits.
            for (var context : contexts) {
                var action = (TerminalAction) org.springframework.test.util.ReflectionTestUtils.getField(context, "terminalAction");
                if (action != null && action.publication() != null) { action.publication().abandonIfUnused(); }
            }
            registry.runtime.timer().close();
            registry.runtime.closeRequestExecutors();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void externalCompletionCleanupFailureStillPublishesAndClosesItsPermit(boolean asynchronous) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        var context = RequestProtocolTestSupport.context(config, 314L);
        try {
            var future = RequestProtocolTestSupport.register(registry, context);
            var prefill = mock(PrefillEndpoint.class);
            var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                    prefill, null, null, context.createdAtMs());
            try (var admission = registry.claimAdmissionHandle(314L, future);
                 var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        registry.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
            }
            var cleanupFailure = new IllegalStateException("external completion cleanup failed");
            when(prefill.releaseRequest(item)).thenThrow(cleanupFailure).thenReturn(true);
            if (asynchronous) {
                assertTrue(registry.publishDecisionResponseAsync(314L, future, Response.error(StrategyErrorType.DISPATCH_FAILED)));
                assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            } else {
                assertTrue(future.completeExceptionally(new IllegalStateException("caller failure")));
                assertTrue(future.isCompletedExceptionally());
            }
            assertSame(cleanupFailure, SchedulerTestSupport.failure(registry));
            assertEquals(RequestContext.RequestStage.FINALIZING, context.stage());
            CompletableFuture.runAsync(registry.runtime.responseCompletions()::close).get(2L, TimeUnit.SECONDS);
            registry.resumeCleanup(context);
            assertEquals(0, SchedulerTestSupport.repository(registry).liveRequestCount());
            verify(prefill, times(2)).releaseRequest(item);
        } finally {
            var action = (TerminalAction) org.springframework.test.util.ReflectionTestUtils.getField(context, "terminalAction");
            if (action != null && action.publication() != null) { action.publication().abandonIfUnused(); }
            registry.runtime.timer().close();
            registry.runtime.closeRequestExecutors();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cleanupFailureStillReleasesAdmissionAndAllowsExpiry(boolean abort) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            var context = RequestProtocolTestSupport.context(config, 302L);
            var future = RequestProtocolTestSupport.register(registry, context);
            var requestContext = registry.findRequestContext(302L);
            var prefill = mock(PrefillEndpoint.class);
            context.setFuture(future);
            var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                    prefill, null, null, requestContext.createdAtMs());
            var cleanupFailure = new IllegalStateException("Prefill cleanup failed");
            when(prefill.releaseRequest(item)).thenThrow(cleanupFailure).thenReturn(true);
            try (var admission = registry.claimAdmissionHandle(302L, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertTrue((registry.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                registry.failDeliveryPreparation(item, new IllegalStateException("preparation failed"));
                assertSame(cleanupFailure, assertThrows(IllegalStateException.class, () -> {
                    if (abort) {
                        admission.terminate(Response.error(StrategyErrorType.DISPATCH_FAILED));
                    } else {
                        admission.finish();
                    }
                }));
            }
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount() == 0);
            verify(prefill, times(2)).releaseRequest(item);
            assertEquals(RequestState.Phase.FAILED, requestContext.snapshot().state());
            assertTimeoutPreemptively(Duration.ofSeconds(2),
                    () -> assertTrue(RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)));
        } finally {
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
            }
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void pendingFailureAndLateDecodeStatusCannotStrandAnExpiredAdmission(boolean clientCancellation) throws Exception {
        long requestId = 301L;
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getRequestLifecycle().getRequest().setTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            var context = RequestProtocolTestSupport.context(config, requestId);
            var prefill = mock(PrefillEndpoint.class);
            var decode = RequestProtocolTestSupport.decodeEndpoint();
            var reservation = new DecodeResources.ReservationHandle(1L, requestId, 1L);
            var prefillStatus = new ServerStatus();
            prefillStatus.setRole(RoleType.PREFILL);
            prefillStatus.setServerIp("127.0.0.1");
            prefillStatus.setGrpcPort(8081);
            var future = RequestProtocolTestSupport.register(registry, context);
            RequestContext requestContext = registry.findRequestContext(requestId);
            context.setFuture(future);
            var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillStatus, null,
                    prefill, decode, reservation, requestContext.createdAtMs());

            try (var admission = registry.claimAdmissionHandle(requestId, future); var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertTrue((registry.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                if (clientCancellation) {
                    registry.cancel(requestId, 0L, CancelReason.CLIENT_CANCELLED);
                }
                registry.failDeliveryPreparation(item, new IllegalStateException("preparation failed"));
                if (clientCancellation) {
                    assertFalse(future.isDone());
                } else {
                    assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess(),
                            "failure publication must not wait for admission cleanup");
                    assertEquals(RequestState.Phase.FAILED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(requestId, 0L).state());
                }
                assertFalse(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).removeExactTerminal(org.flexlb.balance.scheduler.SchedulerTestSupport.terminalRecord(registry, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(requestContext.getRequestId(), 0L)), Long.MAX_VALUE));
                RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, requestContext.createdAtMs() + 300L);
                synchronized (requestContext) {
                    assertTrue(requestContext.inactivityDeadlineAtMs().isEmpty(),
                            "the fired deadline stays disarmed until the admission is completed");
                }
                RequestProtocolTestSupport.applyDecodeStatus(registry, decode,
                        DecodeResources.DecodeRequestStatus.active(reservation));
                synchronized (requestContext) {
                    requestContext.scheduler().acceptDecodeStatus(requestContext, decode, DecodeResources.DecodeRequestStatus.active(reservation), System.currentTimeMillis() + TimeUnit.HOURS.toMillis(1L));
                }
            }

            // Expiration remains a cleanup fact even when the result was already delivered;
            // later Decode activity cannot undo it before the admission owner exits.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(clientCancellation ? RequestState.Phase.CANCELLED : RequestState.Phase.FAILED,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(requestId, 0L).state());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
            verify(decode, times(1)).release(reservation, DecodeResources.ReleaseReason.EXPIRED);
            verify(prefill, times(1)).releaseRequest(item);
        } finally {
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
            }
        }
    }
}
