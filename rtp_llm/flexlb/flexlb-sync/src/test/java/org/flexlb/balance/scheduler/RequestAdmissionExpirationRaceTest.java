package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * A selected failure must not consume the cleanup deadline while admission still owns resources.
 */
class RequestAdmissionExpirationRaceTest {
    @Test
    void shutdownPublishesEveryClaimedResponseDespiteTheFirstCleanupFailure() throws Exception {
        var config = SchedulingTestConfig.batchConfig();
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
                        org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(
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
                            org.flexlb.balance.PlacementResult.Status.SUCCESS,
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
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
            assertSame(cleanupFailure, SchedulerTestSupport.failure(registry));
            assertEquals(
                    4,
                    releases.get(),
                    "one failed release retries without skipping another terminal owner");
            for (var future : futures) {
                assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            }
            assertEquals(
                    0,
                    SchedulerTestSupport.repository(registry).liveRequestCount(),
                    "transient local release failure must recover without another notification");
            for (var context : contexts) {
                registry.scheduleClose(context);
            }
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
            assertEquals(
                    4,
                    releases.get(),
                    "duplicate cleanup must not release an archived request again");
            assertEquals(0, SchedulerTestSupport.repository(registry).liveRequestCount());
            CompletableFuture.runAsync(
                            SchedulerTestSupport.responseCompletions(
                                            org.flexlb.balance.scheduler.SchedulerTestSupport
                                                    .runtime(registry))
                                    ::close)
                    .get(2L, TimeUnit.SECONDS);
        } finally {
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void externalCompletionCleanupFailureStillPublishesAndClosesItsPermit(boolean asynchronous)
            throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var context = RequestProtocolTestSupport.context(config, 314L);
        try {
            var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
            var prefill = mock(PrefillEndpoint.class);
            var item =
                    org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            prefill,
                            null,
                            null,
                            context.snapshot().createdAtMs());
            var admission = SchedulerTestSupport.claim(registry, "314", future);
            try (var completion = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertEquals(
                        org.flexlb.balance.PlacementResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                registry,
                                item,
                                RequestProtocolTestSupport.publication(() -> true)));
            }
            var cleanupFailure = new IllegalStateException("external completion cleanup failed");
            when(prefill.releaseRequest(item)).thenThrow(cleanupFailure).thenReturn(true);
            if (asynchronous) {
                assertTrue(
                        registry.publish(
                                context, Response.error(StrategyErrorType.DISPATCH_FAILED), false));
                assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            } else {
                assertTrue(
                        future.completeExceptionally(new IllegalStateException("caller failure")));
                assertTrue(future.isCompletedExceptionally());
            }
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
            assertSame(cleanupFailure, SchedulerTestSupport.failure(registry));
            assertEquals(SchedulerTestSupport.Stage.FINISHED, SchedulerTestSupport.stage(context));
            assertEquals(0, SchedulerTestSupport.repository(registry).liveRequestCount());
            CompletableFuture.runAsync(
                            SchedulerTestSupport.responseCompletions(
                                            org.flexlb.balance.scheduler.SchedulerTestSupport
                                                    .runtime(registry))
                                    ::close)
                    .get(2L, TimeUnit.SECONDS);
            registry.scheduleClose(context);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
            assertEquals(0, SchedulerTestSupport.repository(registry).liveRequestCount());
            verify(prefill, times(2)).releaseRequest(item);
        } finally {
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cleanupFailureStillReleasesAdmissionAndAllowsExpiry(boolean abort) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            var context = RequestProtocolTestSupport.context(config, 302L);
            var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
            var requestContext = registry.repository.findActive("302");
            var prefill = mock(PrefillEndpoint.class);
            context.data().setFuture(future);
            var item =
                    org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            null,
                            null,
                            prefill,
                            null,
                            null,
                            requestContext.snapshot().createdAtMs());
            var cleanupFailure = new IllegalStateException("Prefill cleanup failed");
            when(prefill.releaseRequest(item)).thenThrow(cleanupFailure).thenReturn(true);
            var admission = SchedulerTestSupport.claim(registry, "302", future);
            try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertTrue(
                        (RequestProtocolTestSupport.commitRoute(
                                        registry,
                                        item,
                                        RequestProtocolTestSupport.publication(() -> true))
                                == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                registry.onDeliveryResult(
                        new ContextResult.PreparationFailure(
                                null, item, "preparation failed", System.currentTimeMillis()));
                assertSame(
                        cleanupFailure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> {
                                    if (abort) {
                                        AbstractRequestScheduler.failPlacement(
                                                admission,
                                                Response.error(StrategyErrorType.DISPATCH_FAILED));
                                    } else {
                                        AbstractRequestScheduler.finishPlacement(admission);
                                    }
                                }));
            }
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                                            .liveRequestCount()
                                    == 0);
            verify(prefill, times(2)).releaseRequest(item);
            assertEquals(
                    RequestState.Phase.FAILED,
                    RequestState.from(requestContext.snapshot()).state());
            assertTimeoutPreemptively(
                    Duration.ofSeconds(2),
                    () ->
                            assertTrue(
                                    RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(
                                            registry)));
        } finally {
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
                registry.closeOutstandingAndTerminalize();
            }
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void pendingFailureAndLateDecodeStatusCannotStrandAnExpiredAdmission(boolean clientCancellation)
            throws Exception {
        long requestId = 301L;
        var config = SchedulingTestConfig.newConfig();
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getRequestLifecycle().getRequest().setTimeoutMs(300L);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        try {
            var context = RequestProtocolTestSupport.context(config, requestId);
            var prefill = mock(PrefillEndpoint.class);
            var decode = RequestProtocolTestSupport.decodeEndpoint();
            var reservation =
                    new DecodeResources.RequestReservation(1L, String.valueOf(requestId), 1L);
            var prefillStatus = new ServerStatus();
            prefillStatus.setRole(RoleType.PREFILL);
            prefillStatus.setServerIp("127.0.0.1");
            prefillStatus.setGrpcPort(8081);
            var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
            RequestContext requestContext =
                    registry.repository.findActive(String.valueOf(requestId));
            context.data().setFuture(future);
            var item =
                    org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(
                            context,
                            new Response(),
                            prefillStatus,
                            null,
                            prefill,
                            decode,
                            reservation,
                            requestContext.snapshot().createdAtMs());

            var admission = SchedulerTestSupport.claim(registry, String.valueOf(requestId), future);
            try (var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
                assertNotNull(admission);
                assertTrue(
                        (RequestProtocolTestSupport.commitRoute(
                                        registry,
                                        item,
                                        RequestProtocolTestSupport.publication(() -> true))
                                == org.flexlb.balance.PlacementResult.Status.SUCCESS));
                if (clientCancellation) {
                    registry.cancel(String.valueOf(requestId), 0L, CancelReason.CLIENT_CANCELLED);
                }
                registry.onDeliveryResult(
                        new ContextResult.PreparationFailure(
                                null, item, "preparation failed", System.currentTimeMillis()));
                if (clientCancellation) {
                    assertFalse(future.isDone());
                } else {
                    assertFalse(
                            future.get(2L, TimeUnit.SECONDS).isSuccess(),
                            "failure publication must not wait for admission cleanup");
                    assertEquals(
                            RequestState.Phase.FAILED,
                            org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                                    .getRequestState(String.valueOf(requestId), 0L)
                                    .state());
                }
                assertFalse(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                                .removeExactTerminal(
                                        org.flexlb.balance.scheduler.SchedulerTestSupport
                                                .terminalRecord(
                                                        registry,
                                                        org.flexlb.balance.scheduler
                                                                .SchedulerTestSupport.repository(
                                                                        registry)
                                                                .getRequestState(
                                                                        requestContext
                                                                                .data()
                                                                                .getRequestId(),
                                                                        0L)),
                                        Long.MAX_VALUE));
                RequestProtocolTestSupport.expireInactiveRequest(
                        registry, requestContext, requestContext.snapshot().createdAtMs() + 300L);
                synchronized (requestContext) {
                    assertTrue(
                            requestContext.snapshot().close().expired(),
                            "accepted inactivity expiry remains a cleanup requirement");
                    assertNotNull(requestContext.snapshot().placement());
                    if (clientCancellation) {
                        assertNull(requestContext.snapshot().outcome());
                        assertEquals(
                                ContextResult.EndKind.FRONTEND,
                                requestContext.snapshot().pendingEnd().kind());
                        assertEquals(
                                CancelReason.CLIENT_CANCELLED,
                                requestContext.snapshot().cancellation().reason());
                        assertFalse(future.isDone());
                    } else {
                        assertEquals(
                                RequestState.Phase.FAILED,
                                requestContext.snapshot().outcome().phase());
                    }
                }
                RequestProtocolTestSupport.applyDecodeStatus(
                        registry, decode, DecodeResources.DecodeRequestStatus.active(reservation));
                synchronized (requestContext) {
                    registry.onWorkerRequestEvent(
                            requestContext,
                            new ContextResult.WorkerRequestEvent(
                                    item,
                                    reservation,
                                    RoleType.DECODE,
                                    ContextResult.WorkerRequestEventType.ACTIVE,
                                    0L,
                                    null,
                                    System.currentTimeMillis() + TimeUnit.HOURS.toMillis(1L)));
                }
            }

            // Expiration remains a cleanup fact even when the result was already delivered;
            // later Decode activity cannot undo it before the admission owner exits.
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(
                    clientCancellation ? RequestState.Phase.CANCELLED : RequestState.Phase.FAILED,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                            .getRequestState(String.valueOf(requestId), 0L)
                            .state());
            assertEquals(
                    0,
                    org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                            .liveRequestCount());
            verify(decode, times(1)).release(reservation, DecodeResources.ReleaseReason.EXPIRED);
            verify(prefill, times(1)).releaseRequest(item);
        } finally {
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
                registry.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(
                                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry))
                        .close();
                SchedulerTestSupport.closeRequestExecutors(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
            }
        }
    }
}
