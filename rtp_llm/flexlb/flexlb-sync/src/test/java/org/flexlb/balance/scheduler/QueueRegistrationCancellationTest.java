package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.*;

/** Real registration and queue lifecycle; endpoint selection is isolated by a mock router. */
class QueueRegistrationCancellationTest {
    @Test
    void registrationHookRunsBeforeSelectionAndNormalRequestsStillStart() throws Exception {
        try (Fixture f = new Fixture()) {
            AtomicBoolean listening = new AtomicBoolean();
            when(f.router.select(any(), nullable(String.class))).thenAnswer(call -> {
                assertTrue(listening.get(), "selection cannot precede listener installation");
                return PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            });
            RequestContext context = f.context(901L);
            var result = f.queue.submit(context, () -> {
                assertSame(f.queue, context.data().getScheduler());
                assertNotNull(context.data().getFuture());
                assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
                verify(f.router, never()).select(any(), nullable(String.class));
                listening.set(true);
            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            verify(f.router).select(any(), nullable(String.class));
        }
    }

    @Test
    void cancellationDuringRegistrationHookPreventsEnqueueAndSelection() throws Exception {
        try (Fixture f = new Fixture()) {
            RequestContext context = f.context(902L);
            var result = f.queue.submit(context, () -> {
                assertSame(f.queue, context.data().getScheduler());
                assertNotNull(f.queue.cancel(902L, 0L, CancelReason.CLIENT_CANCELLED));
            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
            verify(f.router, never()).select(any(), nullable(String.class));
        }
    }

    @Test
    void concurrentCancellationBeforeHookReturnsPreventsSelection() throws Exception {
        try (Fixture f = new Fixture(); var executor = Executors.newSingleThreadExecutor()) {
            var registered = new CountDownLatch(1);
            var resume = new CountDownLatch(1);
            var submitted = executor.submit(() -> f.queue.submit(f.context(903L), () -> {
                registered.countDown();
                try {
                    assertTrue(resume.await(2L, TimeUnit.SECONDS));
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(error);
                }
            }));
            try {
                assertTrue(registered.await(2L, TimeUnit.SECONDS));
                assertNotNull(f.queue.cancel(903L, 0L, CancelReason.CLIENT_CANCELLED));
                SchedulerTestSupport.runtime(f.queue).shutdown();

                assertFalse(submitted.isDone(), "a cancelled hook holds no scheduling resources");
            } finally {
                resume.countDown();
            }
            assertFalse(submitted.get(2L, TimeUnit.SECONDS).get(2L, TimeUnit.SECONDS).isSuccess());
            verify(f.router, never()).select(any(), nullable(String.class));
            SchedulerTestSupport.runtime(f.queue).shutdown();

        }
    }

    @Test
    void failedHookTerminatesRequestAndRejectedSubmissionsDoNotInvokeHook() throws Exception {
        try (Fixture f = new Fixture()) {
            var result = f.queue.submit(f.context(904L), () -> {
                throw new IllegalStateException("listener installation failed");
            });
            assertFalse(result.get(2L, TimeUnit.SECONDS).isSuccess());
            verify(f.router, never()).select(any(), nullable(String.class));
            SchedulerTestSupport.runtime(f.queue).requests().stopAccepting();
            AtomicBoolean called = new AtomicBoolean();
            assertFalse(f.queue.submit(f.context(905L), () -> called.set(true)).get().isSuccess());
            assertFalse(called.get());
        }
    }

    @Test
    void cacheVictimTerminatesAndHighHitStillExpiresWithoutNewArrivals() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(100L);
        policy.setMaxWaitMs(800L);
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class))).thenAnswer(call -> {
                RequestContext context = call.getArgument(0);
                return new PrefillQueueShedding.Estimate(100, context.data().getRequestId() == 910L ? 0 : 99, 100, 1);
            });
            var cold = f.context(910L);
            var hot = f.context(911L);
            cold.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50,
                    f.config.resolveExpiresAtMs(System.currentTimeMillis())));
            hot.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50,
                    f.config.resolveExpiresAtMs(System.currentTimeMillis())));
            var coldResult = f.queue.submit(cold);
            var hotResult = f.queue.submit(hot);
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), coldResult.get(2, TimeUnit.SECONDS).getCode());
            assertFalse(hotResult.get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(System.currentTimeMillis() >= hot.data().getRequestExpiresAtMs());
            assertEquals(CancelReason.DEADLINE_EXCEEDED, hot.snapshot().cancellation().reason());
            RequestProtocolTestSupport.awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.queue) == 0);
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
        }
    }

    @Test
    void decodeOnlyPressureDoesNotTriggerCacheEviction() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(1L);
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.DECODE, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class)))
                    .thenReturn(new PrefillQueueShedding.Estimate(100, 0, 10_000, 1));
            var result = f.queue.submit(f.context(912L));
            assertThrows(java.util.concurrent.TimeoutException.class, () -> result.get(300, TimeUnit.MILLISECONDS));
            f.queue.cancel(912L, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), result.get(2, TimeUnit.SECONDS).getCode());
        }
    }

    @Test
    void predictedOverloadCannotEvictBeforeRealQueueWaitThreshold() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(1_000L);
        policy.setMaxWaitMs(3_000L);
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class)))
                    .thenReturn(new PrefillQueueShedding.Estimate(100, 0, Long.MAX_VALUE, 1));
            var result = f.queue.submit(f.context(920L));
            verify(f.router, timeout(1000)).select(any(), nullable(String.class));
            assertThrows(java.util.concurrent.TimeoutException.class, () -> result.get(300, TimeUnit.MILLISECONDS));
            f.queue.cancel(920L, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), result.get(2, TimeUnit.SECONDS).getCode());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void actualWaitTriggersEvictionWithoutPredictedWork(boolean missingEstimate) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(100L);
        policy.setMaxWaitMs(3_000L);
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class))).thenReturn(missingEstimate ? null
                    : new PrefillQueueShedding.Estimate(100, 99, 0, 1000));
            long start = System.nanoTime();
            var response = f.queue.submit(f.context(921L)).get(3, TimeUnit.SECONDS);
            assertTrue(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start) > 100);
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
            RequestProtocolTestSupport.awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.queue) == 0);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({"false,0", "false,90", "true,0", "true,90"})
    void lowEngineQueueSuppressesSoftEvictionButNotHardDeadline(boolean batch, long hitTokens) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(100L);
        policy.setMaxWaitMs(800L);
        try (Fixture f = new Fixture(policy, batch)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null)));
            when(f.router.estimateQueuedWork(any(), nullable(String.class)))
                    .thenReturn(new PrefillQueueShedding.Estimate(100, hitTokens, Long.MAX_VALUE, 1));
            when(f.router.isPrefillClusterOverloaded(null)).thenReturn(false);
            var context = f.context(923L);
            context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50,
                    f.config.resolveExpiresAtMs(System.currentTimeMillis())));
            var result = f.queue.submit(context);
            assertThrows(java.util.concurrent.TimeoutException.class, () -> result.get(400, TimeUnit.MILLISECONDS));
            assertFalse(result.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellation().reason());
            verify(f.router, atLeastOnce()).isPrefillClusterOverloaded(null);
        }
    }

    @Test
    void hardDeadlineWinsEvenWithRecoveryAndNoCacheEstimate() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(200L);
        policy.setMaxWaitMs(200L);
        policy.setRecovery(new org.flexlb.config.PrefillQueueSheddingConfig.RecoveryConfig());
        try (Fixture f = new Fixture(policy)) {
            when(f.router.select(any(), nullable(String.class))).thenReturn(PlacementResult.blocked(
                    new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null)));
            var context = f.context(922L);
            context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(99,
                    f.config.resolveExpiresAtMs(System.currentTimeMillis())));
            assertFalse(f.queue.submit(context).get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(CancelReason.DEADLINE_EXCEEDED, context.snapshot().cancellation().reason());
        }
    }

    private static void block(Fixture f, org.flexlb.dao.route.RoleType role) {
        when(f.router.select(any(), nullable(String.class))).thenAnswer(call ->
                PlacementResult.blocked(new PlacementKey(role, call.getArgument(1), null)));
        when(f.router.estimateQueuedWork(any(), nullable(String.class)))
                .thenReturn(new PrefillQueueShedding.Estimate(100, 0, 100, 1));
    }

    private static PrefillColdRequestProtection.Snapshot protection(Fixture f, String pool) {
        return (PrefillColdRequestProtection.Snapshot) f.queue.coldRequestProtectionSnapshot().get(pool);
    }

    private static void awaitParked(RequestContext context) throws Exception {
        var entries = (java.util.Map<?, ?>) org.springframework.test.util.ReflectionTestUtils
                .getField(context.data().getScheduler(), "queuedEntries");
        RequestProtocolTestSupport.awaitCondition(() -> {
            var entry = (GlobalQueueEntry) entries.get(context);
            return entry != null && entry.lastBlocker != null;
        });
    }

    private static void hardTimeout(Fixture f, long id, long expectedCount) throws Exception {
        var context = f.context(id);
        var future = f.queue.submit(context);
        awaitParked(context);
        var exact = context.snapshot().deadlines().inactivity();
        assertNotNull(exact);
        f.queue.onDeadline(exact, exact.atMs());
        assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
        RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>") != null
                && protection(f, "<default>").hardTimeouts() == expectedCount);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void concurrentDeadlineCallbacksCountExactlyOnceForRequestAndInactivity(boolean inactivity) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(60_000); policy.setMaxWaitMs(120_000);
        try (var f = new Fixture(policy); var threads = Executors.newFixedThreadPool(8)) {
            f.config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            block(f, org.flexlb.dao.route.RoleType.PREFILL);
            var context = f.context(940L);
            var future = f.queue.submit(context);
            awaitParked(context);
            var exact = inactivity ? context.snapshot().deadlines().inactivity() : context.snapshot().deadlines().request();
            assertNotNull(exact);
            var tasks = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int i = 0; i < 16; i++) { tasks.add(threads.submit(() -> f.queue.onDeadline(exact, exact.atMs()))); }
            for (var task : tasks) { task.get(2, TimeUnit.SECONDS); }
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>") != null
                    && protection(f, "<default>").hardTimeouts() == 1);
            f.queue.onDeadline(exact, exact.atMs() + 1);
            assertEquals(1, protection(f, "<default>").hardTimeouts());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value = org.flexlb.dao.route.RoleType.class, names = {"DECODE", "VIT"})
    void otherRoleTimeoutsCannotActivateColdProtection(org.flexlb.dao.route.RoleType role) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        try (var f = new Fixture(policy)) {
            f.config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            block(f, role);
            var context = f.context(941L); var future = f.queue.submit(context);
            awaitParked(context);
            var exact = context.snapshot().deadlines().inactivity();
            f.queue.onDeadline(exact, exact.atMs());
            assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>") != null);
            assertEquals(0, protection(f, "<default>").hardTimeouts());
        }
    }

    @Test
    void clientCancellationAndPrematureDeadlineCancellationCannotCountAsHardTimeouts() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        try (var f = new Fixture(policy)) {
            f.config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            block(f, org.flexlb.dao.route.RoleType.PREFILL);
            for (var reason : new CancelReason[]{CancelReason.CLIENT_CANCELLED, CancelReason.DEADLINE_EXCEEDED}) {
                var context = f.context(reason.ordinal() + 950L); var future = f.queue.submit(context);
                awaitParked(context);
                var exact = context.snapshot().deadlines().inactivity();
                f.queue.cancel(context, 0L, reason);
                f.queue.onDeadline(exact, exact.atMs());
                assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            }
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>") != null);
            assertEquals(0, protection(f, "<default>").hardTimeouts());
        }
    }

    @Test
    void realQueueRetainsHardHistoryAcrossEmptyPeriodsAndGraduallyEvictsOnlyItsYoungColdRequests() throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(60_000); policy.setMaxWaitMs(120_000);
        var clock = new java.util.concurrent.atomic.AtomicLong();
        try (var f = new Fixture(policy, false, clock::get)) {
            f.config.getRequestLifecycle().getRequest().setTimeoutMs(100_000L);
            block(f, org.flexlb.dao.route.RoleType.PREFILL);
            when(f.router.estimateQueuedWork(any(), nullable(String.class))).thenAnswer(call -> {
                RequestContext context = call.getArgument(0);
                return new PrefillQueueShedding.Estimate(100, context.data().getRequestId() == 1200L ? 99 : 0, 100, 1);
            });
            when(f.router.resolvePolicyGroup(any())).thenAnswer(call -> {
                RequestContext context = call.getArgument(0);
                return context.data().getRequestId() == 1201L ? "other" : null;
            });
            hardTimeout(f, 960L, 1);
            clock.set(60_000); hardTimeout(f, 961L, 2);
            clock.set(120_000); hardTimeout(f, 962L, 3);
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>").state() == PrefillColdRequestProtection.State.PROTECTING);
            assertEquals(0, protection(f, "<default>").strength());
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.queue));
            var cold = new java.util.ArrayList<CompletableFuture<Response>>();
            for (long id = 1000; id < 1100; id++) { cold.add(f.queue.submit(f.context(id))); }
            var hot = f.context(1200L);
            var hotFuture = f.queue.submit(hot);
            var otherPool = f.context(1201L);
            var otherFuture = f.queue.submit(otherPool);
            awaitParked(hot); awaitParked(otherPool);
            for (int i = 1; i <= 20; i++) {
                clock.set(120_000 + i * 1_000L);
                double expected = i / 60.0;
                RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>").strength() >= expected - 1e-9);
            }
            long evicted = cold.stream().filter(CompletableFuture::isDone).count();
            assertTrue(evicted > 0 && evicted < 20, "young cold eviction must ramp, not reject the whole burst");
            for (var future : cold) {
                if (future.isDone()) { assertTrue(future.join().getErrorMessage().contains("cold-request grace")); }
            }
            assertFalse(hotFuture.isDone(), () -> "hot response: " + hotFuture.join().getErrorMessage());
            assertFalse(otherFuture.isDone(), () -> "other pool response: " + otherFuture.join().getErrorMessage());
            assertEquals(PrefillColdRequestProtection.State.NORMAL, protection(f, "group:other").state());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.NullAndEmptySource
    @org.junit.jupiter.params.provider.ValueSource(strings = {"explicit-group", "vit-group"})
    void workerQueueDeadlineUsesRequestPoolAfterGlobalEntryIsGone(String pool) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        try (var f = new Fixture(policy)) {
            var context = f.context(1300L);
            context.setPrefillQueuePool(pool);
            var before = mock(ContextResult.Snapshot.class);
            var route = mock(RequestRoute.class);
            when(before.route()).thenReturn(route);
            when(route.coldTokenBalance()).thenReturn(true);
            // Worker metadata must not replace the request's routing scope, including null.
            f.queue.onQueueDeadlineAccepted(context, before);
            String key = pool == null ? "<default>" : "group:" + pool;
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, key) != null
                    && protection(f, key).hardTimeouts() == 1);
            verify(route).coldTokenBalance();
            verifyNoMoreInteractions(route);
            assertEquals(1, f.queue.coldRequestProtectionSnapshot().size());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void hotOrUnknownPrefillTimeoutsCannotWithdrawColdGrace(boolean unknown) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(60_000); policy.setMaxWaitMs(120_000);
        var clock = new java.util.concurrent.atomic.AtomicLong();
        try (var f = new Fixture(policy, false, clock::get)) {
            block(f, org.flexlb.dao.route.RoleType.PREFILL);
            when(f.router.estimateQueuedWork(any(), nullable(String.class))).thenReturn(unknown ? null
                    : new PrefillQueueShedding.Estimate(100, 90, 10, 1));
            for (int i = 0; i < 3; i++) {
                clock.set(i * 60_000L);
                var context = f.context(1400L + i);
                var future = f.queue.submit(context);
                awaitParked(context);
                var exact = context.snapshot().deadlines().inactivity();
                f.queue.onDeadline(exact, exact.atMs());
                assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
            }
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>") != null);
            assertEquals(0, protection(f, "<default>").hardTimeouts());
            assertEquals(PrefillColdRequestProtection.State.NORMAL, protection(f, "<default>").state());
        }
    }

    @Test
    void hotWorkerQueueTimeoutDoesNotCreateColdProtectionPool() throws Exception {
        try (var f = new Fixture(new org.flexlb.config.PrefillQueueSheddingConfig())) {
            var before = mock(ContextResult.Snapshot.class);
            var route = mock(RequestRoute.class);
            when(before.route()).thenReturn(route);
            var context = f.context(1500L);
            f.queue.onQueueDeadlineAccepted(context, before);
            verify(route).coldTokenBalance();
            var pending = (java.util.Queue<?>) org.springframework.test.util.ReflectionTestUtils.getField(f.queue, "hardTimeouts");
            assertTrue(pending.isEmpty());
            assertTrue(f.queue.coldRequestProtectionSnapshot().isEmpty());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value = org.flexlb.balance.strategy.CostBasedPrefillStrategy.ColdAdmission.class,
            names = {"AVAILABLE", "UNKNOWN", "INSUFFICIENT_CAPACITY"})
    void currentAdmissionVetoesHistoricalProtectionAndWakesColdRequests(
            org.flexlb.balance.strategy.CostBasedPrefillStrategy.ColdAdmission admission) throws Exception {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setExpectedWaitThresholdMs(60_000); policy.setMaxWaitMs(120_000);
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var probes = new java.util.concurrent.atomic.AtomicInteger();
        var plans = new java.util.concurrent.atomic.AtomicInteger();
        try (var f = new Fixture(policy, false, clock::get)) {
            block(f, org.flexlb.dao.route.RoleType.PREFILL);
            when(f.router.select(any(), nullable(String.class))).thenAnswer(call -> {
                plans.incrementAndGet();
                return PlacementResult.blocked(new PlacementKey(org.flexlb.dao.route.RoleType.PREFILL, null, null));
            });
            when(f.router.coldPrefillAdmission(any(), nullable(String.class))).thenAnswer(call -> {
                probes.incrementAndGet();
                if (admission == org.flexlb.balance.strategy.CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY) {
                    f.availability.changed(PlacementKey.anyGroup(org.flexlb.dao.route.RoleType.PREFILL));
                }
                return admission;
            });
            hardTimeout(f, 1600L, 1);
            clock.set(60_000); hardTimeout(f, 1601L, 2);
            clock.set(120_000); hardTimeout(f, 1602L, 3);
            RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>").state() == PrefillColdRequestProtection.State.PROTECTING);
            var cold = new java.util.ArrayList<CompletableFuture<Response>>();
            for (int i = 0; i < 100; i++) { cold.add(f.queue.submit(f.context(1700L + i))); }
            RequestProtocolTestSupport.awaitCondition(() -> plans.get() >= 103);
            for (int i = 1; i <= 20; i++) {
                clock.set(120_000 + i * 1_000L);
                double strength = i / 60.0;
                RequestProtocolTestSupport.awaitCondition(() -> protection(f, "<default>").strength() >= strength - 1e-9);
            }
            assertTrue(probes.get() > 0, "historical pressure must reach a current admission recheck");
            assertTrue(plans.get() > 103, "capacity veto must actively wake placement without waiting for another edge");
            assertTrue(cold.stream().noneMatch(CompletableFuture::isDone), "available/unknown capacity cannot authorize early rejection");
            assertEquals(PrefillColdRequestProtection.State.PROTECTING, protection(f, "<default>").state(),
                    "a capacity recovery must not erase the historical state");
        }
    }

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        final RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        final SchedulerRuntime runtime;
        final PlacementAvailability availability = new PlacementAvailability();
        final QueuedRequestScheduler queue;

        Fixture() { this(null); }

        Fixture(org.flexlb.config.PrefillQueueSheddingConfig policy) { this(policy, false); }

        Fixture(org.flexlb.config.PrefillQueueSheddingConfig policy, boolean batch) {
            this(policy, batch, () -> TimeUnit.NANOSECONDS.toMillis(System.nanoTime()));
        }

        Fixture(org.flexlb.config.PrefillQueueSheddingConfig policy, boolean batch, java.util.function.LongSupplier clock) {
            when(router.isPrefillClusterOverloaded(nullable(String.class))).thenReturn(true);
            when(router.coldPrefillAdmission(any(), nullable(String.class)))
                    .thenReturn(org.flexlb.balance.strategy.CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY);
            config.queueScheduler().setPrefillQueueShedding(policy);
            SchedulingTestConfig.useFifoQueue(config);
            if (policy != null) {
                SchedulingTestConfig.useNonBatchDispatcher(config);
                config.queueScheduler().getDecision().setType(org.flexlb.config.DecisionPolicyConfig.Type.SINGLE);
            }
            if (batch) {
                SchedulingTestConfig.useBatchDispatcher(config);
                config.getRouter().getRoles().getPrefill().setTokenBalance(new org.flexlb.config.RoutingConfig.TokenBalanceConfig());
            }
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var batches = mock(DeliveryMetricsReporter.class);
            runtime = new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class), batches,
                    mock(RequestSchedulerReporter.class), SchedulerTestSupport.dispatcher(), service, mock(EngineGrpcClient.class));
            queue = new QueuedRequestScheduler(config, router, batches, mock(DecodeCapacityAcquirer.class),
                    runtime, availability, SchedulingTestConfig.capacity(config), org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class), clock);
            runtime.registerScheduler(queue);
            queue.start();
        }

        RequestContext context(long id) { return RequestProtocolTestSupport.context(config, id); }

        @Override public void close() { runtime.shutdown(); }
    }
}
