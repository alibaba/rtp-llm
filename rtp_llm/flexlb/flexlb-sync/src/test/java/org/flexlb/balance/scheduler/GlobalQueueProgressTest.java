package org.flexlb.balance.scheduler;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrefillQueueSheddingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongConsumer;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks queue-to-decision ordering and progress across asynchronous planners.
 */
class GlobalQueueProgressTest {
    @Test
    void queueAdmissionEventPrecedesDecisionOnPlannerThread() throws Exception {
        FlexlbTrace.configure(OpenTelemetry.noop(), "test");
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            var span = mock(Span.class);
            when(span.isRecording()).thenReturn(true);
            when(span.storeInContext(any(Context.class))).thenCallRealMethod();
            BalanceContext context = RequestLifecycleTestSupport.context(f.config, 90L);
            context.setTraceContext(Context.root().with(span));
            f.groups.put(90L, "g1");
            f.scheduler.submit(context);
            verify(span, timeout(1000)).addEvent(eq("flexlb.decision.started"), any(Attributes.class), anyLong(), eq(TimeUnit.NANOSECONDS));
            var order = org.mockito.Mockito.inOrder(span);
            order.verify(span).addEvent(eq("flexlb.queue.enqueued"), any(Attributes.class), anyLong(), eq(TimeUnit.NANOSECONDS));
            order.verify(span).addEvent(eq("flexlb.decision.started"), any(Attributes.class), anyLong(), eq(TimeUnit.NANOSECONDS));
        } finally {
            FlexlbTrace.configure(null, "");
        }
    }

    @Test
    void completedBacklogDoesNotDelayRefillingAReleasedSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch firstPlanning = new CountDownLatch(1);
            CountDownLatch finishFirstPlanning = new CountDownLatch(1);
            CountDownLatch firstCommitting = new CountDownLatch(1);
            CountDownLatch secondCommitting = new CountDownLatch(1);
            CountDownLatch finishFirst = new CountDownLatch(1);
            CountDownLatch finishSecond = new CountDownLatch(1);
            CountDownLatch thirdPlanning = new CountDownLatch(1);
            f.onAdmission = id -> {
                if (id == 1L) {
                    firstCommitting.countDown();
                    await(finishFirst);
                } else if (id == 2L) {
                    secondCommitting.countDown();
                    await(finishSecond);
                }
            };
            f.onSelection = id -> {
                if (id == 1L) {
                    firstPlanning.countDown();
                    await(finishFirstPlanning);
                } else if (id == 2L) {
                    await(firstCommitting);
                } else if (id == 3L) {
                    thirdPlanning.countDown();
                }
            };
            try {
                f.submit(1, "b");
                await(firstPlanning);
                f.submit(2, "c");
                awaitCondition(() -> f.selected.contains(2L));
                finishFirstPlanning.countDown();
                await(firstCommitting);
                awaitCondition(f::hasBufferedPlan);
                f.submit(3, "d");
                finishFirst.countDown();
                await(secondCommitting);
                assertTrue(thirdPlanning.await(1, TimeUnit.SECONDS),
                        "R3 must reuse R1's slot before the buffered R2 commit finishes");
            } finally {
                finishFirstPlanning.countDown();
                finishFirst.countDown();
                finishSecond.countDown();
            }
        }
    }

    @Test
    void readyResultBatchCommitsBeforeOneSharedBackfillRound() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, 3)) {
            CountDownLatch firstCommitting = new CountDownLatch(1);
            CountDownLatch finishFirst = new CountDownLatch(1);
            // Counted when new planning work was claimed between the commits
            // of one buffered batch instead of after the whole batch.
            var claimedDuringBatch = new AtomicInteger();
            f.onAdmission = id -> {
                if (id == 1L) {
                    firstCommitting.countDown();
                    await(finishFirst);
                } else if (id == 2L || id == 3L) {
                    // Runs on the decision thread: inFlight only changes
                    // there, so this observes the batch/claim interleaving
                    // without racing planner-thread selections.
                    if (f.inFlightRequestIds().contains(5L)) {
                        claimedDuringBatch.incrementAndGet();
                    }
                }
            };
            f.onSelection = id -> {
                if (id == 2L || id == 3L) {
                    // R2 and R3 publish only while the decision thread is
                    // committing R1, so their results buffer together.
                    await(firstCommitting);
                }
            };
            try {
                f.submit(1, "b");
                f.submit(2, "b");
                f.submit(3, "b");
                await(firstCommitting);
                awaitCondition(() -> f.bufferedPlanCount() == 2);
                // R4 fills the slot R1 releases mid-round; R5 may only be
                // claimed after the buffered R2 and R3 are both committed.
                f.submit(4, "b");
                f.submit(5, "b");
                finishFirst.countDown();
                awaitCondition(() -> f.admitted.size() == 5);
                assertEquals(0, claimedDuringBatch.get(),
                        "new work must not be claimed between the commits of"
                                + " one buffered result batch");
            } finally {
                finishFirst.countDown();
            }
        }
    }

    @Test
    void slowPlannerDoesNotBlockOtherResultsOrRefillingItsPeersSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            f.onSelection = id -> {
                if (id == 1L) {
                    started.countDown();
                    await(finish);
                }
            };
            try {
                f.submit(1, "b");
                await(started);
                f.submit(2, "c");
                awaitCondition(() -> f.admitted.contains(2L));
                // Two planner slots: R3 must reuse R2's slot while R1 is still held.
                f.submit(3, "d");
                awaitCondition(() -> f.admitted.contains(3L));
                assertEquals(List.of(2L, 3L), f.admissionOrder);
                finish.countDown();
                awaitCondition(() -> f.admitted.contains(1L));
                assertEquals(List.of(2L, 3L, 1L), f.admissionOrder);
            } finally {
                finish.countDown();
            }
        }
    }

    @Test
    void awakenedOlderRequestUsesFreeSlotWithoutInvalidatingRunningPlan() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            AtomicInteger newerSelections = new AtomicInteger();
            f.onSelection = id -> {
                if (id == 2L) {
                    newerSelections.incrementAndGet();
                    started.countDown();
                    await(finish);
                }
            };
            try {
                f.submit(1, "a");
                f.submit(2, "b");
                await(started);
                f.aSlots.set(1);
                f.release("a");
                awaitCondition(() -> f.admitted.contains(1L));
                assertFalse(f.admitted.contains(2L));
                finish.countDown();
                awaitCondition(() -> f.admitted.size() == 2);
                assertEquals(1, newerSelections.get(), "a wakeup must not discard running work");
            } finally {
                finish.countDown();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cancelledPlannerKeepsItsSlotUntilItsResourcesAreClosed(boolean costOrder) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, false, costOrder)) {
            CountDownLatch started = new CountDownLatch(2);
            CountDownLatch finish = new CountDownLatch(1);
            CountDownLatch thirdStarted = new CountDownLatch(1);
            f.onSelection = id -> {
                if (id <= 2L) {
                    started.countDown();
                    await(finish);
                } else {
                    thirdStarted.countDown();
                }
            };
            try {
                f.submit(1, "b");
                f.submit(2, "c");
                await(started);
                f.requests.get(1L).cancel(false);
                f.submit(3, "d");
                assertFalse(thirdStarted.await(100, TimeUnit.MILLISECONDS),
                        "cancellation must not allow unbounded outstanding planners");
                finish.countDown();
                awaitCondition(() -> f.admitted.contains(3L));
                assertFalse(f.admitted.contains(1L));
                verify(f.routes.get(1L), timeout(1000)).close();
                verify(f.mutations.get(1L), timeout(1000)).close();
            } finally {
                finish.countDown();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void shutdownClosesLatePlanWithoutAdmittingIt(boolean costOrder) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, false, costOrder)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            f.onSelection = id -> {
                started.countDown();
                await(finish);
            };
            try {
                f.submit(1, "b");
                await(started);
                Thread closer = new Thread(f.scheduler::closePlacement);
                closer.start();
                awaitCondition(() -> f.scheduler.getQueuedRequestCount() == 0);
                finish.countDown();
                awaitCondition(() -> f.routes.containsKey(1L));
                verify(f.routes.get(1L), timeout(1000)).close();
                verify(f.mutations.get(1L), timeout(1000)).close();
                assertTrue(f.admitted.isEmpty());
                closer.join(2000);
                assertFalse(closer.isAlive());
            } finally {
                finish.countDown();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void freeSlotGoesToHighestPriorityThenFifoWithoutInterruptingRunningPlans(boolean costOrder) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true, costOrder)) {
            CountDownLatch started = new CountDownLatch(2);
            CountDownLatch finishFirst = new CountDownLatch(1);
            CountDownLatch finishSecond = new CountDownLatch(1);
            f.onSelection = id -> {
                if (id <= 2) {
                    started.countDown();
                    await(id == 1 ? finishFirst : finishSecond);
                }
            };
            try {
                f.submit(1, "b", 10);
                f.submit(2, "b", 10);
                await(started);
                f.submit(3, "b", 10);
                f.submit(4, "b", 90);
                f.submit(5, "b", 90);
                finishFirst.countDown();
                awaitCondition(() -> f.admitted.size() == 4);
                assertEquals(List.of(1L, 4L, 5L, 3L), f.admissionOrder);
                assertFalse(f.admitted.contains(2L));
            } finally {
                finishFirst.countDown();
                finishSecond.countDown();
            }
        }
    }

    @Test
    void requestSpecificDecodeFailureDoesNotBlockSmallerRequestsInTheSameGroup() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a");
            awaitCondition(() -> f.selected.contains(1L));
            f.submit(2, "a");
            awaitCondition(() -> f.admitted.contains(2L));
            assertFalse(f.admitted.contains(1L));
        }
    }

    @Test
    void queuedPreemptionIsNotBlockedByOrdinarySeatExhaustion() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.submit(1, "a");
            awaitCondition(() -> f.admitted.contains(1L));
        }
    }

    @Test
    void admissionFailureDoesNotFenceSmallerRequestOnTheSameWorker() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            f.admissionBlocked.add(1L);
            f.submit(1, "a");
            f.submit(2, "a");
            awaitCondition(() -> f.admitted.contains(2L));
            assertFalse(f.admitted.contains(1L));
            assertEquals(1, f.scheduler.getQueuedRequestCount());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void planningOneRequestDoesNotHoldSparePlanner(boolean enabled) throws Exception {
        var release = new CountDownLatch(1);
        var firstStarted = new CountDownLatch(1);
        var secondStarted = new CountDownLatch(1);
        try (Fixture f = new Fixture(RoleType.PREFILL, false, enabled)) {
            f.aSlots.set(1);
            f.onSelection = id -> {
                if (id == 1801L) {
                    firstStarted.countDown();
                    await(release);
                } else {
                    secondStarted.countDown();
                }
            };
            Object queue = ReflectionTestUtils.getField(f.scheduler, "globalQueue");
            var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
            lock.lock();
            try {
                f.submit(1801L, "a");
                f.submit(1802L, "a");
            } finally {
                lock.unlock();
            }
            try {
                assertTrue(firstStarted.await(2, TimeUnit.SECONDS));
                assertTrue(secondStarted.await(2, TimeUnit.SECONDS),
                        "a running request must not hold another planner slot");
            } finally {
                release.countDown();
            }
            awaitCondition(() -> f.admitted.size() == 2);
        } finally {
            release.countDown();
        }
    }

    @Test
    void cancelledGlobalWaiterDoesNotBlockLaterWorkOrStartAfterCancellation() throws Exception {
        var release = new CountDownLatch(1);
        var firstStarted = new CountDownLatch(1);
        var cancelledSelections = new AtomicInteger();
        try (Fixture f = new Fixture(RoleType.PREFILL, false, true)) {
            f.aSlots.set(1);
            f.onSelection = id -> {
                if (id == 1801L) {
                    firstStarted.countDown();
                    await(release);
                }
                if (id == 1802L) {
                    cancelledSelections.incrementAndGet();
                }
            };
            f.submit(1801L, "a");
            assertTrue(firstStarted.await(2, TimeUnit.SECONDS));
            var queue = ReflectionTestUtils.getField(f.scheduler, "globalQueue");
            var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
            lock.lock();
            try {
                f.submit(1802L, "a");
                f.requests.get(1802L).cancel(false);
            } finally {
                lock.unlock();
            }
            f.submit(1803L, "a");
            release.countDown();
            awaitCondition(() -> f.admitted.contains(1803L));
            assertEquals(0, cancelledSelections.get());
            assertFalse(f.admitted.contains(1802L));
            awaitCondition(() -> f.scheduler.getQueuedRequestCount() == 0);
        } finally {
            release.countDown();
        }
    }

    @Test
    void slowSheddingRoundDoesNotBlockDispatchOrReviveAnEvictedRequest() throws Exception {
        var sheddingStarted = new CountDownLatch(1);
        var finishShedding = new CountDownLatch(1);
        var config = new PrefillQueueSheddingConfig();
        config.setExpectedWaitThresholdMs(1L);
        config.setMaxWaitMs(1000L);
        try (Fixture f = new Fixture(RoleType.PREFILL, false, true, config)) {
            when(f.router.estimateQueuedWork(any(), eq("a")))
                    .thenReturn(new PrefillQueueShedding.Estimate(1000L, 1L, 5L, 1));
            when(f.router.prefillPoolSnapshot("a")).thenAnswer(call -> {
                sheddingStarted.countDown();
                await(finishShedding);
                return new CostBasedPrefillStrategy.PrefillPoolSnapshot(1, 0, 1000L);
            });
            when(f.router.isPrefillClusterOverloaded(eq("a"), anyLong())).thenReturn(true);
            try {
                f.submit(1850L, "a");
                assertTrue(sheddingStarted.await(2, TimeUnit.SECONDS));
                f.submit(1851L, "b");
                awaitCondition(() -> f.admitted.contains(1851L));
                finishShedding.countDown();
                awaitCondition(() -> f.scheduler.getQueuedRequestCount() == 0);
                f.aSlots.incrementAndGet();
                f.release("a");
                f.submit(1852L, "b");
                awaitCondition(() -> f.admitted.contains(1852L));
                assertFalse(f.admitted.contains(1850L));
            } finally {
                finishShedding.countDown();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void globalRetriesPreservePriorityAndArrivalOrderWithWorkerCostOrder(boolean sameWorker) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true, true, null, 1)) {
            var ids = List.of(1803L, 1802L, 1801L);
            f.admissionBlocked.addAll(ids);
            for (int i = 0; i < ids.size(); i++) {
                long id = ids.get(i);
                f.submit(id, sameWorker ? "a" : "g" + id, i == 2 ? 90 : 50);
                RequestLifecycleTestSupport.awaitGlobalCapacityWaiters(f.scheduler, i + 1);
            }
            // Release all groups before the only planner can claim a retry.
            Object queue = ReflectionTestUtils.getField(f.scheduler, "globalQueue");
            var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
            lock.lock();
            try {
                f.admissionBlocked.clear();
                if (sameWorker) {
                    f.release("a");
                } else {
                    ids.reversed().forEach(id -> f.release("g" + id));
                }
            } finally {
                lock.unlock();
            }
            awaitCondition(() -> f.admissionOrder.size() == 3);
            assertEquals(List.of(1801L, 1803L, 1802L), f.admissionOrder);
        }
    }

    @Test
    void sheddingScanHasACadenceEvenWithRepeatedSchedulerWakeups() throws Exception {
        var scanTimes = new CopyOnWriteArrayList<Long>();
        var firstScan = new CountDownLatch(1);
        try (Fixture f = new Fixture(RoleType.PREFILL, false, false, new PrefillQueueSheddingConfig())) {
            when(f.router.estimateQueuedWork(any(), eq("a")))
                    .thenReturn(new PrefillQueueShedding.Estimate(1024L, 0L, 5L, 1));
            when(f.router.prefillPoolSnapshot("a")).thenAnswer(call -> {
                scanTimes.add(System.nanoTime());
                firstScan.countDown();
                return null;
            });
            f.submit(2201L, "a");
            assertTrue(firstScan.await(2, TimeUnit.SECONDS));
            for (long id = 2202L; id < 2222L; id++) {
                f.submit(id, "b");
            }
            awaitCondition(() -> f.admitted.size() == 20);
            awaitCondition(() -> scanTimes.size() >= 2);
            for (int index = 1, count = scanTimes.size(); index < count; index++) {
                long gap = scanTimes.get(index) - scanTimes.get(index - 1);
                // The snapshot is read just after the scan starts, outside the queue lock.
                assertTrue(gap >= TimeUnit.MILLISECONDS.toNanos(75L),
                        "scheduler wakeups must not trigger extra shedding scans; gap ns=" + gap);
            }
        }
    }

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config;
        private final PlacementAvailability availability = new PlacementAvailability();
        private final AtomicInteger aSlots = new AtomicInteger();
        private final Map<Long, RouteAdmission> routes = new ConcurrentHashMap<>();
        private final Map<Long, AdmissionHandle> mutations = new ConcurrentHashMap<>();
        private final Map<Long, String> groups = new ConcurrentHashMap<>();
        private final Map<Long, CompletableFuture<Response>> requests = new ConcurrentHashMap<>();
        private final Set<Long> selected = ConcurrentHashMap.newKeySet();
        private final Set<Long> admitted = ConcurrentHashMap.newKeySet();
        private final List<Long> admissionOrder = new CopyOnWriteArrayList<>();
        private final Set<Long> admissionBlocked = ConcurrentHashMap.newKeySet();
        private final Set<Long> decodeBlocked = ConcurrentHashMap.newKeySet();
        private volatile LongConsumer onAdmission = ignored -> { };
        private volatile LongConsumer onSelection = ignored -> { };
        private final RequestScheduler scheduler;
        private final DefaultRouter router;
        private final RoleType role;

        private static FlexlbConfig plannerConfig(int plannerThreads) {
            FlexlbConfig config = spy(SchedulingTestConfig.batchConfig());
            var runtime = spy(config.getInternalRuntime());
            when(runtime.getQueuePlannerThreads()).thenReturn(plannerThreads);
            when(config.getInternalRuntime()).thenReturn(runtime);
            return config;
        }

        private Fixture(RoleType role) {
            this(role, false);
        }

        private Fixture(RoleType role, int plannerThreads) {
            this(role, false, false, null, plannerThreads);
        }

        private Fixture(RoleType role, boolean preempt) {
            this(role, preempt, false, null, 2);
        }

        private Fixture(RoleType role, boolean preempt, boolean prefillCostOrder) {
            this(role, preempt, prefillCostOrder, null, 2);
        }

        private Fixture(RoleType role, boolean preempt, boolean prefillCostOrder,
                PrefillQueueSheddingConfig sheddingConfig) {
            this(role, preempt, prefillCostOrder, sheddingConfig, 2);
        }

        private Fixture(RoleType role, boolean preempt, boolean prefillCostOrder,
                PrefillQueueSheddingConfig sheddingConfig, int plannerThreads) {
            this.role = role;
            this.config = plannerConfig(plannerThreads);
            SchedulingTestConfig.useFifoQueue(config);
            config.queueScheduler().setPrefillQueueShedding(sheddingConfig);
            if (preempt) {
                SchedulingTestConfig.usePriorityQueue(config);
                SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
            }
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
            if (prefillCostOrder) {
                SchedulingTestConfig.useSingleDecision(config);
                config.getDispatcher().setPrefillCostOrder(true);
            }
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            DefaultRouter router = mock(DefaultRouter.class);
            when(router.resolvePolicyGroup(any())).thenAnswer(i -> groups.get(((BalanceContext) i.getArgument(0)).getRequestId()));
            EndpointRegistry endpoints = mock(EndpointRegistry.class);
            PrefillRoutingEntry a = endpoint("a");
            RequestRegistry lifecycle = mock(RequestRegistry.class);
            when(lifecycle.register(any())).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                CompletableFuture<Response> future = new CompletableFuture<>();
                requests.put(context.getRequestId(), future);
                return future;
            });
            when(lifecycle.claimAdmissionHandle(anyLong(), any())).thenAnswer(i -> {
                AdmissionHandle mutation = mock(AdmissionHandle.class);
                mutations.put(i.getArgument(0), mutation);
                return mutation;
            });
            when(router.select(any(), nullable(String.class))).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                long id = context.getRequestId();
                selected.add(id);
                onSelection.accept(id);
                if ("a".equals(groups.get(id)) && aSlots.get() == 0 && !preempt) {
                    return PlacementResult.blocked(new PlacementKey(role, "a"));
                }
                if (decodeBlocked.contains(id)) {
                    return PlacementResult.blocked(new PlacementKey(RoleType.DECODE, groups.get(id)));
                }
                RouteAdmission route = mock(RouteAdmission.class);
                routes.put(id, route);
                ScheduledRequest item = mock(ScheduledRequest.class);
                ServerStatus status = new ServerStatus();
                status.setRole(role);
                when(item.prefill()).thenReturn(status);
                when(item.prefillEp()).thenReturn(a.endpoint());
                when(route.tryEnqueue(any(), any(), any())).thenAnswer(commit -> {
                    onAdmission.accept(id);
                    if (admissionBlocked.contains(id)) {
                        return PlacementResult.blocked(PlacementKey.exact(role, groups.get(id), groups.get(id) + ":8000"));
                    }
                    admissionOrder.add(id);
                    admitted.add(id);
                    return PlacementResult.success(item);
                });
                return PlacementResult.success(route);
            });
            scheduler = new RequestScheduler(service, router, endpoints, mock(BatchSchedulerReporter.class),
                    mock(EvictionManager.class), lifecycle, availability);
            this.router = router;
        }

        private void submit(long id, String group) {
            submit(id, group, 50);
        }

        private void submit(long id, String group, int priority) {
            groups.put(id, group);
            BalanceContext context = RequestLifecycleTestSupport.context(config, id);
            context.setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(
                    priority, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
            scheduler.submit(context);
        }

        private boolean hasBufferedPlan() {
            return bufferedPlanCount() > 0;
        }

        private int bufferedPlanCount() {
            // Observe the handoff under its lock to make the regression independent
            // of thread timing, without adding a hook to production code.
            Object queue = ReflectionTestUtils.getField(scheduler, "globalQueue");
            var lock = (ReentrantLock)
                    ReflectionTestUtils.getField(queue, "lock");
            lock.lock();
            try {
                var completed = (Queue<?>) ReflectionTestUtils
                        .getField(queue, "completedPlans");
                return completed.size();
            } finally {
                lock.unlock();
            }
        }

        private List<Long> inFlightRequestIds() {
            // Called from the decision thread, which is the only writer of
            // inFlight, so no lock is needed to read a consistent view.
            Object queue = ReflectionTestUtils.getField(scheduler, "globalQueue");
            var inFlight = (Set<?>) ReflectionTestUtils.getField(queue, "inFlight");
            return inFlight.stream().map(GlobalQueueEntry.class::cast)
                    .map(entry -> entry.context.getRequestId()).sorted().toList();
        }

        private void release(String group) {
            availability.capacityChanged(PlacementKey.exact(role, group, group + ":8000"));
        }

        public void close() {
            requests.values().forEach(future -> future.complete(new Response()));
            scheduler.closePlacement();
        }

        private static PrefillRoutingEntry endpoint(String group) {
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            WorkerStatus status = mock(WorkerStatus.class);
            WorkerStatus.TopologySnapshot topology = mock(WorkerStatus.TopologySnapshot.class);
            when(status.topologySnapshot()).thenReturn(topology);
            when(topology.group()).thenReturn(group);
            when(endpoint.getStatus()).thenReturn(status);
            return new PrefillRoutingEntry(group + ":8000", endpoint);
        }
    }
}
