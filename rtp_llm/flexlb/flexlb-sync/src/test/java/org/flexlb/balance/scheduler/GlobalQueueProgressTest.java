package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
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
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongConsumer;
import java.util.stream.LongStream;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GlobalQueueProgressTest {
    @Test
    void countingTheSameFutureTwiceDoesNotChangeTheTotalTwice() {
        QueueCapacity capacity = new QueueCapacity();
        CompletableFuture<Response> future = new CompletableFuture<>();

        capacity.increaseTotalQueueCount(future);
        capacity.increaseTotalQueueCount(future);
        assertEquals(1L, capacity.globalQueueAndEngineSubQueueCount());
        capacity.decreaseTotalQueueCount(future);
        capacity.decreaseTotalQueueCount(future);
        assertEquals(0L, capacity.globalQueueAndEngineSubQueueCount());
    }

    @Test
    void globalQueueEntryUsesTheTotalCountOnlyForNonBatchDispatch() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        config.getScheduler().setMaxQueuedRequests(1);
        QueueCapacity capacity = new QueueCapacity();

        assertFalse(capacity.isGlobalQueueEntryLimitReached(config, 0));
        assertTrue(capacity.isGlobalQueueEntryLimitReached(config, 1));

        CompletableFuture<Response> batchRequest = new CompletableFuture<>();
        capacity.increaseTotalQueueCount(batchRequest);
        assertEquals(1L, capacity.globalQueueAndEngineSubQueueCount());
        assertFalse(capacity.isGlobalQueueEntryLimitReached(config, 0));
        capacity.decreaseTotalQueueCount(batchRequest);

        SchedulingTestConfig.useNonBatchDispatcher(config);
        CompletableFuture<Response> transferredRequest = new CompletableFuture<>();
        capacity.increaseTotalQueueCount(transferredRequest);
        assertTrue(capacity.isGlobalQueueEntryLimitReached(config, 0));
        capacity.decreaseTotalQueueCount(transferredRequest);
        assertFalse(capacity.isGlobalQueueEntryLimitReached(config, 0));
    }

    @Test
    void globalQueueTotalLimitDoesNotIncreaseWhenEndpointsGrow() throws Exception {
        try (Fixture f = new Fixture(RoleType.PDFUSION)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.config.getDispatcher().setPrefillSubqueueLimitMultiplier(1);
            f.prefillEndpointCount.set(1);
            f.decodeBlocked.add(1L);
            f.submit(1L, "b");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            f.submit(2L, "b");
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                    f.requests.get(2L).get(5, TimeUnit.SECONDS).getCode());

            f.fusionEndpointCount.set(1);
            f.submit(3L, "b");
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                    f.requests.get(3L).get(5, TimeUnit.SECONDS).getCode());
            f.requests.get(1L).complete(new Response());
            awaitCondition(() -> f.queueCapacity.globalQueueAndEngineSubQueueCount() == 0L);
            f.submit(4L, "b");
            awaitCondition(() -> f.admitted.contains(4L));
        }
    }

    @Test
    void queueBudgetUsesCurrentEndpointCountForEachRoutingShape() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        config.getDispatcher().setMaxInflightPerPrefillWorker(2);
        config.getDispatcher().setPrefillSubqueueLimitMultiplier(3);
        QueueCapacity capacity = new QueueCapacity();
        AtomicInteger prefillEndpoints = new AtomicInteger(1);
        AtomicInteger fusionEndpoints = new AtomicInteger();
        assertEquals(6L, capacity.engineSubQueueLimit(config, prefillEndpoints.get()));
        prefillEndpoints.set(2);
        assertEquals(12L, capacity.engineSubQueueLimit(config, prefillEndpoints.get()));
        prefillEndpoints.set(1);
        assertEquals(6L, capacity.engineSubQueueLimit(config, prefillEndpoints.get()));

        fusionEndpoints.set(1);
        assertEquals(6L, capacity.engineSubQueueLimit(config, fusionEndpoints.get()));
        assertEquals(6L, capacity.engineSubQueueLimit(config, prefillEndpoints.get()));
        fusionEndpoints.set(2);
        assertEquals(12L, capacity.engineSubQueueLimit(config, fusionEndpoints.get()));
        assertEquals(6L, capacity.engineSubQueueLimit(config, prefillEndpoints.get()));
        config.getScheduler().setMaxQueuedRequests(10);
        assertEquals(10L, capacity.engineSubQueueLimit(config, fusionEndpoints.get()));
        assertEquals(6L, capacity.engineSubQueueLimit(config, prefillEndpoints.get()));
    }

    @Test
    void engineSubQueueReservationsNeverExceedTheConfiguredTotalQueueLimit() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getScheduler().setMaxQueuedRequests(3);
        config.getDispatcher().setMaxInflightPerPrefillWorker(2);
        config.getDispatcher().setPrefillSubqueueLimitMultiplier(4);
        QueueCapacity capacity = new QueueCapacity();

        assertEquals(3L, capacity.engineSubQueueLimit(config, 100));
        for (int i = 0; i < 3; i++) {
            assertTrue(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 100));
        }
        assertFalse(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 100));
        assertTrue(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PDFUSION, 100));
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
        assertTrue(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 100));
    }

    @Test
    void batchDispatcherCountsEngineQueueRequestsWithoutApplyingTheSubQueueLimit() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        config.getScheduler().setMaxQueuedRequests(1);
        QueueCapacity capacity = new QueueCapacity();

        capacity.increaseEngineSubQueueCount(RoleType.PREFILL);
        capacity.increaseEngineSubQueueCount(RoleType.PREFILL);
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        assertFalse(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
        assertFalse(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
        assertTrue(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
    }

    @Test
    void releasingAnEmptyEngineSubQueueDoesNotCreateExtraCapacity() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.getScheduler().setMaxQueuedRequests(1);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getDispatcher().setPrefillSubqueueLimitMultiplier(1);
        QueueCapacity capacity = new QueueCapacity();

        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
        assertTrue(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL, false);
        assertFalse(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL, true);
        capacity.releaseEngineSubQueueSlot(RoleType.PREFILL);
        assertTrue(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
        assertFalse(capacity.tryReserveEngineSubQueueSlot(config, RoleType.PREFILL, 1));
    }

    @Test
    void transferredRequestRetainsTotalQueueSeatUntilItsRouteCompletes() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.config.getScheduler().setMaxQueuedRequests(2);
            f.config.getDispatcher().setPrefillSubqueueLimitMultiplier(1);
            f.submit(1L, "b");
            f.submit(2L, "b");
            awaitCondition(() -> f.admitted.containsAll(List.of(1L, 2L)));
            assertEquals(0, f.scheduler.getQueuedRequestCount());
            assertEquals(2L, f.queueCapacity.globalQueueAndEngineSubQueueCount());

            f.submit(3L, "b");
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(),
                    f.requests.get(3L).get(5, TimeUnit.SECONDS).getCode());
            f.requests.get(1L).complete(new Response());
            awaitCondition(() -> f.queueCapacity.globalQueueAndEngineSubQueueCount() == 1L);
            f.submit(4L, "b");
            awaitCondition(() -> f.admitted.contains(4L));
        }
    }

    @Test
    void requeueRestoresTotalQueueCountAfterDeliveryPreparationReleasedIt() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.submit(1L, "b");
            awaitCondition(() -> f.admitted.contains(1L) && f.scheduler.getQueuedRequestCount() == 0);
            var future = f.requests.get(1L);
            f.queueCapacity.decreaseTotalQueueCount(future);
            assertEquals(0L, f.queueCapacity.globalQueueAndEngineSubQueueCount());
            f.decodeBlocked.add(1L);
            var previous = mock(ScheduledRequest.class);
            when(previous.future()).thenReturn(future);
            var queue = (GlobalQueueCoordinator) ReflectionTestUtils.getField(f.scheduler, "globalQueue");

            assertTrue(queue.requeue(previous));

            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);
            assertEquals(1L, f.queueCapacity.globalQueueAndEngineSubQueueCount());
            future.complete(new Response());
            awaitCondition(() -> f.queueCapacity.globalQueueAndEngineSubQueueCount() == 0L);
        }
    }

    @Test
    void globalQueueWaitIsRecordedOnlyWhenTheRequestLeavesTheQueue() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.submit(1L, "a");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);
            GlobalQueueCoordinator queue = (GlobalQueueCoordinator) ReflectionTestUtils.getField(
                    f.scheduler, "globalQueue");
            assertTrue(queue.oldestWaitingAgeMs() >= 0L);
            org.mockito.Mockito.verify(f.reporter, org.mockito.Mockito.never())
                    .reportGlobalQueueResidenceMs(anyLong());

            f.aSlots.set(1);
            f.release("a");
            awaitCondition(() -> f.admitted.contains(1L));
            verify(f.reporter, timeout(5_000)).reportGlobalQueueResidenceMs(anyLong());
            assertEquals(0L, queue.oldestWaitingAgeMs());
        }
    }

    @Test
    void globalPreemptionUsesTheTotalQueueLimit() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(2);
            f.config.getDispatcher().setPrefillSubqueueLimitMultiplier(1);
            f.decodeBlocked.addAll(Set.of(1L, 2L));
            f.submit(1L, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);
            f.submit(2L, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 2);
            assertFalse(f.requests.get(1L).isDone());
            assertFalse(f.requests.get(2L).isDone());

            f.submit(3L, "a", 90);
            awaitCondition(() -> f.requests.get(2L).isDone());
            assertEquals(8435, f.requests.get(2L).join().getCode());
            assertFalse(f.requests.get(1L).isDone());
        }
    }

    @Test
    void completedArrivalDoesNotDisplaceWaitingRequest() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            f.groups.put(2L, "a");
            BalanceContext incoming = RequestLifecycleTestSupport.context(f.config, 2L);
            CompletableFuture<Response> completed = CompletableFuture.completedFuture(new Response());
            GlobalQueueCoordinator queue = (GlobalQueueCoordinator) ReflectionTestUtils.getField(f.scheduler, "globalQueue");
            assertEquals(GlobalQueueCoordinator.OfferResult.COMPLETED, queue.offer(incoming, completed, 90));
            assertEquals(1, f.scheduler.getQueuedRequestCount());
            assertFalse(f.requests.get(1L).isDone());
        }
    }

    @Test
    void concurrentHigherPriorityArrivalsReplaceOnlyOneQueuedRequest() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            CountDownLatch finishPlanning = new CountDownLatch(1);
            f.onSelection = id -> {
                if (id > 1) {
                    await(finishPlanning);
                }
            };
            try {
                LongStream.rangeClosed(2, 20).parallel().forEach(id -> f.submit(id, "a", 90));
                awaitCondition(() -> f.requests.get(1L).isDone());

                assertEquals(8435, f.requests.get(1L).join().getCode());
                assertEquals(1, f.scheduler.getQueuedRequestCount());
                assertEquals(18, f.requests.values().stream()
                        .filter(CompletableFuture::isDone)
                        .map(CompletableFuture::join)
                        .filter(response -> response.getCode()
                                == StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED.getErrorCode())
                        .count());
            } finally {
                finishPlanning.countDown();
            }
        }
    }

    @Test
    void higherPriorityRequestReplacesNewestQueuedRequestAtLowestPriority() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(2);
            f.decodeBlocked.addAll(Set.of(1L, 2L));
            f.submit(1, "a", 20);
            f.submit(2, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 2);

            f.submit(3, "a", 90);
            awaitCondition(() -> f.requests.get(2L).isDone());
            assertEquals(8435, f.requests.get(2L).join().getCode());
            assertFalse(f.requests.get(1L).isDone());
            awaitCondition(() -> f.admitted.contains(3L));
            assertEquals(1, f.scheduler.getQueuedRequestCount());
        }
    }

    @ParameterizedTest
    @CsvSource({"40,50", "49,50", "60,61", "99,100"})
    void eligibleRequestCanReplaceAnyLowerPriority(int waitingPriority, int incomingPriority) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a", waitingPriority);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            f.submit(2, "a", incomingPriority);
            awaitCondition(() -> f.requests.get(1L).isDone());
            assertEquals(8435, f.requests.get(1L).join().getCode());
            awaitCondition(() -> f.admitted.contains(2L));
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {50, 90})
    void fullPriorityQueueRejectsWithoutLowerPriorityAndReplacesWhenPossible(int highPriority) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a", highPriority);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            f.submit(2, "a", 40);
            Response low = f.requests.get(2L).join();
            assertEquals(8433, low.getCode());
            assertEquals("Too many requests. Your requests are being throttled "
                    + "due to system capacity limits. Please try again later.", low.getErrorMessage());

            f.submit(3, "a", highPriority);
            Response equal = f.requests.get(3L).join();
            assertEquals(8434, equal.getCode());
            assertEquals("Too many requests. Your requests are being throttled "
                    + "due to system capacity limits. Please try again later.",
                    equal.getErrorMessage());

            f.submit(4, "a", highPriority + 5);
            Response displaced = f.requests.get(1L).get(5, TimeUnit.SECONDS);
            assertEquals(8435, displaced.getCode());
            assertEquals("Too many requests. Your requests are being throttled "
                    + "due to system capacity limits. Please try again later.", displaced.getErrorMessage());
            awaitCondition(() -> f.admitted.contains(4L));
        }
    }

    @Test
    void requestBelowThresholdDoesNotReplaceEvenLowerQueuedPriority() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            f.submit(2, "a", 40);
            Response rejected = f.requests.get(2L).join();
            assertEquals(8433, rejected.getCode());
            assertEquals("Too many requests. Your requests are being throttled "
                    + "due to system capacity limits. Please try again later.", rejected.getErrorMessage());
            assertFalse(f.requests.get(1L).isDone());
            assertEquals(1, f.scheduler.getQueuedRequestCount());
        }
    }

    @Test
    void updatedThresholdAppliesToSubsequentArrivals() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a", 40);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            FlexlbConfig updated = SchedulingTestConfig.batchConfig();
            SchedulingTestConfig.usePriorityQueue(updated);
            updated.getScheduler().setMaxQueuedRequests(1);
            updated.priorityOrdering().setHighPriorityThreshold(70);
            f.currentConfig.set(updated);

            f.submit(2, "a", 60);
            assertEquals(8433, f.requests.get(2L).join().getCode());
            assertFalse(f.requests.get(1L).isDone());

            f.submit(3, "a", 70);
            assertEquals(8435, f.requests.get(1L).get(5, TimeUnit.SECONDS).getCode());
            awaitCondition(() -> f.admitted.contains(3L));
        }
    }

    @Test
    void fullQueueDoesNotReplaceLowerPriorityRequestDuringPlanning() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            CountDownLatch planning = new CountDownLatch(1);
            CountDownLatch finishPlanning = new CountDownLatch(1);
            f.onSelection = id -> {
                if (id == 1) {
                    planning.countDown();
                    await(finishPlanning);
                }
            };
            try {
                f.submit(1, "a", 20);
                await(planning);
                f.submit(2, "a", 90);
                assertEquals(StrategyErrorType.QUEUE_FULL_HIGH_PRIORITY_REJECTED.getErrorCode(),
                        f.requests.get(2L).join().getCode());
                assertFalse(f.requests.get(1L).isDone());
            } finally {
                finishPlanning.countDown();
            }
        }
    }

    @Test
    void fullQueueReplacesLowestEligiblePriorityEvenWhenOthersAreLowerThanIncoming() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.config.getScheduler().setMaxQueuedRequests(3);
            f.decodeBlocked.addAll(Set.of(1L, 2L, 3L));
            f.submit(1, "a", 30);
            f.submit(2, "a", 60);
            f.submit(3, "a", 10);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 3);

            f.submit(4, "a", 70);
            Response displaced = f.requests.get(3L).get(5, TimeUnit.SECONDS);
            assertEquals(8435, displaced.getCode());
            assertFalse(f.requests.get(1L).isDone());
            assertFalse(f.requests.get(2L).isDone());
            awaitCondition(() -> f.admitted.contains(4L));
        }
    }

    @Test
    void fifoQueueKeepsItsCapacityLimitRegardlessOfNumericPriority() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.submit(1, "a", 20);
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);

            f.submit(2, "a", 90);
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), f.requests.get(2L).join().getCode());
            assertFalse(f.requests.get(1L).isDone());
        }
    }

    @Test
    void concurrentArrivalsNeverExceedGlobalQueueCapacity() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.config.getScheduler().setMaxQueuedRequests(2);
            LongStream.rangeClosed(1, 20).parallel().forEach(id -> f.submit(id, "a"));

            assertEquals(2, f.scheduler.getQueuedRequestCount());
            assertEquals(18, f.requests.values().stream()
                    .filter(CompletableFuture::isDone)
                    .map(CompletableFuture::join)
                    .filter(response -> response.getCode() == StrategyErrorType.QUEUE_FULL.getErrorCode())
                    .count());
        }
    }

    @Test
    void globalQueueRejectsAtCapacityAndAcceptsAfterSpaceIsFreed() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.config.getScheduler().setMaxQueuedRequests(1);
            f.submit(1, "a");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);
            assertEquals(1, f.scheduler.getQueuedRequestCount());

            f.submit(2, "a");
            Response rejected = f.requests.get(2L).join();
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), rejected.getCode());
            assertEquals(1, f.scheduler.getQueuedRequestCount());
            assertFalse(f.selected.contains(2L));

            f.requests.get(1L).complete(new Response());
            awaitCondition(() -> f.scheduler.getQueuedRequestCount() == 0);
            f.submit(3, "a");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);
            assertFalse(f.requests.get(3L).isDone());
        }
    }

    @Test
    void globalQueueUsesUpdatedCapacityForNewArrivals() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.config.getScheduler().setMaxQueuedRequests(2);
            f.submit(1, "a");
            f.submit(2, "a");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 2);
            GlobalQueueCoordinator queue = (GlobalQueueCoordinator) ReflectionTestUtils.getField(f.scheduler, "globalQueue");
            assertEquals(2, queue.size());
            assertEquals(2, queue.maxQueuedRequests());

            FlexlbConfig reduced = SchedulingTestConfig.batchConfig();
            reduced.getScheduler().setMaxQueuedRequests(1);
            f.currentConfig.set(reduced);
            assertEquals(1, queue.maxQueuedRequests());
            f.submit(3, "a");
            assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), f.requests.get(3L).join().getCode());
            assertEquals(2, f.scheduler.getQueuedRequestCount());

            FlexlbConfig increased = SchedulingTestConfig.batchConfig();
            increased.getScheduler().setMaxQueuedRequests(3);
            f.currentConfig.set(increased);
            f.submit(4, "a");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 3);
            assertFalse(f.requests.get(4L).isDone());
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

    @Test
    void cancelledPlannerKeepsItsSlotUntilItsResourcesAreClosed() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
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

    @Test
    void shutdownClosesLatePlanWithoutAdmittingIt() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
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

    @Test
    void freeSlotGoesToHighestPriorityThenFifoWithoutInterruptingRunningPlans() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
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
    void decodeReservationFailureReportsEachAttemptBeforePrefillEnqueue() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            f.decodeReservationBlocked.add(1L);
            f.submit(1, "a");
            awaitCondition(() -> f.scheduler.getBlockedRequestCount() == 1);
            verify(f.reporter, timeout(5_000)).reportDecodeReservationBlocked("decode:8001");
            assertFalse(f.admitted.contains(1L));
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

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = twoPlannerConfig();
        private final AtomicReference<FlexlbConfig> currentConfig = new AtomicReference<>(config);
        private final PlacementAvailability availability;
        private final QueueCapacity queueCapacity;
        private final java.util.Set<Long> decodeReservationBlocked = ConcurrentHashMap.newKeySet();
        private final BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        private final AtomicInteger prefillEndpointCount = new AtomicInteger();
        private final AtomicInteger fusionEndpointCount = new AtomicInteger();
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
        private final RoleType role;

        private static FlexlbConfig twoPlannerConfig() {
            FlexlbConfig config = spy(SchedulingTestConfig.batchConfig());
            var runtime = spy(config.getInternalRuntime());
            when(runtime.getQueuePlannerThreads()).thenReturn(2);
            when(config.getInternalRuntime()).thenReturn(runtime);
            return config;
        }

        private Fixture(RoleType role) {
            this(role, false);
        }

        private Fixture(RoleType role, boolean preempt) {
            this.role = role;
            availability = new PlacementAvailability();
            queueCapacity = new QueueCapacity();
            SchedulingTestConfig.useFifoQueue(config);
            if (preempt) {
                SchedulingTestConfig.usePriorityQueue(config);
                SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
            }
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenAnswer(ignored -> currentConfig.get());
            DefaultRouter router = mock(DefaultRouter.class);
            when(router.resolvePolicyGroup(any())).thenAnswer(i -> groups.get(((BalanceContext) i.getArgument(0)).getRequestId()));
            EndpointRegistry endpoints = mock(EndpointRegistry.class);
            when(endpoints.getEndpointCount(RoleType.PREFILL))
                    .thenAnswer(ignored -> prefillEndpointCount.get());
            when(endpoints.getEndpointCount(RoleType.PDFUSION))
                    .thenAnswer(ignored -> fusionEndpointCount.get());
            PrefillRoutingEntry a = endpoint("a");
            RequestRegistry lifecycle = mock(RequestRegistry.class);
            when(lifecycle.publishDecisionResponseAsync(anyString(), any(), any())).thenAnswer(call -> {
                CompletableFuture<Response> victim = call.getArgument(1);
                return victim.complete(call.getArgument(2));
            });
            when(lifecycle.register(any())).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                CompletableFuture<Response> future = new CompletableFuture<>();
                requests.put(Long.parseLong(context.getRequestId()), future);
                return future;
            });
            when(lifecycle.claimAdmissionHandle(anyString(), any())).thenAnswer(i -> {
                AdmissionHandle mutation = mock(AdmissionHandle.class);
                mutations.put(Long.parseLong(i.getArgument(0)), mutation);
                return mutation;
            });
            when(router.select(any(), nullable(String.class))).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                long id = Long.parseLong(context.getRequestId());
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
                    if (decodeReservationBlocked.contains(id)) {
                        var decode = mock(org.flexlb.balance.endpoint.DecodeEndpoint.class);
                        when(decode.ipPort()).thenReturn("decode:8001");
                        when(route.decodeEndpoint()).thenReturn(decode);
                        return PlacementResult.blocked(PlacementKey.exact(RoleType.DECODE, "a", "decode:8001"));
                    }
                    if (admissionBlocked.contains(id)) {
                        return PlacementResult.blocked(PlacementKey.exact(role, "a", "a:8000"));
                    }
                    admissionOrder.add(id);
                    admitted.add(id);
                    return PlacementResult.success(item);
                });
                return PlacementResult.success(route);
            });
            scheduler = new RequestScheduler(service, router, endpoints, reporter,
                    mock(EvictionManager.class), lifecycle, availability, queueCapacity);
        }

        private void submit(long id, String group) {
            submit(id, group, 50);
        }

        private void submit(long id, String group, int priority) {
            groups.put(id, group);
            BalanceContext context = RequestLifecycleTestSupport.context(currentConfig.get(), id);
            context.setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(
                    priority, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
            scheduler.submit(context);
        }

        private boolean hasBufferedPlan() {
            // Observe the handoff under its lock to make the regression independent
            // of thread timing, without adding a hook to production code.
            Object queue = ReflectionTestUtils.getField(scheduler, "globalQueue");
            var lock = (ReentrantLock)
                    ReflectionTestUtils.getField(queue, "lock");
            lock.lock();
            try {
                var completed = (Queue<?>) ReflectionTestUtils
                        .getField(queue, "completedPlans");
                return !completed.isEmpty();
            } finally {
                lock.unlock();
            }
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
