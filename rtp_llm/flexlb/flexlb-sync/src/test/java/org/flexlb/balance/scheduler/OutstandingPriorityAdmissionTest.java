package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OutstandingPriorityAdmissionTest {
    private FlexlbConfig config;
    private RequestRegistry registry;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        ConfigService configs = mock(ConfigService.class);
        when(configs.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(configs, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class), mock(EngineCancelChannel.class));
    }

    @AfterEach
    void tearDown() {
        registry.closeAdmissionAndAwaitMutations();
        registry.closeOutstandingAndTerminalize();
        registry.closeExpiration();
        registry.closePublisher();
    }

    @Test
    void higherPriorityTransfersTheLowestPriorityPermitWithoutIncreasingCapacity() throws Exception {
        var low = registry.register(context(1, 10), 2);
        var medium = registry.register(context(2, 50), 2);
        var high = registry.register(context(3, 90), 2);

        assertCode(StrategyErrorType.PRIORITY_PREEMPTED, low);
        assertFalse(medium.isDone());
        assertFalse(high.isDone());
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(4, 50), 2));
        assertNull(registry.requestSlot(4));
        assertEquals(2, registry.liveRequestCount());
        registry.cancelRequest(3, 0, CancelReason.CLIENT_CANCELLED);
        assertCode(StrategyErrorType.REQUEST_CANCELLED, high);
        assertFalse(registry.register(context(5, 10), 2).isDone());
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(6, 10), 2));
    }

    @Test
    void equalLowerAndFifoAdmissionsDoNotDisplaceQueuedOwners() throws Exception {
        var low = registry.register(context(1, 50), 1);
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(2, 50), 1));
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(3, 10), 1));
        SchedulingTestConfig.useFifoQueue(config);
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(4, 90), 1));
        assertFalse(low.isDone());
        assertEquals(1, registry.snapshotSlots().size());
    }

    @Test
    void expiredHighPriorityAdmissionDoesNotDisplaceLiveWork() throws Exception {
        var low = registry.register(context(1, 10), 1);
        var expired = context(2, 90);
        expired.setSchedulingMetadata(SchedulingMetadata.explicit(90, System.currentTimeMillis() - 1));
        assertCode(StrategyErrorType.BATCH_SLO_EXPIRED, registry.register(expired, 1));
        assertFalse(low.isDone());
        assertNull(registry.requestSlot(2));
    }

    @Test
    void duplicateHighPrioritySubmissionsDisplaceExactlyOneVictim() throws Exception {
        var lows = fillLowPriorityAdmissions();
        var responses = submitConcurrentHighPriorities(true);
        assertEquals(1, responses.stream().filter(response -> !response.isDone()).count());
        for (var response : responses) {
            if (response.isDone()) {
                assertCode(StrategyErrorType.INVALID_REQUEST, response);
            }
        }
        RequestLifecycleTestSupport.awaitCondition(() ->
                lows.stream().filter(CompletableFuture::isDone).count() == 1);
        assertEquals(8, registry.liveRequestCount());
        assertEquals(9, registry.snapshotSlots().size());
    }

    @Test
    void concurrentHighPriorityAdmissionsKeepTheExactOutstandingBound() throws Exception {
        var lows = fillLowPriorityAdmissions();
        var responses = submitConcurrentHighPriorities(false);
        assertEquals(8, responses.stream().filter(response -> !response.isDone()).count());
        for (var response : responses) {
            if (response.isDone()) {
                assertCode(StrategyErrorType.QUEUE_FULL, response);
            }
        }
        for (var low : lows) {
            assertCode(StrategyErrorType.PRIORITY_PREEMPTED, low);
        }
        assertEquals(8, registry.liveRequestCount());
        assertEquals(16, registry.snapshotSlots().size());
    }

    @Test
    void admissionMutationIsNotLocallyPreemptibleUntilItsOwnerSettles() throws Exception {
        var low = registry.register(context(1, 10), 1);
        try (var mutation = registry.claimAdmissionMutation(1, low)) {
            assertNotNull(mutation);
            assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(2, 90), 1));
            assertFalse(low.isDone());
        }
        assertFalse(registry.register(context(3, 90), 1).isDone());
        assertCode(StrategyErrorType.PRIORITY_PREEMPTED, low);
    }

    @Test
    void aDeliveryClaimCannotYieldItsOutstandingPermit() throws Exception {
        var low = registry.register(context(1, 10), 1);
        var slot = registry.requestSlot(1);
        synchronized (slot) {
            slot.startRouteDecisionDelivery();
        }
        SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(2, 90), 1));
        assertFalse(low.isDone());
    }

    @Test
    void placedVictimsRespectTheExistingStagePolicyAndCleanupRunsOutsideAdmissionLock() throws Exception {
        var lowContext = context(1, 10);
        var low = registry.register(lowContext, 1);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        var reservation = new DecodeEndpoint.ReservationHandle(1, 1, 1);
        var item = new ScheduledRequest(lowContext, low, new Response(), null, null, null,
                decode, reservation, System.currentTimeMillis());
        RequestLifecycleTestSupport.bind(registry, new RequestLifecycleTestSupport.Registered(item, low));
        assertCode(StrategyErrorType.QUEUE_FULL, registry.register(context(2, 90), 1));
        SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
        CountDownLatch cleanupStarted = new CountDownLatch(1);
        CountDownLatch finishCleanup = new CountDownLatch(1);
        doAnswer(invocation -> {
            cleanupStarted.countDown();
            await(finishCleanup);
            return null;
        }).when(decode).releaseReservationExact(any());
        try (var executor = Executors.newFixedThreadPool(2)) {
            var replacement = executor.submit(() -> registry.register(context(3, 90), 1));
            assertTrue(cleanupStarted.await(5, TimeUnit.SECONDS));
            try {
                var contender = executor.submit(() -> registry.register(context(4, 50), 1));
                assertCode(StrategyErrorType.QUEUE_FULL, contender.get(2, TimeUnit.SECONDS));
                assertCode(StrategyErrorType.INVALID_REQUEST, registry.register(context(3, 90), 1));
                assertFalse(low.isDone(), "victim completion waits for its exact resource cleanup");
            } finally {
                finishCleanup.countDown();
            }
            assertFalse(replacement.get(5, TimeUnit.SECONDS).isDone());
        }
        assertCode(StrategyErrorType.PRIORITY_PREEMPTED, low);
        verify(decode).releaseReservationExact(reservation);
    }

    private List<CompletableFuture<Response>> fillLowPriorityAdmissions() {
        var lows = new ArrayList<CompletableFuture<Response>>();
        for (int i = 1; i <= 8; i++) {
            lows.add(registry.register(context(i, 10), 8));
        }
        return lows;
    }

    private List<CompletableFuture<Response>> submitConcurrentHighPriorities(boolean duplicate) throws Exception {
        try (var executor = Executors.newFixedThreadPool(8)) {
            CountDownLatch start = new CountDownLatch(1);
            var submissions = new ArrayList<java.util.concurrent.Future<CompletableFuture<Response>>>();
            for (int i = 100; i < 132; i++) {
                int id = duplicate ? 100 : i;
                submissions.add(executor.submit(() -> {
                    await(start);
                    return registry.register(context(id, 90), 8);
                }));
            }
            start.countDown();
            var responses = new ArrayList<CompletableFuture<Response>>();
            for (var submission : submissions) {
                responses.add(submission.get(5, TimeUnit.SECONDS));
            }
            return responses;
        }
    }

    private BalanceContext context(long id, int priority) {
        var context = RequestLifecycleTestSupport.context(config, id);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority,
                System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    private static void assertCode(StrategyErrorType error, CompletableFuture<Response> future) throws Exception {
        assertEquals(error.getErrorCode(), future.get(5, TimeUnit.SECONDS).getCode());
    }
}
