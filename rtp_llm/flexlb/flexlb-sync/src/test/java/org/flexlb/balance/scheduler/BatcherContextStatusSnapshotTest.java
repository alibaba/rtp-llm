package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillState;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.PriorityBlockingQueue;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class BatcherContextStatusSnapshotTest {

    /**
     * {@code batchCapacitySnapshot} must read one coherent {@link
     * WorkerStatus.EngineObservation}. A newer status committed in the middle of
     * that read must not tear the token/KV values used for one capacity
     * decision. The spy publishes a second, smaller status exactly once while
     * the first observation is being returned.
     */
    @Test
    void kvCapacityUsesOneCoherentEngineObservationEvenWhenANewerResponsePublishes() {
        WorkerStatusResponse first = statusResponse(700L, 600L, 1_000L, 1L);
        WorkerStatusResponse second = statusResponse(3L, 5L, 10L, 2L);

        WorkerStatus status = spy(WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group-a", "10.0.0.1", 8080, 9090, "site-a"));
        publish(status, first);

        AtomicBoolean publishedSecond = new AtomicBoolean();
        doAnswer(invocation -> {
            WorkerStatus.EngineObservation captured =
                    (WorkerStatus.EngineObservation) invocation.callRealMethod();
            if (publishedSecond.compareAndSet(false, true)) {
                publish(status, second);
            }
            return captured;
        }).when(status).committedEngineObservation();

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        BatcherContext context = context(endpoint);

        BatcherContext.BatchCapacitySnapshot capacity =
                context.batchCapacitySnapshot();
        assertEquals(700L, capacity.batchTokenCapacity());
        assertEquals(600L, capacity.batchKvCapacity());
        assertTrue(publishedSecond.get(), "the newer status must have committed");
        // A subsequent read now observes the newer committed status.
        assertEquals(5L,
                status.committedEngineObservation().availableKvCacheTokens());
    }

    private static void publish(WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            status.publishPreparedStatus(
                    status.prepareNewStatus(
                            status.freezeStatusResponse(response)));
        } finally {
            status.lock.unlock();
        }
    }

    private static WorkerStatusResponse statusResponse(
            long maxBatchTokens,
            long availableKvTokens,
            long totalKvTokens,
            long statusVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setMaxBatchTokensSize(maxBatchTokens);
        response.setAvailableKvCacheTokens(availableKvTokens);
        response.setTotalKvCacheTokens(totalKvTokens);
        response.setStatusVersion(statusVersion);
        return response;
    }

    private static BatcherContext context(PrefillEndpoint endpoint) {
        PriorityBlockingQueue<ScheduledRequest> queue =
                new PriorityBlockingQueue<>(11, WorkerBatcher.PRIORITY_QUEUE_ORDER);
        ReentrantLock queueLock = new ReentrantLock();
        PrefillState workRegistry =
                new PrefillState(queueLock, queue, () -> { });
        return new BatcherContext(
                "snapshot-test",
                endpoint,
                new FlexlbConfig(),
                mock(EndpointEventSink.class),
                queue,
                new AtomicLong(),
                queueLock,
                WorkerBatcher.PRIORITY_QUEUE_ORDER,
                Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                        .thenComparingLong(GroupPlanner.Item::requestId),
                false,
                mock(DeliveryStrategy.class),
                workRegistry);
    }
}
