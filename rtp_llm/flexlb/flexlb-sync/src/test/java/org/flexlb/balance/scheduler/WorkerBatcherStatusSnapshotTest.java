package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class WorkerBatcherStatusSnapshotTest {

    @Test
    void maxSequenceLengthBacksUnpublishedBatchTokenCapacity() {
        WorkerStatusResponse response = statusResponse(
                0L, 0L, 0L, 1L);
        response.setMaxSeqLen(150L);
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group-a", "10.0.0.1",
                8080, 9090, "site-a");
        publish(status, response);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        WorkerBatcher runtime = new WorkerBatcher(
                "snapshot-test", endpoint, new FlexlbConfig(),
                mock(DeliveryStrategy.class),
                mock(EndpointEventProjector.class));

        assertEquals(150L, runtime.captureRouteProjectionInputs()
                .queue().constraints().batchTokenCapacity());
    }

    @Test
    void capacityProjectionUsesOneCoherentEngineObservation() {
        WorkerStatusResponse first = statusResponse(
                700L, 600L, 1_000L, 1L);
        WorkerStatusResponse second = statusResponse(
                3L, 5L, 10L, 2L);
        WorkerStatus status = spy(WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group-a", "10.0.0.1",
                8080, 9090, "site-a"));
        publish(status, first);

        AtomicBoolean publishedSecond = new AtomicBoolean();
        doAnswer(invocation -> {
            WorkerStatus.EngineObservation captured =
                    (WorkerStatus.EngineObservation)
                            invocation.callRealMethod();
            if (publishedSecond.compareAndSet(false, true)) {
                publish(status, second);
            }
            return captured;
        }).when(status).committedEngineObservation();

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        WorkerBatcher runtime = new WorkerBatcher(
                "snapshot-test",
                endpoint,
                new FlexlbConfig(),
                mock(DeliveryStrategy.class),
                mock(EndpointEventProjector.class));

        GroupPlanner.Constraints capacity = runtime
                .captureRouteProjectionInputs().queue().constraints();
        assertEquals(700L, capacity.batchTokenCapacity());
        assertEquals(600L, capacity.batchKvCapacity());
        assertTrue(publishedSecond.get());
        assertEquals(5L,
                status.committedEngineObservation()
                        .availableKvCacheTokens());
    }

    private static void publish(
            WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            status.publishPreparedStatus(status.prepareNewStatus(
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
}
