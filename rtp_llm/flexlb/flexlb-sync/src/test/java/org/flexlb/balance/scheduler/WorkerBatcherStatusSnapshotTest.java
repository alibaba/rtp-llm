package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class WorkerBatcherStatusSnapshotTest {

    @ParameterizedTest
    @CsvSource({
            "0, 150, 0, 0, 150, 9223372036854775807",
            "0, 0, 0, 0, 9223372036854775807, 9223372036854775807",
            "200, 150, 1000, 100, 200, 100",
            "200, 150, -1, 100, 200, 0"
    })
    void capacityFallbacksAndBoundsRemainStable(long batchTokens, long maxSeqLen, long availableKv,
                                               long totalKv, long expectedTokens, long expectedKv) {
        WorkerStatusResponse response = statusResponse(batchTokens, availableKv, totalKv, 1L);
        response.setMaxSeqLen(maxSeqLen);
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group-a", "10.0.0.1",
                8080, 9090, "site-a");
        publish(status, response);

        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                SchedulingTestConfig.newConfig(), status, mock(DeliveryStrategy.class), mock(AbstractRequestScheduler.class));

        GroupPlanner.Constraints constraints = endpoint.captureRouteProjectionInputs().queue().constraints();
        assertEquals(expectedTokens, constraints.batchTokenCapacity());
        assertEquals(expectedKv, constraints.batchKvCapacity());
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

        PrefillEndpoint endpoint = org.flexlb.balance.endpoint.EndpointTestSupport.unstartedPrefill(
                SchedulingTestConfig.newConfig(), status, mock(DeliveryStrategy.class), mock(AbstractRequestScheduler.class));

        GroupPlanner.Constraints capacity = endpoint
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
