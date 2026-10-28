package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

/**
 * Checks coherent worker capacity observations and fallback values in batcher snapshots.
 */
class WorkerBatcherStatusSnapshotTest {

    @Test
    void configuredFallbackBacksUnpublishedEngineCapacities() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFallbackBatchTokenCapacity(2_097_152L);
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "group-a", "10.0.0.1", 8080, 9090, "site-a");
        publish(status, statusResponse(0L, 0L, 0L, 1L));
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        WorkerBatcher runtime =
                new WorkerBatcher(
                        "snapshot-test",
                        endpoint,
                        config,
                        mock(DeliveryStrategy.class),
                        mock(EndpointEventProjector.class));

        assertEquals(
                2_097_152L,
                runtime.captureRouteProjectionInputs().queue().constraints().batchTokenCapacity());
    }

    @Test
    void maxSequenceLengthBacksUnpublishedBatchTokenCapacity() {
        WorkerStatusResponse response = statusResponse(0L, 0L, 0L, 1L);
        response.setMaxSeqLen(150L);
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "group-a", "10.0.0.1", 8080, 9090, "site-a");
        publish(status, response);

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        WorkerBatcher runtime =
                new WorkerBatcher(
                        "snapshot-test",
                        endpoint,
                        org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig(),
                        mock(DeliveryStrategy.class),
                        mock(EndpointEventProjector.class));

        assertEquals(
                150L,
                runtime.captureRouteProjectionInputs().queue().constraints().batchTokenCapacity());
    }

    @Test
    void capacityProjectionUsesOneCoherentEngineObservation() {
        WorkerStatusResponse first = statusResponse(700L, 600L, 1_000L, 1L);
        WorkerStatusResponse second = statusResponse(3L, 5L, 10L, 2L);
        WorkerStatus status =
                spy(
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "group-a", "10.0.0.1", 8080, 9090, "site-a"));
        publish(status, first);

        AtomicBoolean publishedSecond = new AtomicBoolean();
        doAnswer(
                        invocation -> {
                            WorkerStatus.EngineObservation captured =
                                    (WorkerStatus.EngineObservation) invocation.callRealMethod();
                            if (publishedSecond.compareAndSet(false, true)) {
                                publish(status, second);
                            }
                            return captured;
                        })
                .when(status)
                .committedEngineObservation();

        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        WorkerBatcher runtime =
                new WorkerBatcher(
                        "snapshot-test",
                        endpoint,
                        org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig(),
                        mock(DeliveryStrategy.class),
                        mock(EndpointEventProjector.class));

        GroupPlanner.Constraints capacity =
                runtime.captureRouteProjectionInputs().queue().constraints();
        assertEquals(700L, capacity.batchTokenCapacity());
        assertEquals(600L, capacity.batchKvCapacity());
        assertTrue(publishedSecond.get());
        assertEquals(5L, status.committedEngineObservation().availableKvCacheTokens());
    }

    @Test
    void publishedKvBoundPreventsOverBudgetGroups() {
        WorkerBatcher runtime = runtimeFor(statusResponse(16_384L, 80_000L, 100_000L, 1L));
        GroupPlanner.Constraints constraints =
                runtime.captureRouteProjectionInputs().queue().constraints();
        List<GroupPlanner.Item> requests =
                IntStream.range(0, 8)
                        .mapToObj(i -> new GroupPlanner.Item("r" + i, 0, i, 0L, Long.MAX_VALUE, 10_000L, 0L))
                        .toList();

        GroupPlanner.Selection<GroupPlanner.Item> selected =
                GroupPlanner.select(requests, GroupPlanner.itemAccess(), constraints, null);

        assertEquals(List.of(requests.get(0)), selected.items());
        assertEquals(80_000L, constraints.batchKvCapacity());
    }

    @Test
    void publishedQueryOnlyBudgetStillAllowsFullGroup() {
        WorkerBatcher runtime = runtimeFor(statusResponse(262_144L, 80_000L, 100_000L, 1L));
        List<GroupPlanner.Item> requests =
                IntStream.range(0, 8)
                        .mapToObj(i -> new GroupPlanner.Item("r" + i, 0, i, 0L, Long.MAX_VALUE, 10_000L, 0L))
                        .toList();

        GroupPlanner.Selection<GroupPlanner.Item> selected =
                GroupPlanner.select(requests, GroupPlanner.itemAccess(),
                        runtime.captureRouteProjectionInputs().queue().constraints(), null);

        assertEquals(requests, selected.items());
    }

    @Test
    void publishedSingletonCapDoesNotActivateCapacityFallback() {
        WorkerBatcher runtime = runtimeFor(statusResponse(1L, 80_000L, 100_000L, 1L));
        GroupPlanner.Item head = new GroupPlanner.Item("head", 0, 0L, 0L, Long.MAX_VALUE, 1L, 0L);
        GroupPlanner.Item next = new GroupPlanner.Item("next", 0, 1L, 0L, Long.MAX_VALUE, 1L, 0L);

        GroupPlanner.Selection<GroupPlanner.Item> selected =
                GroupPlanner.select(List.of(head, next), GroupPlanner.itemAccess(),
                        runtime.captureRouteProjectionInputs().queue().constraints(), null);

        assertEquals(List.of(head), selected.items());
    }

    @Test
    void publishedKvBoundPreservesOversizedHeadWithEnoughPhysicalKv() {
        WorkerBatcher runtime = runtimeFor(statusResponse(16_384L, 80_000L, 100_000L, 1L));
        GroupPlanner.Item head = new GroupPlanner.Item("head", 0, 0L, 0L, Long.MAX_VALUE, 20_000L, 0L);
        GroupPlanner.Item next = new GroupPlanner.Item("next", 0, 1L, 0L, Long.MAX_VALUE, 1L, 0L);

        GroupPlanner.Selection<GroupPlanner.Item> selected = GroupPlanner.select(
                List.of(head, next), GroupPlanner.itemAccess(),
                runtime.captureRouteProjectionInputs().queue().constraints(), null);

        assertEquals(List.of(head), selected.items());
        assertEquals(20_000L, selected.shape().kvTokens());
    }

    private static WorkerBatcher runtimeFor(WorkerStatusResponse response) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "group-a", "10.0.0.1", 8080, 9090, "site-a");
        publish(status, response);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(status);
        return new WorkerBatcher("snapshot-test", endpoint, SchedulingTestConfig.newConfig(),
                mock(DeliveryStrategy.class), mock(EndpointEventProjector.class));
    }

    private static void publish(WorkerStatus status, WorkerStatusResponse response) {
        status.lock.lock();
        try {
            status.publishPreparedStatus(
                    status.prepareNewStatus(status.freezeStatusResponse(response)));
        } finally {
            status.lock.unlock();
        }
    }

    private static WorkerStatusResponse statusResponse(
            long maxBatchTokens, long availableKvTokens, long totalKvTokens, long statusVersion) {
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
