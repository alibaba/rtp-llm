package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Twin overload protection at real FlexLB endpoint state machines. WorkerStatus
 * replaces the external engine; there is no mock performance model or GPU here.
 * Batch-limit assertions exercise deliveries owned by this Master. They do not
 * require unowned engine batches to consume local batch delivery capacity.
 */
class TwinOverloadAdmissionTest {

    private final List<WorkerEndpoint> endpoints = new ArrayList<>();

    @AfterEach
    void closeEndpoints() {
        endpoints.forEach(WorkerEndpoint::close);
    }

    @Test
    void groupSizeLimitAloneDoesNotCapTotalInflightRequests() {
        FlexlbConfig config = config(null);
        PrefillEndpoint endpoint = prefill(config);
        commitBatch(endpoint, config, 1L, 1L);
        commitBatch(endpoint, config, 2L, 9L);

        assertEquals(2, endpoint.getInflightBatchCount());
        assertEquals(16, endpoint.observedRequestCount(),
                "maxRequests=8 bounds one decision, not the engine's total inflight work");
    }

    @Test
    void boundedBatchSlotRemainsOccupiedUntilLastMemberFinishes() {
        FlexlbConfig config = config(1);
        PrefillEndpoint endpoint = prefill(config);
        commitBatch(endpoint, config, 11L, 1L);
        assertEquals(8, endpoint.observedRequestCount());
        assertTrue(!endpoint.batchAdmissionAvailability(1).isAvailable());

        Map<String, TaskInfo> finished = tasks(1L, 7, 11L, TaskPhase.RUNNING);
        applyStatus(endpoint, tasks(8L, 1, 11L, TaskPhase.RUNNING), finished);
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.observedRequestCount(),
                "completed members immediately leave ownership while the last member retains the batch");
        assertTrue(!endpoint.batchAdmissionAvailability(1).isAvailable(),
                "seven finished members do not release the final member's batch slot");

        applyStatus(endpoint, Map.of(), tasks(8L, 1, 11L, TaskPhase.RUNNING));
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(0, endpoint.observedRequestCount());
        assertTrue(endpoint.batchAdmissionAvailability(1).isAvailable());
        // The full snapshot may repeat across polls without releasing ownership twice.
        applyStatus(endpoint, Map.of(), tasks(8L, 1, 11L, TaskPhase.RUNNING));
        assertEquals(0, endpoint.observedRequestCount());
    }

    @ParameterizedTest
    @EnumSource(value = TaskPhase.class, names = {"RECEIVED", "RUNNING"})
    void locallyOwnedBatchStatusMustRetainExactlyOneDeliverySlot(TaskPhase phase) {
        FlexlbConfig config = config(2);
        PrefillEndpoint endpoint = prefill(config);
        commitBatch(endpoint, config, 70L, 101L);
        assertEquals(8, endpoint.observedRequestCount());

        applyStatus(endpoint, tasks(101L, 8, 70L, phase), Map.of());
        assertEquals(8, endpoint.observedRequestCount(),
                "the local batch and its engine observation describe the same eight requests");
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "a local batch must consume one slot without being counted again by worker status");

        applyStatus(endpoint, tasks(101L, 8, 70L, phase), Map.of());
        assertEquals(1, endpoint.getInflightBatchCount(),
                "repeated status must neither release nor duplicate the local batch slot");

        applyStatus(endpoint, Map.of(), tasks(101L, 8, 70L, phase));
        assertEquals(0, endpoint.observedRequestCount());
    }

    @Test
    void localBatchReservationMustBlockAnotherUntilReleased() {
        FlexlbConfig config = config(1);
        PrefillEndpoint endpoint = prefill(config);
        ScheduledRequest first = item(endpoint, config, 1L);
        ScheduledRequest staged = item(endpoint, config, 2L);
        assertTrue(EndpointTestSupport.offer(endpoint, first));
        assertTrue(EndpointTestSupport.offer(endpoint, staged));
        PrefillState.ReservationResult<PrefillState.BatchReservation> acquired =
                endpoint.reserveBatch(first, 70L, 1);
        assertEquals(PrefillState.CapacityStatus.ACQUIRED, acquired.status());
        try (PrefillState.BatchReservation ignored = acquired.reservation()) {
            assertEquals(2, endpoint.queuedRequestCount());
            PrefillState.ReservationResult<PrefillState.BatchReservation> blocked =
                    endpoint.reserveBatch(staged, 71L, 1);
            try (PrefillState.BatchReservation unexpected = blocked.reservation()) {
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL, blocked.status(),
                        "a locally reserved batch must occupy the only slot at final admission");
            }
        }
        assertEquals(2, endpoint.queuedRequestCount());
        PrefillState.ReservationResult<PrefillState.BatchReservation> resumed =
                endpoint.reserveBatch(staged, 71L, 1);
        try (PrefillState.BatchReservation ignored = resumed.reservation()) {
            assertEquals(PrefillState.CapacityStatus.ACQUIRED, resumed.status());
        }
    }

    @Test
    void nonBatchPreservesUnknownEngineBacklogUntilAnAuthoritativeUpdate() {
        FlexlbConfig config = config(null);
        config.setDispatcher(DispatcherConfig.nonBatch());
        PrefillEndpoint endpoint = prefill(config);
        applyStatus(endpoint, tasks(101L, 8, 70L, TaskPhase.RUNNING), Map.of());
        assertEquals(8, endpoint.observedRequestCount());
        assertTrue(endpoint.captureRouteProjectionInputs().work().totalRemainingWorkMs().isEmpty());
        applyStatus(endpoint, Map.of(), tasks(101L, 8, 70L, TaskPhase.RUNNING));
        assertEquals(0, endpoint.observedRequestCount());
        assertEquals(0L, endpoint.captureRouteProjectionInputs().work().totalRemainingWorkMs().orElseThrow());
    }

    @Test
    void decodeRequestCapMustBlockEvenWithAlmostEmptyKvPool() {
        DecodeEndpoint endpoint = new DecodeEndpoint(
                EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.2", 8080, 8090),
                EndpointTestSupport.noopEventSink());
        endpoints.add(endpoint);
        applyStatus(endpoint, Map.of(), Map.of());
        for (long id = 1; id <= 9; id++) {
            try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
                assertNotNull(pin);
                assertNotNull(endpoint.tryReservePlacementPinned(pin, id, 128L, 256L, 50));
            }
        }
        for (long id = 1; id <= 8; id++) {
            DecodeEndpoint.EngineDispatchPermitAcquisition acquisition =
                    endpoint.acquireEngineDispatchPermit(endpoint.reservationHandle(id), new DecodeEndpoint.AdmissionCapacity(8L, 90L));
            assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, acquisition.status());
            assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.permit().transferToEngineLifecycle());
        }
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                endpoint.acquireEngineDispatchPermit(endpoint.reservationHandle(9L), new DecodeEndpoint.AdmissionCapacity(8L, 90L)).status());

        applyStatus(endpoint, tasks(2L, 7, 10L, TaskPhase.RUNNING),
                tasks(1L, 1, 10L, TaskPhase.RUNNING));
        DecodeEndpoint.EngineDispatchPermitAcquisition resumed =
                endpoint.acquireEngineDispatchPermit(endpoint.reservationHandle(9L), new DecodeEndpoint.AdmissionCapacity(8L, 90L));
        assertEquals(DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED, resumed.status());
        assertTrue(resumed.permit().release());
    }

    private PrefillEndpoint prefill(FlexlbConfig config) {
        EndpointTestSupport.TestRequestRuntime runtime = EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint = new PrefillEndpoint(
                EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                config, EndpointTestSupport.routeStrategy(runtime), runtime.events(),
                mock(BatchSchedulerReporter.class));
        endpoints.add(endpoint);
        endpoint.startGeneration();
        return endpoint;
    }

    private static FlexlbConfig config(Integer batchLimit) {
        FlexlbConfig config = org.flexlb.balance.scheduler.SchedulingTestConfig.newConfig();
        config.fixedWindowDecision().setMaxRequests(8);
        config.queueScheduler().setQueueTimeoutMs(3_600_000L);
        config.fixedWindowDecision().setMaxCollectionWaitMs(400L);
        if (batchLimit != null) {
            config.getDispatcher().setMaxInflightPerPrefillWorker(batchLimit);
        }
        return config;
    }

    private static void commitBatch(PrefillEndpoint endpoint, FlexlbConfig config,
                                    long batchId, long firstRequestId) {
        List<ScheduledRequest> items = new ArrayList<>();
        for (long id = firstRequestId; id < firstRequestId + 8; id++) {
            ScheduledRequest item = item(endpoint, config, id);
            assertTrue(EndpointTestSupport.offer(endpoint, item));
            items.add(item);
        }
        int limit = config.getDispatcher().getMaxInflightPerPrefillWorker();
        PrefillState.ReservationResult<PrefillState.BatchReservation> acquisition =
                endpoint.reserveBatch(items.getFirst(), batchId, limit);
        assertEquals(PrefillState.CapacityStatus.ACQUIRED, acquisition.status());
        try (PrefillState.BatchReservation reservation = acquisition.reservation();
             PrefillState.CommittedHandoff ignored = reservation.commit(items, 300_000L)) {
            // Releasing the handoff closes only the generation pin, not engine work.
        }
    }

    private static ScheduledRequest item(PrefillEndpoint endpoint, FlexlbConfig config, long id) {
        Request request = new Request();
        request.setRequestId(id);
        request.setSeqLen(128L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        ServerStatus selected = new ServerStatus();
        selected.setRole(RoleType.PREFILL);
        selected.setServerIp("127.0.0.1");
        selected.setHttpPort(8080);
        selected.setGrpcPort(8090);
        return new ScheduledRequest(context, null, null, selected, null,
                endpoint, null, null, System.currentTimeMillis());
    }

    private static Map<String, TaskInfo> tasks(long firstId, int count, long batchId, TaskPhase phase) {
        Map<String, TaskInfo> result = new LinkedHashMap<>();
        for (long id = firstId; id < firstId + count; id++) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(id);
            task.setBatchId(batchId);
            task.setPhase(phase);
            task.setInputLength(128L);
            task.setErrorCode(0);
            result.put(Long.toString(id), task);
        }
        return result;
    }

    private static void applyStatus(WorkerEndpoint endpoint,
                                    Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setTotalKvCacheTokens(22_000_000L);
        response.setAvailableKvCacheTokens(21_900_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }
}
