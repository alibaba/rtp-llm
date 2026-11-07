package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RouteExecution;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
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

        assertEquals(2, endpoint.stats().batchCount());
        assertEquals(
                16,
                endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests(),
                "maxRequests=8 bounds one decision, not the engine's total inflight work");
    }

    @Test
    void boundedBatchSlotRemainsOccupiedUntilLastMemberFinishes() {
        FlexlbConfig config = config(1);
        PrefillEndpoint endpoint = prefill(config);
        commitBatch(endpoint, config, 11L, 1L);
        assertEquals(8, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
        assertTrue(!endpoint.batchCapacityAvailable(1));

        Map<String, TaskInfo> finished = tasks(1L, 7, 11L, TaskPhase.RUNNING);
        applyStatus(endpoint, tasks(8L, 1, 11L, TaskPhase.RUNNING), finished);
        assertEquals(1, endpoint.stats().batchCount());
        assertEquals(
                1,
                endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests(),
                "completed members immediately leave ownership while the last member retains the batch");
        assertTrue(
                !endpoint.batchCapacityAvailable(1),
                "seven finished members do not release the final member's batch slot");

        applyStatus(endpoint, Map.of(), tasks(8L, 1, 11L, TaskPhase.RUNNING));
        assertEquals(0, endpoint.stats().batchCount());
        assertEquals(0, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
        assertTrue(endpoint.batchCapacityAvailable(1));
        // The full snapshot may repeat across polls without releasing ownership twice.
        applyStatus(endpoint, Map.of(), tasks(8L, 1, 11L, TaskPhase.RUNNING));
        assertEquals(0, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
    }

    @ParameterizedTest
    @EnumSource(
            value = TaskPhase.class,
            names = {"RECEIVED", "RUNNING"})
    void locallyOwnedBatchStatusMustRetainExactlyOneDeliverySlot(TaskPhase phase) {
        FlexlbConfig config = config(2);
        PrefillEndpoint endpoint = prefill(config);
        commitBatch(endpoint, config, 70L, 101L);
        assertEquals(8, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());

        applyStatus(endpoint, tasks(101L, 8, 70L, phase), Map.of());
        assertEquals(
                8,
                endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests(),
                "the local batch and its engine observation describe the same eight requests");
        assertEquals(1, endpoint.stats().batchCount());
        assertEquals(
                1,
                endpoint.stats().batchCount(),
                "a local batch must consume one slot without being counted again by worker status");

        applyStatus(endpoint, tasks(101L, 8, 70L, phase), Map.of());
        assertEquals(
                1,
                endpoint.stats().batchCount(),
                "repeated status must neither release nor duplicate the local batch slot");

        applyStatus(endpoint, Map.of(), tasks(101L, 8, 70L, phase));
        assertEquals(0, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
    }

    @Test
    void localBatchRecordMustBlockAnotherUntilReleased() {
        FlexlbConfig config = config(1);
        PrefillEndpoint endpoint = prefill(config);
        RouteExecution first = item(endpoint, config, 1L);
        RouteExecution staged = item(endpoint, config, 2L);
        assertTrue(EndpointTestSupport.offer(endpoint, first));
        assertTrue(EndpointTestSupport.offer(endpoint, staged));
        CapacityBoundary.Attempt<PrefillEndpoint.BatchRecord> acquired =
                endpoint.reserveBatch(70L, 1);
        assertTrue(acquired.accepted());
        {
            PrefillEndpoint.BatchRecord ignored = acquired.value();
            try (var preparationIgnored = EndpointTestSupport.preparation(ignored)) {
                assertEquals(2, endpoint.progress().queueDepth());
                CapacityBoundary.Attempt<PrefillEndpoint.BatchRecord> blocked =
                        endpoint.reserveBatch(71L, 1);
                {
                    PrefillEndpoint.BatchRecord unexpected = blocked.value();
                    try (var preparationUnexpected = EndpointTestSupport.preparation(unexpected)) {
                        assertEquals(
                                CapacityBoundary.Resource.PREFILL_BATCH,
                                blocked.boundary().resource(),
                                "a locally reserved batch must occupy the only slot at final admission");
                    }
                }
            }
        }
        assertEquals(2, endpoint.progress().queueDepth());
        CapacityBoundary.Attempt<PrefillEndpoint.BatchRecord> resumed =
                endpoint.reserveBatch(71L, 1);
        {
            PrefillEndpoint.BatchRecord ignored = resumed.value();
            try (var preparationIgnored = EndpointTestSupport.preparation(ignored)) {
                assertTrue(resumed.accepted());
            }
        }
    }

    @Test
    void nonBatchPreservesUnknownEngineBacklogUntilAnAuthoritativeUpdate() {
        FlexlbConfig config = config(null);
        config.setDispatcher(ConfigTestSupport.nonBatch());
        PrefillEndpoint endpoint = prefill(config);
        applyStatus(endpoint, tasks(101L, 8, 70L, TaskPhase.RUNNING), Map.of());
        assertEquals(8, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
        assertTrue(endpoint.captureRouteProjectionInputs().work().totalRemainingWorkMs().isEmpty());
        applyStatus(endpoint, Map.of(), tasks(101L, 8, 70L, TaskPhase.RUNNING));
        assertEquals(0, endpoint.admissionSummary(0, endpoint.requestLimit).occupiedRequests());
        assertEquals(
                0L,
                endpoint.captureRouteProjectionInputs()
                        .work()
                        .totalRemainingWorkMs()
                        .orElseThrow());
    }

    @Test
    void decodeRequestCapMustBlockEvenWithAlmostEmptyKvPool() {
        DecodeEndpoint endpoint =
                EndpointTestSupport.decode(
                        EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.2", 8080, 8090),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                EndpointTestSupport.noopEventSink()));
        endpoints.add(endpoint);
        applyStatus(endpoint, Map.of(), Map.of());
        for (long id = 1; id <= 9; id++) {
            {
                WorkerEndpoint selectedEndpoint = endpoint;

                assertNotNull(selectedEndpoint);
                assertNotNull(
                        endpoint.tryReserveQueuedRequest(String.valueOf(id), 128L, 256L, 50, null)
                                .value());
            }
        }
        for (long id = 1; id <= 8; id++) {
            org.flexlb.balance.endpoint.DecodeResources.Acquisition<
                            org.flexlb.balance.endpoint.DecodeResources
                                    .EngineDispatchPermitAcquireStatus,
                            DecodeEndpoint.EngineDispatchPermit>
                    acquisition =
                            endpoint.acquireDispatchPermit(
                                    EndpointTestSupport.decodeReservation(endpoint, id),
                                    new DecodeResources.AdmissionCapacity(8L, 90L));
            assertEquals(
                    DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED,
                    acquisition.status());
            assertEquals(
                    DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                    acquisition.value().dispatch().status());
        }
        assertEquals(
                DecodeResources.EngineDispatchPermitAcquireStatus.CAPACITY_FULL,
                endpoint.acquireDispatchPermit(
                                EndpointTestSupport.decodeReservation(endpoint, 9L),
                                new DecodeResources.AdmissionCapacity(8L, 90L))
                        .status());

        applyStatus(
                endpoint,
                tasks(2L, 7, 10L, TaskPhase.RUNNING),
                tasks(1L, 1, 10L, TaskPhase.RUNNING));
        org.flexlb.balance.endpoint.DecodeResources.Acquisition<
                        org.flexlb.balance.endpoint.DecodeResources
                                .EngineDispatchPermitAcquireStatus,
                        DecodeEndpoint.EngineDispatchPermit>
                resumed =
                        endpoint.acquireDispatchPermit(
                                EndpointTestSupport.decodeReservation(endpoint, 9L),
                                new DecodeResources.AdmissionCapacity(8L, 90L));
        assertEquals(DecodeResources.EngineDispatchPermitAcquireStatus.ACQUIRED, resumed.status());
        assertTrue(resumed.value().release());
    }

    private PrefillEndpoint prefill(FlexlbConfig config) {
        EndpointTestSupport.TestRequestRuntime runtime = EndpointTestSupport.requestRuntime();
        PrefillEndpoint endpoint =
                EndpointTestSupport.prefill(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        config,
                        EndpointTestSupport.routeStrategy(runtime),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                runtime.events()),
                        mock(DeliveryMetricsReporter.class));
        endpoints.add(endpoint);
        return endpoint;
    }

    private static FlexlbConfig config(Integer batchLimit) {
        FlexlbConfig config = SchedulingTestConfig.newConfig();
        config.decisionPolicy().setMaxRequests(8);
        config.queueScheduler().setQueueTimeoutMs(3_600_000L);
        config.decisionPolicy().setMaxCollectionWaitMs(400L);
        if (batchLimit != null) {
            config.getDispatcher().setMaxInflightPerPrefillWorker(batchLimit);
        }
        return config;
    }

    private static void commitBatch(
            PrefillEndpoint endpoint, FlexlbConfig config, long batchId, long firstRequestId) {
        List<RouteExecution> items = new ArrayList<>();
        for (long id = firstRequestId; id < firstRequestId + 8; id++) {
            RouteExecution item = item(endpoint, config, id);
            assertTrue(EndpointTestSupport.offer(endpoint, item));
            items.add(item);
        }
        int limit = config.getDispatcher().getMaxInflightPerPrefillWorker();
        CapacityBoundary.Attempt<PrefillEndpoint.BatchRecord> acquisition =
                endpoint.reserveBatch(batchId, limit);
        assertTrue(acquisition.accepted());
        PrefillEndpoint state = (PrefillEndpoint) endpoint;
        {
            PrefillEndpoint.BatchRecord reservation = acquisition.value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                    org.flexlb.balance.scheduler.Delivery ignored =
                            EndpointTestSupport.commitBatch(state, reservation, items, 300_000L)) {
                // Releasing the handoff closes only the selected Worker, not engine work.
            }
        }
    }

    private static RouteExecution item(PrefillEndpoint endpoint, FlexlbConfig config, long id) {
        Request request = Request.builder().requestId(String.valueOf(id)).seqLen(128L).build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        ServerStatus selected = new ServerStatus();
        selected.setRole(RoleType.PREFILL);
        selected.setServerIp("127.0.0.1");
        selected.setHttpPort(8080);
        selected.setGrpcPort(8090);
        return SchedulingTestConfig.createRoute(
                context, null, selected, null, endpoint, null, null, System.currentTimeMillis());
    }

    private static Map<String, TaskInfo> tasks(
            long firstId, int count, long batchId, TaskPhase phase) {
        Map<String, TaskInfo> result = new LinkedHashMap<>();
        for (long id = firstId; id < firstId + count; id++) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(String.valueOf(id));
            task.setBatchId(batchId);
            task.setPhase(phase);
            task.setInputLength(128L);
            task.setErrorCode(0);
            result.put(Long.toString(id), task);
        }
        return result;
    }

    private static void applyStatus(
            WorkerEndpoint endpoint,
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setTotalKvCacheTokens(22_000_000L);
        response.setAvailableKvCacheTokens(21_900_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }
}
