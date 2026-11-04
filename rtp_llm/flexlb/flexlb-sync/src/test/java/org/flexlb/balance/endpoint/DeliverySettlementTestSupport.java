package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Real endpoint accounting used by delivery callback ordering tests. */
public final class DeliverySettlementTestSupport {
    private final ReentrantLock lock = new ReentrantLock();
    public final PrefillEndpoint prefill;
    private final EndpointRetirementTestSupport generation =
            new EndpointRetirementTestSupport(() -> {});

    public DeliverySettlementTestSupport() {
        this(
                Comparator.comparingLong(
                        org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem::requestId));
    }

    public DeliverySettlementTestSupport(
            Comparator<org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem> ordering) {
        prefill =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, PrefillActiveIndex.ordered(4, ordering), System::currentTimeMillis);
    }

    public void enqueue(RequestRoute item) {
        lock.lock();
        try {
            assertSame(
                    item, org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(prefill, item));
        } finally {
            lock.unlock();
        }
    }

    public CapacityBoundary.Attempt<PrefillEndpoint.BatchReservation> reserveBatch(
            long batchId, int maxBatches) {
        return prefill.reserveBatch(batchId, maxBatches, generation.tryPinGeneration());
    }

    public org.flexlb.balance.scheduler.Delivery commit(
            org.flexlb.balance.scheduler.Delivery plan) {
        return prefill.commitLocked(plan);
    }

    public void commit(long batchId, List<RequestRoute> items) {
        lock.lock();
        try {
            for (RequestRoute item : items) {
                assertSame(
                        item,
                        org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(prefill, item));
            }
        } finally {
            lock.unlock();
        }
        {
            var reservation =
                    prefill.reserveBatch(batchId, 2, generation.tryPinGeneration()).value();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                assertNotNull(reservation);
                try (var handoff =
                        EndpointTestSupport.commitBatch(prefill, reservation, items, 100L)) {
                    assertNotNull(handoff);
                }
            }
        }
    }

    public List<PrefillEndpoint.PrefillRequestStatus> finish(long batchId, RequestRoute item) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(item.requestId());
        finished.setBatchId(batchId);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of(Long.toString(item.requestId()), finished));
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);
        var result = EndpointTestSupport.reconcile(prefill, observation, ignored -> 100L);
        return result.requestStatuses();
    }

    public static void queueDecode(
            DecodeEndpoint endpoint, DecodeResources.ReservationHandle reservation) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(endpoint.markQueued(pin, reservation).status().applied());
        }
    }

    public static void dispatchDecode(
            DecodeEndpoint endpoint, DecodeResources.ReservationHandle reservation) {
        queueDecode(endpoint, reservation);
        var permit =
                endpoint.acquireDispatchPermit(
                                reservation, new DecodeResources.AdmissionCapacity(0, 100L))
                        .handle();
        assertNotNull(permit);
        assertEquals(
                DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                permit.dispatch().status());
    }

    public static void decodeStatus(DecodeEndpoint endpoint, long requestId, boolean finished) {
        decodeStatus(endpoint, requestId, finished, 0L);
    }

    public static void decodeStatus(
            DecodeEndpoint endpoint, long requestId, boolean finished, long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setErrorCode(errorCode);
        task.setPhase(finished ? null : TaskPhase.KV_ALLOCATED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        var tasks = Map.of(Long.toString(requestId), task);
        response.setRunningTaskInfo(finished ? Map.of() : tasks);
        response.setFinishedTaskInfo(finished ? tasks : Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }
}
