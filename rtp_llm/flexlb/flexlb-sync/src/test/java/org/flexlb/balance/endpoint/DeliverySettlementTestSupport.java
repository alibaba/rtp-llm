package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatusTestSupport;

import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.*;

/** Real endpoint accounting used by delivery callback ordering tests. */
public final class DeliverySettlementTestSupport {
    private final ReentrantLock lock = new ReentrantLock();
    public final PrefillState prefill = new PrefillState(lock, PrefillActiveIndex.ordered(4, Comparator.comparingLong(RequestRoute::requestId)), System::currentTimeMillis);
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

    public void enqueue(RequestRoute item) {
        lock.lock();
        try {
            assertSame(item, prefill.updateQueue(PrefillState.QueueChange.ENQUEUE, item, Long.MAX_VALUE, System.currentTimeMillis()));
        } finally {
            lock.unlock();
        }
    }

    public PrefillState.ReservationResult<PrefillState.BatchReservation> reserveBatch(
            RequestRoute head, long batchId, int maxBatches) {
        return prefill.reserveBatch(head, batchId, maxBatches, generation.tryAcquireHandoff());
    }

    public void commit(long batchId, List<RequestRoute> items) {
        lock.lock();
        try {
            for (RequestRoute item : items) {
                assertSame(item, prefill.updateQueue(PrefillState.QueueChange.ENQUEUE, item, Long.MAX_VALUE, System.currentTimeMillis()));
            }
        } finally {
            lock.unlock();
        }
        {
            var reservation = prefill.reserveBatch(items.getFirst(), batchId, 2,
                generation.tryAcquireHandoff()).reservation();
            try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                assertNotNull(reservation);
                try (var handoff = EndpointTestSupport.commitBatch(prefill, reservation, items, 100L)) {
                    assertNotNull(handoff);
                }
            }
        }
    }

    public List<PrefillState.PrefillRequestStatus> finish(long batchId, RequestRoute item) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(item.requestId());
        finished.setBatchId(batchId);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of(Long.toString(item.requestId()), finished));
        var observation = WorkerStatusTestSupport.freezeStatusResponse(EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090), response);
        var result = EndpointTestSupport.reconcile(prefill, observation, ignored -> 100L);
        return result.requestStatuses();
    }

    public static void queueDecode(DecodeEndpoint endpoint, DecodeResources.ReservationHandle reservation) {
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

    public static void dispatchDecode(DecodeEndpoint endpoint, DecodeResources.ReservationHandle reservation) {
        queueDecode(endpoint, reservation);
        var permit = endpoint.acquireDispatchPermit(reservation, new DecodeResources.AdmissionCapacity(0, 100L)).handle();
        assertNotNull(permit);
        assertEquals(DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                permit.dispatch().status());
    }

    public static void decodeStatus(DecodeEndpoint endpoint, long requestId, boolean finished) {
        decodeStatus(endpoint, requestId, finished, 0L);
    }

    public static void decodeStatus(DecodeEndpoint endpoint, long requestId, boolean finished, long errorCode) {
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
