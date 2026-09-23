package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Real endpoint accounting used by delivery callback ordering tests. */
public final class DeliverySettlementTestSupport {
    private final ReentrantLock lock = new ReentrantLock();
    public final PrefillState prefill = new PrefillState(lock,
                PrefillActiveIndex.ordered(4, Comparator.comparing(ScheduledRequest::requestId)),
            System::currentTimeMillis, () -> { });
    private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

    public void enqueue(ScheduledRequest item) {
        lock.lock();
        try {
            assertTrue(prefill.enqueueActiveUnderLock(item, Long.MAX_VALUE));
        } finally {
            lock.unlock();
        }
    }

    public PrefillState.ReservationResult<PrefillState.BatchReservation> reserveBatch(
            ScheduledRequest head, long batchId, int maxBatches) {
        return prefill.reserveBatch(head, batchId, maxBatches, generation.tryAcquireHandoff());
    }

    public void commit(long batchId, List<ScheduledRequest> items) {
        lock.lock();
        try {
            for (ScheduledRequest item : items) {
                assertTrue(prefill.enqueueActiveUnderLock(item, Long.MAX_VALUE));
            }
        } finally {
            lock.unlock();
        }
        try (var reservation = prefill.reserveBatch(items.getFirst(), batchId, 2,
                generation.tryAcquireHandoff()).reservation()) {
            assertNotNull(reservation);
            try (var handoff = reservation.commit(items, 100L)) {
                assertNotNull(handoff);
            }
        }
    }

    public List<PrefillState.WorkerStatusFact> finish(long batchId, ScheduledRequest item) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(item.requestId());
        finished.setBatchId(batchId);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of(item.requestId(), finished));
        var observation = EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090)
                .freezeStatusResponse(response);
        var result = prefill.reconcileWorkerStatus(observation, ignored -> 100L, () -> { }, () -> { });
        assertNull(result.publicationFailure());
        return result.schedulerFacts();
    }

    public static void queueDecode(DecodeEndpoint endpoint, DecodeEndpoint.ReservationHandle reservation) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        try (var pin = endpoint.tryPinGeneration()) {
            assertTrue(endpoint.markQueued(pin, reservation));
        }
    }

    public static void dispatchDecode(DecodeEndpoint endpoint, DecodeEndpoint.ReservationHandle reservation) {
        queueDecode(endpoint, reservation);
        var permit = endpoint.acquireDispatchPermit(reservation, new DecodeEndpoint.AdmissionCapacity(0, 100L)).permit();
        assertNotNull(permit);
        assertEquals(DecodeEndpoint.EngineDispatchPermitTransferStatus.TRANSFERRED,
                permit.dispatch());
    }

    public static void decodeStatus(DecodeEndpoint endpoint, long requestId, boolean finished) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
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
