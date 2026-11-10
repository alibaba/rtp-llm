package org.flexlb.balance.endpoint;

import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;

import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Real endpoint accounting used by delivery callback ordering tests.
 */
public final class DeliverySettlementTestSupport {
    private final ReentrantLock lock = new ReentrantLock();
    public final PrefillEndpoint prefill;
    private final EndpointRetirementTestSupport generation =
            new EndpointRetirementTestSupport(() -> {});

    public DeliverySettlementTestSupport() {
        prefill =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, OrderedPrefillRequestQueue.fifo(4), System::currentTimeMillis);
    }

    public void enqueue(RouteResult item) {
        runWithStateLock(
                lock,
                () -> {
                    assertSame(
                            item,
                            org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(prefill, item));
                });
    }

    public org.flexlb.balance.delivery.CapacityBoundary.Attempt<WorkSnapshot> commit(
            PrefillEndpoint.DeliverySelection plan) {
        return prefill.dequeueForDelivery(plan);
    }

    public void commit(long batchId, List<RouteResult> items) {
        runWithStateLock(
                lock,
                () -> {
                    for (RouteResult item : items) {
                        assertSame(
                                item,
                                org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                        prefill, item));
                    }
                });
        {
            var reservation =
                    new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(batchId, 2);
            {
                assertNotNull(reservation);
                {
                    var handoff =
                            EndpointTestSupport.commitBatch(prefill, reservation, items, 100L);

                    assertNotNull(handoff);
                }
            }
        }
    }

    public List<PrefillEndpoint.PrefillRequestStatus> finish(long batchId, RouteResult item) {
        TaskInfo finished = new TaskInfo();
        finished.setRequestId(String.valueOf(item.requestId()));
        finished.setBatchId(batchId);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of(item.requestId(), finished));
        var observation =
                WorkerStatusTestSupport.freezeStatusResponse(
                        EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8090),
                        response);
        var result = EndpointTestSupport.reconcile(prefill, observation, ignored -> 100L);
        return result.requestStatuses();
    }

    public static void dispatchDecode(
            DecodeEndpoint endpoint, DecodeResources.RequestReservation reservation) {
        queueDecode(endpoint, reservation);
        var permit =
                endpoint.acquireDispatchPermit(
                                reservation, new DecodeResources.AdmissionCapacity(0, 100L))
                        .value();
        assertNotNull(permit);
        assertEquals(
                DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED,
                permit.dispatch().status());
    }

    public static void queueDecode(
            DecodeEndpoint endpoint, DecodeResources.RequestReservation reservation) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        org.junit.jupiter.api.Assertions.assertTrue(
                endpoint.resourceSnapshot()
                        .requests()
                        .get(reservation.requestId())
                        .phase()
                        .isMasterQueued());
    }

    public static void decodeStatus(DecodeEndpoint endpoint, long requestId, boolean finished) {
        decodeStatus(endpoint, requestId, finished, 0L);
    }

    public static void decodeStatus(
            DecodeEndpoint endpoint, long requestId, boolean finished, long errorCode) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
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
