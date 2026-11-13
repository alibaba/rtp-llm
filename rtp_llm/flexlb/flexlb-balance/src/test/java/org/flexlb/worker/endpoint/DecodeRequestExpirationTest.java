package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Exact local lease expiration must not wait for Engine reconciliation.
 */
class DecodeRequestExpirationTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeEndpoint.RequestRegistration> reservations = new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.8", 8080, 8081);
        endpoint =
                EndpointTestSupport.decode(
                        status,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                EndpointTestSupport.noopEventSink()));
        updateStatus(Map.of(), Map.of(), 10_000);
    }

    @Test
    void expireUnobservedShadowReleasesAllAccountingAndBlocksStaleStatus() {
        DecodeEndpoint.RequestRegistration reservation = reserve(1L, 500, 700, 0);

        assertEquals(true, endpoint.releaseRequest(reservation));
        assertEquals(false, endpoint.releaseRequest(reservation));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(0, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(endpoint.resourceSnapshot()));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());

        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertFalse(isConfirmed(1L), "late status cannot recreate expired local ownership");
        assertEquals(
                9_500,
                endpoint.selectionSnapshot().usage().hardKvAvailable(),
                "physical KV remains Engine-reported");
    }

    @Test
    void confirmedCleanupPreservesWorkerCapacity() {
        DecodeEndpoint.RequestRegistration reservation = reserve(1L, 500, 700, 0);
        updateStatus(Map.of("1", task(1L, TaskPhase.RUNNING, 500)), Map.of(), 9_500);
        assertEquals(1, confirmedCount());

        assertEquals(true, endpoint.releaseRequest(reservation));
        assertEquals(false, endpoint.releaseRequest(reservation));
        assertEquals(0, confirmedCount());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(9_500, endpoint.selectionSnapshot().usage().hardKvAvailable());
        assertEquals(500, endpoint.selectionSnapshot().realKvUsed());
    }

    @Test
    void staleEndpointAndReservationIdentityCannotExpireNewRequest() {
        DecodeEndpoint.RequestRegistration original = reserve(1L, 500, 700, 0);
        assertEquals(
                false,
                endpoint.releaseRequest(
                        new DecodeEndpoint.RequestRegistration(
                                original.endpointGenerationId() + 1L, String.valueOf(1L))));
        assertEquals(
                false,
                endpoint.releaseRequest(
                        new DecodeEndpoint.RequestRegistration(
                                original.endpointGenerationId(), String.valueOf(1L))));
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());

        assertEquals(true, endpoint.releaseRequest(original));
        endpoint.evictExpiredRequests(-1L, requestId -> false);
        DecodeEndpoint.RequestRegistration replacement = reserve(1L, 300, 450, 0);
        assertEquals(false, endpoint.releaseRequest(original));
        assertEquals(300, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
        assertEquals(true, endpoint.releaseRequest(replacement));
    }

    private void updateStatus(
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished,
            long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    private DecodeEndpoint.RequestRegistration reserve(
            long requestId, long hardKv, long expectedKv, int priority) {
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            assertTrue(selectedEndpoint != null);
            DecodeEndpoint.RequestRegistration reservation =
                    EndpointTestSupport.registerDecode(
                            endpoint, requestId, hardKv, expectedKv, priority);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private boolean isConfirmed(long requestId) {
        return endpoint.resourceSnapshot().requests().values().stream()
                .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                .anyMatch(view -> view.requestId().equals(String.valueOf(requestId)));
    }

    private long confirmedCount() {
        return endpoint.resourceSnapshot().requests().values().stream()
                .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                .count();
    }

    private static TaskInfo task(long requestId, TaskPhase phase, long inputLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setPhase(phase);
        task.setInputLength(inputLength);
        task.setErrorCode(0);
        return task;
    }
}
