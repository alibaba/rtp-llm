package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.DecodeTaskPhase;
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
 * Checks local request phases and immutable views as worker observations replace local reservations.
 */
class DecodeEndpointLayeredViewTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;
    private final Map<Long, DecodeEndpoint.RequestRegistration> reservations = new HashMap<>();

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint =
                EndpointTestSupport.decode(
                        status,
                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                EndpointTestSupport.noopEventSink()));
        updateStatus(Map.of(), Map.of(), 20_000);
    }

    @Test
    void routingViewKeepsEndpointAddressAcrossCapacityChanges() {
        var before = endpoint.selectionSnapshot();
        assertEquals(endpoint.ipPort(), before.address());
        org.junit.jupiter.api.Assertions.assertSame(before, endpoint.selectionSnapshot());
        reserve(77L, 100, 100, 30);
        var after = endpoint.selectionSnapshot();
        org.junit.jupiter.api.Assertions.assertNotSame(before, after);
        assertEquals(endpoint.ipPort(), after.address());
        assertTrue(after.admissionVersion() > before.admissionVersion());
    }

    // ==================== calibrate: layer split + inheritance ====================

    @Test
    void calibrate_splitsConfirmedIntoAcceptedAndRunningLayers() {
        reserve(1L, 500, 508, 30);
        reserve(2L, 500, 508, 40);

        TaskInfo accepted = runningTask(1L, TaskPhase.KV_ALLOCATED, 256);
        TaskInfo running = runningTask(2L, TaskPhase.RUNNING, 512);
        updateStatus(Map.of("1", accepted, "2", running), null, 10_000);

        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        snapshotRequest.phase()
                                                == org.flexlb.worker.status.DecodeTaskPhase
                                                        .ACCEPTED_NOT_RUNNING)
                        .count());
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        snapshotRequest.phase()
                                                == org.flexlb.worker.status.DecodeTaskPhase.RUNNING)
                        .count());
        assertEquals(
                2,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                        .count());
        assertTrue(isConfirmed(1L));
        assertTrue(isConfirmed(2L));

        // Layered view inherits priority from the shadow entry
        // removed this round; KV is the reported inputLength estimate.
        DecodeResources.DecodeRequestView acceptedView = confirmedView(1L);
        assertEquals(DecodeTaskPhase.ACCEPTED_NOT_RUNNING, acceptedView.phase());
        assertEquals(30, acceptedView.priority());
        assertEquals(256, acceptedView.kvTokens());

        DecodeResources.DecodeRequestView runningView = confirmedView(2L);
        assertEquals(DecodeTaskPhase.RUNNING, runningView.phase());
        assertEquals(40, runningView.priority());
    }

    @Test
    void calibrate_promotesAcceptedToRunningOnRefresh() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        snapshotRequest.phase()
                                                == org.flexlb.worker.status.DecodeTaskPhase
                                                        .ACCEPTED_NOT_RUNNING)
                        .count());

        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);

        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        snapshotRequest.phase()
                                                == org.flexlb.worker.status.DecodeTaskPhase
                                                        .ACCEPTED_NOT_RUNNING)
                        .count());
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        snapshotRequest.phase()
                                                == org.flexlb.worker.status.DecodeTaskPhase.RUNNING)
                        .count());
        // Identity fields stay from first sight.
        assertEquals(30, confirmedView(1L).priority());
    }

    // ==================== calibrate: registry follows the reports ====================

    @Test
    void calibrate_dropsEntriesNoLongerReported() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertTrue(isConfirmed(1L));

        // Next report no longer lists the request as confirmed — this is the
        // release-confirmation signal the accepted-eviction wait polls for.
        updateStatus(Map.of(), null, 10_000);

        assertFalse(isConfirmed(1L));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        snapshotRequest.phase()
                                                == org.flexlb.worker.status.DecodeTaskPhase
                                                        .ACCEPTED_NOT_RUNNING)
                        .count());
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                        .count());
    }

    @Test
    void calibrate_finishedRemovesTrackedEntry() {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)), null, 10_000);
        assertTrue(isConfirmed(1L));

        // Same round lists it both running and finished: finished wins.
        TaskInfo finished = runningTask(1L, TaskPhase.RUNNING, 256);
        finished.setErrorCode(0);
        updateStatus(
                Map.of("1", runningTask(1L, TaskPhase.RUNNING, 256)),
                Map.of("1", finished),
                10_000);

        assertFalse(isConfirmed(1L));
    }

    @Test
    void evictExpiredRequests_purgesStaleTrackedEntries() throws InterruptedException {
        reserve(1L, 500, 508, 30);
        updateStatus(Map.of("1", runningTask(1L, TaskPhase.KV_ALLOCATED, 256)), null, 10_000);
        assertTrue(isConfirmed(1L));

        long versionBefore = endpoint.selectionSnapshot().admissionVersion();
        endpoint.evictExpiredRequests(-1, requestId -> false);

        assertFalse(isConfirmed(1L));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                        .count(),
                "TTL removal clears the local request record");
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertTrue(endpoint.selectionSnapshot().admissionVersion() > versionBefore);
    }

    // ==================== accounting invariants unchanged (iron rule 5) ====================

    // ==================== token-fenced weak-ACK preemption ====================

    // ==================== snapshot capture: request phases ====================

    // ==================== helpers ====================

    @Test
    void exactOwnershipQueryRetainsEngineCapacityUntilWorkerCompletion() {
        var reservation = reserve(701L, 400, 408, 30);
        assertTrue(endpoint.registrationState(reservation).registered());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(400, endpoint.selectionSnapshot().usage().hardReservedKvTokens());

        updateStatus(Map.of("701", runningTask(701L, TaskPhase.RUNNING, 400)), null, 19_600);
        assertTrue(endpoint.registrationState(reservation).registered());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(19_600, endpoint.selectionSnapshot().usage().hardKvAvailable());

        updateStatus(Map.of(), Map.of("701", runningTask(701L, TaskPhase.RUNNING, 400)), 20_000);
        assertFalse(endpoint.registrationState(reservation).registered());
        assertFalse(endpoint.registrationState(reservation).registered());
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void absentOldReservationIsCompleteWithoutTouchingTheReplacement() {
        var replacement = reserve(702L, 400, 408, 30);
        var old =
                new DecodeEndpoint.RequestRegistration(
                        replacement.endpointGenerationId(),
                        String.valueOf(replacement.requestId()));
        assertEquals(false, endpoint.releaseRequest(old));
        assertFalse(endpoint.registrationState(old).registered());
        assertEquals(400, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
        assertEquals(true, endpoint.releaseRequest(replacement));
        assertEquals(false, endpoint.releaseRequest(replacement));
        assertFalse(endpoint.registrationState(replacement).registered());
        assertEquals(0, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
    }

    private DecodeResources.DecodeRequestView confirmedView(long requestId) {
        return endpoint.resourceSnapshot().requests().values().stream()
                .filter(snapshotRequest -> snapshotRequest.phase().isEngineConfirmed())
                .filter(view -> view.requestId().equals(String.valueOf(requestId)))
                .findFirst()
                .orElseThrow(() -> new AssertionError("request " + requestId + " not tracked"));
    }

    private void updateStatus(
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished,
            long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(20_000L);
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

    private static TaskInfo runningTask(long requestId, TaskPhase phase, long inputLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        task.setPhase(phase);
        task.setInputLength(inputLength);
        return task;
    }
}
