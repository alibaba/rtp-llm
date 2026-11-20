package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.DecodeTaskPhase;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * Checks reservation, worker feedback and exact cleanup against one capacity record.
 */
class DecodeEndpointTest {
    private WorkerStatus status;
    private DecodeEndpoint endpoint;

    private final Map<Long, DecodeEndpoint.RequestRegistration> reservations = new HashMap<>();

    @Test
    void endpointDoesNotOwnPreemptionAttemptsOrDedicatedCapacityWaiters() {
        var fields =
                java.util.Arrays.stream(DecodeEndpoint.class.getDeclaredFields())
                        .map(java.lang.reflect.Field::getName)
                        .toList();
        assertFalse(fields.contains("preemptionAttempts"));
        assertFalse(fields.contains("preemptionRegistrations"));
        assertFalse(fields.contains("engineDispatchCapacityListeners"));
        assertFalse(fields.contains("priorityPreemptionHeldKv"));
    }

    @BeforeEach
    void setUp() {
        status = EndpointTestSupport.workerStatus(RoleType.DECODE, "10.0.0.1", 8080, 8081);
        endpoint = EndpointTestSupport.decode(status);
    }

    @Test
    void registrationIdentityDoesNotRetainMutableNotificationState() {
        for (var field : DecodeEndpoint.RequestRegistration.class.getDeclaredFields()) {
            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) continue;
            assertTrue(java.lang.reflect.Modifier.isFinal(field.getModifiers()), field.getName());
            assertFalse(WorkerEventSink.class.isAssignableFrom(field.getType()), field.getName());
        }
    }

    @Test
    void finalConfirmationAssociatesFeedbackWithoutConfirmingWorkerCapacity() {
        var registration = reserve(701L, 100L, 200L);
        var before = endpoint.selectionSnapshot().usage();
        var events = new java.util.ArrayList<WorkerRequestEvent>();
        assertEquals(
                WorkerEndpoint.CommitStatus.COMMITTED,
                endpoint.confirmSelection(
                        registration,
                        event -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            events.add(event);
                        }));
        assertEquals(before, endpoint.selectionSnapshot().usage());
        assertEquals(
                DecodeResources.RegistrationState.UNCONFIRMED,
                endpoint.registrationState(registration));
        assertEquals(
                WorkerEndpoint.CommitStatus.INVALID_REGISTRATION,
                endpoint.confirmSelection(registration, events::add));
        updateStatus(Map.of("701", task(701L)), Map.of(), 10000L);
        assertEquals(1, events.size());
        assertTrue(events.getFirst().registration() == registration);
        assertTrue(events.getFirst().endpoint() == endpoint);
        assertEquals(
                DecodeResources.RegistrationState.CONFIRMED,
                endpoint.registrationState(registration));
    }

    @Test
    void finishedBeforeFinalConfirmationRejectsTheRemovedRegistration() {
        var registration = reserve(702L, 100L, 200L);
        updateStatus(Map.of(), Map.of("702", task(702L)), 10000L);
        var events = new java.util.ArrayList<WorkerRequestEvent>();
        assertEquals(
                WorkerEndpoint.CommitStatus.INVALID_REGISTRATION,
                endpoint.confirmSelection(registration, events::add));
        assertTrue(events.isEmpty());
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void delayedFeedbackKeepsTheOriginalReceiverWhenRequestIdIsReused() {
        var first = reserve(703L, 100L, 200L);
        var oldEvents = new java.util.ArrayList<WorkerRequestEvent>();
        var newEvents = new java.util.ArrayList<WorkerRequestEvent>();
        assertEquals(
                WorkerEndpoint.CommitStatus.COMMITTED,
                endpoint.confirmSelection(first, oldEvents::add));
        var response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of("703", task(703L)));
        response.setAvailableKvCacheTokens(10000L);
        response.setTotalKvCacheTokens(10000L);
        var delayed = EndpointTestSupport.applyStatus(endpoint, response);
        var second = reserve(703L, 100L, 200L);
        assertEquals(
                WorkerEndpoint.CommitStatus.COMMITTED,
                endpoint.confirmSelection(second, newEvents::add));
        delayed.run();
        assertEquals(1, oldEvents.size());
        assertTrue(oldEvents.getFirst().registration() == first);
        assertTrue(newEvents.isEmpty());
        assertEquals(
                DecodeResources.RegistrationState.UNCONFIRMED, endpoint.registrationState(second));
        assertEquals(
                WorkerEndpoint.CommitStatus.INVALID_REGISTRATION,
                endpoint.confirmSelection(first, newEvents::add));
    }

    @Test
    void notificationUsesTheReceiverPresentWhenStatusWasCaptured() {
        var registration = reserve(704L, 100L, 200L);
        var response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("704", task(704L)));
        response.setAvailableKvCacheTokens(10000L);
        response.setTotalKvCacheTokens(10000L);
        var beforeBinding = EndpointTestSupport.applyStatus(endpoint, response);
        var events = new java.util.ArrayList<WorkerRequestEvent>();
        assertEquals(
                WorkerEndpoint.CommitStatus.COMMITTED,
                endpoint.confirmSelection(registration, events::add));
        beforeBinding.run();
        assertTrue(events.isEmpty());

        EndpointTestSupport.applyStatus(endpoint, response).run();
        assertEquals(1, events.size());
        assertTrue(events.getFirst().registration() == registration);
        assertEquals(RequestEventType.ALLOCATED, events.getFirst().type());
    }

    @ParameterizedTest
    @EnumSource(TaskPhase.class)
    void decodeConfirmationTransfersRequestAndKvReservationTogether(TaskPhase phase) {
        var reservation = reserve(700L, 100L, 200L);
        var active = task(700L);
        active.setInputLength(100L);
        active.setPhase(phase);
        var response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("700", active));
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(9900L);
        for (int observation = 0; observation < 2; observation++) {
            EndpointTestSupport.applyStatus(endpoint, response).run();
            assertEquals(
                    org.flexlb.worker.endpoint.DecodeResources.RegistrationState.CONFIRMED,
                    endpoint.registrationState(reservation));
            assertEquals(
                    0,
                    endpoint.resourceSnapshot().requests().values().stream()
                            .filter(
                                    snapshotRequest ->
                                            !org.flexlb.worker.status.WorkerStatusTestSupport
                                                    .isEngineConfirmed(snapshotRequest.phase()))
                            .count());
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
            assertEquals(0L, endpoint.selectionSnapshot().usage().hardReservedKvTokens());
            assertEquals(100L, endpoint.selectionSnapshot().usage().expectedKvUsed());
        }
    }

    @Test
    void admissionAfterSummaryCaptureDoesNotChangeTheCapturedOccupancy() {
        var once = new java.util.concurrent.atomic.AtomicBoolean();
        var lock =
                new java.util.concurrent.locks.ReentrantLock() {
                    @Override
                    public void unlock() {
                        boolean captureReleased = getHoldCount() == 1;
                        super.unlock();
                        if (captureReleased && once.compareAndSet(false, true)) {
                            assertTrue(
                                    org.flexlb.balance.schedule.SelectionTestSupport
                                                    .registerQueuedDecode(
                                                            endpoint, "late", 100L, 150L, 50)
                                                    .status()
                                            == WorkerEndpoint.RegistrationStatus.REGISTERED);
                        }
                    }
                };
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "stateLock", lock);
        var beforeAdmission = endpoint.resourceSnapshot();
        assertTrue(once.get());
        assertEquals(0, beforeAdmission.requests().size());
        var afterAdmission = endpoint.resourceSnapshot();
        assertEquals(1, afterAdmission.requests().get("late").placementRelease().requests());
        assertEquals(100L, afterAdmission.requests().get("late").placementRelease().hardKvTokens());
        assertEquals(0, beforeAdmission.requests().size());
        assertEquals(
                afterAdmission.requests().get("late").placementRelease(),
                endpoint.resourceSnapshot().requests().get("late").placementRelease());
    }

    @ParameterizedTest
    @EnumSource(TaskPhase.class)
    void confirmationPreservesExactReservationAndDerivesPhaseFromWorker(TaskPhase phase) {
        var sink = EndpointTestSupport.noopEventSink();
        endpoint = EndpointTestSupport.decode(status);
        var reservation = reserve(100L, 500L, 500L);
        var route = org.mockito.Mockito.mock(org.flexlb.balance.schedule.RequestRoute.class);
        assertEquals(
                WorkerEndpoint.CommitStatus.COMMITTED,
                endpoint.confirmSelection(
                        reservation, event -> sink.onWorkerRequestEvent(route, event)));
        TaskInfo active = task(100L);
        active.setPhase(phase);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("100", active));
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(10000L);

        EndpointTestSupport.applyStatus(endpoint, response).run();
        verify(sink)
                .onWorkerRequestEvent(
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.argThat(
                                e ->
                                        e.endpoint() == endpoint
                                                && e.registration().equals(reservation)
                                                && e.type()
                                                        == org.flexlb.worker.endpoint
                                                                .RequestEventType.ALLOCATED));
        assertEquals(
                org.flexlb.worker.endpoint.DecodeResources.RegistrationState.CONFIRMED,
                endpoint.registrationState(reservation));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
        assertEquals(
                phase == TaskPhase.RUNNING
                        ? DecodeTaskPhase.RUNNING
                        : DecodeTaskPhase.ACCEPTED_NOT_RUNNING,
                endpoint.resourceSnapshot().requests().get(String.valueOf(100L)).phase());

        EndpointTestSupport.applyStatus(endpoint, response).run();
        verify(sink, times(2))
                .onWorkerRequestEvent(
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.argThat(
                                e ->
                                        e.endpoint() == endpoint
                                                && e.registration().equals(reservation)
                                                && e.type()
                                                        == org.flexlb.worker.endpoint
                                                                .RequestEventType.ALLOCATED));
        assertEquals(
                org.flexlb.worker.endpoint.DecodeResources.RegistrationState.CONFIRMED,
                endpoint.registrationState(reservation));

        WorkerStatusResponse finished = new WorkerStatusResponse();
        finished.setFinishedTaskInfo(Map.of("100", task(100L)));
        EndpointTestSupport.applyStatus(endpoint, finished).run();
        verify(sink)
                .onWorkerRequestEvent(
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.argThat(
                                e ->
                                        e.endpoint() == endpoint
                                                && e.registration().equals(reservation)
                                                && e.type()
                                                        == org.flexlb.worker.endpoint
                                                                .RequestEventType.COMPLETED
                                                && e.errorCode() == 0L));
        assertFalse(endpoint.resourceSnapshot().requests().containsKey(String.valueOf(100L)));
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
    }

    @Test
    void reserve_updatesSnapshotAndInflight() {
        updateStatus(null, null, 10000);
        reserve(100L, 500, 500);
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
        assertEquals(
                9500,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
    }

    @Test
    void release_decrementsInflight() {
        reserve(100L, 500, 500);
        reserve(101L, 300, 300);
        release(100L);

        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
    }

    @Test
    void release_unknownRequestId_noEffect() {
        reserve(100L, 500, 500);
        release(999L);
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
    }

    @Test
    void release_neverGoesNegative() {
        reserve(100L, 100, 100);
        release(100L);
        release(100L);
        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
        assertEquals(
                0,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
    }

    @Test
    void confirmedCleanupWaitsForWorkerCapacityUpdate() {
        var reservation = reserve(100L, 500L, 500L);
        var running = task(100L);
        running.setPhase(TaskPhase.RUNNING);
        running.setInputLength(500L);
        var response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of("100", running));
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(9_500L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        assertEquals(
                9_500L,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));

        endpoint.releaseRequest(reservation);
        assertEquals(
                9_500L,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        endpoint.releaseRequest(reservation);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        assertEquals(
                9_500L,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());

        response.setRunningTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        assertEquals(
                10_000L,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
        assertEquals(0L, endpoint.selectionSnapshot().realKvUsed());
    }

    @Test
    void calibrate_kvAllocatedReleasesFromInflight() {
        reserve(100L, 500, 500);

        TaskInfo running = task(100L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("100", running), null, 10000);

        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
        assertEquals(
                10000,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
    }

    @Test
    void calibrate_finishedFailureReleasesFromInflight() {
        reserve(100L, 500, 500);

        TaskInfo failed = task(100L);
        failed.setErrorCode(1);
        failed.setErrorMessage("timeout");
        updateStatus(null, Map.of("100", failed), 10000);

        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
    }

    @Test
    void calibrate_finishedSuccessReleasesIfStillPresent() {
        reserve(100L, 500, 500);

        TaskInfo success = task(100L);
        success.setErrorCode(0);
        updateStatus(null, Map.of("100", success), 10000);

        assertEquals(
                0,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
    }

    @Test
    void calibrate_updatesReportedKvAvailable() {
        reserve(100L, 500, 500);
        updateStatus(null, null, 10000);

        assertEquals(
                9500,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
    }

    @Test
    void availableKvTokens_accountsForReservations() {
        updateStatus(null, null, 10000);

        reserve(100L, 3000, 3000);
        reserve(101L, 2000, 2000);

        assertEquals(
                5000,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
    }

    @Test
    void ipPort_format() {
        assertEquals("10.0.0.1:8080@0", endpoint.ipPort());
    }

    // ==================== PR-C: getEngineLoad O(1) + queuedPhaseCount drift ===========

    @Test
    void duplicateSelectionDoesNotDoubleCountQueuedWork() {
        reserve(1L, 100, 100);
        var duplicate =
                endpoint.tryRegisterSelection(
                        "1",
                        endpoint.selectionSnapshot(),
                        100L,
                        100L,
                        50,
                        new DecodeResources.AdmissionCapacity(0L, 100L));
        assertEquals(WorkerEndpoint.RegistrationStatus.ALREADY_REGISTERED, duplicate.status());
        assertEquals(
                1,
                endpoint.resourceSnapshot().requests().values().stream()
                        .filter(
                                snapshotRequest ->
                                        !org.flexlb.worker.status.WorkerStatusTestSupport
                                                .isEngineConfirmed(snapshotRequest.phase()))
                        .count());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    // ==================== P1-6: evictExpiredRequests counter (PR-C) ====================

    /**
     * Expiration removes the queued phase and all incremental counters together.
     */

    /**
     * Directly mutate the private counter to simulate drift.
     */
    private void updateStatus(
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished,
            long availableKvCacheTokens) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setAvailableKvCacheTokens(availableKvCacheTokens);
        response.setTotalKvCacheTokens(availableKvCacheTokens);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    private DecodeEndpoint.RequestRegistration reserve(
            long requestId, long hardKv, long expectedKv) {
        {
            WorkerEndpoint selectedEndpoint = endpoint;

            assertNotNull(selectedEndpoint);
            DecodeEndpoint.RequestRegistration reservation =
                    EndpointTestSupport.registerDecode(endpoint, requestId, hardKv, expectedKv, 0);
            reservations.put(requestId, reservation);
            return reservation;
        }
    }

    private void release(long requestId) {
        DecodeEndpoint.RequestRegistration reservation = reservations.get(requestId);
        if (reservation != null) {
            endpoint.releaseRequest(reservation);
        }
    }

    private TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(String.valueOf(requestId));
        return task;
    }
}
