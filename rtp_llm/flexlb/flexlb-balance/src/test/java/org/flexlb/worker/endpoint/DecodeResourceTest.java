package org.flexlb.worker.endpoint;

import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint.RequestRegistration;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.WorkerEndpoint.RegistrationStatus;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Map;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Checks that worker reports, confirmed releases and orphan cleanup count each request once.
 */
class DecodeResourceTest {
    @ParameterizedTest
    @EnumSource(TaskPhase.class)
    void workerReportTransfersCountWithoutDoubleCharging(TaskPhase phase) {
        var endpoint = endpoint(new CapacityNotifier());
        var reservation = reserve(endpoint, "a");
        report(endpoint, Map.of("a", task("a", phase)), Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(10000, endpoint.selectionSnapshot().usage().hardKvAvailable());
        assertSame(reservation, endpoint.resourceSnapshot().requests().get("a").reservation());
        report(endpoint, Map.of("a", task("a", phase)), Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        report(endpoint, Map.of(), Map.of("a", task("a", phase)));
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
        assertFalse(endpoint.releaseRequest(reservation));
    }

    @Test
    void unknownEngineRequestsConsumeSlotsWithoutCreatingLocalRecords() {
        var endpoint = endpoint(new CapacityNotifier());
        report(endpoint, Map.of("other", task("other", TaskPhase.RUNNING)), Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
    }

    @Test
    void workerPhaseChangesDoNotRestoreLocalKvReservation() {
        var endpoint = endpoint(new CapacityNotifier());
        reserve(endpoint, "a");
        report(endpoint, Map.of("a", task("a", TaskPhase.RECEIVED)), Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(10000, endpoint.selectionSnapshot().usage().hardKvAvailable());
        report(endpoint, Map.of("a", task("a", TaskPhase.KV_ALLOCATED)), Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(10000, endpoint.selectionSnapshot().usage().hardKvAvailable());
        report(endpoint, Map.of("a", task("a", TaskPhase.RECEIVED)), Map.of());
        assertEquals(10000, endpoint.selectionSnapshot().usage().hardKvAvailable());
    }

    @Test
    void confirmedCleanupLeavesCapacityToWorkerReports() {
        var endpoint = endpoint(new CapacityNotifier());
        var a = reserve(endpoint, "a");
        var active = Map.of("a", task("a", TaskPhase.RUNNING));
        report(endpoint, active, Map.of());
        assertTrue(endpoint.releaseRequest(a));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        report(endpoint, active, Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertFalse(endpoint.releaseRequest(a));
        report(endpoint, Map.of(), Map.of());
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
        var replacement = reserve(endpoint, "a");
        assertNotSame(a, replacement);
        assertFalse(endpoint.releaseRequest(a));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void finishedRemovesRegistrationWithoutCorrectingReportedCapacity() {
        var endpoint = endpoint(new CapacityNotifier());
        reserve(endpoint, "a");
        var task = task("a", TaskPhase.RUNNING);
        report(endpoint, Map.of("a", task), Map.of("a", task));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertTrue(endpoint.resourceSnapshot().requests().isEmpty());
        report(endpoint, Map.of("a", task), Map.of());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void fullAbsenceRemovesOnlyRequestsPreviouslyReportedByWorker() {
        var endpoint = endpoint(new CapacityNotifier());
        var a = reserve(endpoint, "a");
        report(endpoint, Map.of("a", task("a", TaskPhase.RUNNING)), Map.of());
        var b = reserve(endpoint, "b");
        report(endpoint, Map.of(), Map.of());
        assertFalse(endpoint.registrationState(a).registered());
        assertTrue(endpoint.registrationState(b).registered());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void orphanSweepRechecksRecordAfterSchedulerLookup() {
        var endpoint = endpoint(new CapacityNotifier());
        var old = reserve(endpoint, "a");
        var replacement = new RequestRegistration[1];
        endpoint.evictExpiredRequests(
                -1,
                id -> {
                    assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                    assertTrue(endpoint.releaseRequest(old));
                    replacement[0] = reserve(endpoint, "a");
                    return false;
                });
        assertTrue(endpoint.registrationState(replacement[0]).registered());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void releaseNotifiesOnceOutsideStateLock() {
        var notifier = new CapacityNotifier();
        var endpoint = endpoint(notifier);
        var a = reserve(endpoint, "a");
        var notifications = new AtomicInteger();
        notifier.addListener(
                key -> {
                    assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                    assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
                    notifications.incrementAndGet();
                });
        assertTrue(endpoint.releaseRequest(a));
        assertFalse(endpoint.releaseRequest(a));
        assertEquals(1, notifications.get());
    }

    @Test
    void orphanSweepOnlyQueriesExpiredRequestsAndRechecksLatestObservation() {
        var endpoint = endpoint(new CapacityNotifier());
        var refreshed = reserve(endpoint, "refreshed");
        var expired = reserve(endpoint, "expired");
        var retained = reserve(endpoint, "retained");
        var fresh = reserve(endpoint, "fresh");
        for (var id : java.util.List.of("refreshed", "expired", "retained")) {
            ageRequest(endpoint, id);
        }
        var queried = new java.util.HashSet<String>();
        int removed =
                endpoint.evictExpiredRequests(
                        30_000L,
                        id -> {
                            assertFalse(endpoint.stateLock().isHeldByCurrentThread());
                            queried.add(id);
                            if (id.equals("refreshed")) {
                                report(endpoint, Map.of(id, task(id, TaskPhase.RUNNING)), Map.of());
                            }
                            return id.equals("retained");
                        });
        assertEquals(java.util.Set.of("refreshed", "expired", "retained"), queried);
        assertEquals(1, removed);
        assertTrue(endpoint.registrationState(refreshed).registered());
        assertTrue(endpoint.registrationState(retained).registered());
        assertTrue(endpoint.registrationState(fresh).registered());
        assertFalse(endpoint.registrationState(expired).registered());
        assertEquals(3, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void statusPublishesOneCapacityNotification() {
        var notifier = new CapacityNotifier();
        var endpoint = endpoint(notifier);
        var notifications = new AtomicInteger();
        notifier.addListener(key -> notifications.incrementAndGet());
        report(endpoint, Map.of(), Map.of());
        assertEquals(1, notifications.get());
    }

    @Test
    void concurrentReleaseAndFinishedNeverSubtractAnotherRequest() throws Exception {
        var endpoint = endpoint(new CapacityNotifier());
        var a = reserve(endpoint, "a");
        reserve(endpoint, "b");
        var start = new CyclicBarrier(2);
        try (var workers = Executors.newFixedThreadPool(2)) {
            var release =
                    workers.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                return endpoint.releaseRequest(a);
                            });
            var finished =
                    workers.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                report(
                                        endpoint,
                                        Map.of(),
                                        Map.of("a", task("a", TaskPhase.RUNNING)));
                                return true;
                            });
            release.get(5, TimeUnit.SECONDS);
            finished.get(5, TimeUnit.SECONDS);
        }
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(9900, endpoint.selectionSnapshot().usage().hardKvAvailable());
        assertTrue(endpoint.resourceSnapshot().requests().containsKey("b"));
    }

    private static void ageRequest(DecodeEndpoint endpoint, String id) {
        endpoint.stateLock().lock();
        try {
            var records =
                    (Map<?, ?>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    endpoint, "registeredRequests");
            org.springframework.test.util.ReflectionTestUtils.setField(
                    records.get(id), "observedAtMs", System.currentTimeMillis() - 60_000L);
        } finally {
            endpoint.stateLock().unlock();
        }
    }

    private static DecodeEndpoint endpoint(CapacityNotifier notifier) {
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001);
        var endpoint = new DecodeEndpoint(status, mock(RequestRepository.class), notifier);
        report(endpoint, Map.of(), Map.of());
        return endpoint;
    }

    private static void report(
            DecodeEndpoint endpoint, Map<String, TaskInfo> active, Map<String, TaskInfo> finished) {
        var response = new WorkerStatusResponse();
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(10000L);
        response.setRunningTaskInfo(active);
        response.setFinishedTaskInfo(finished);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    private static RequestRegistration reserve(DecodeEndpoint endpoint, String id) {
        var result =
                endpoint.tryRegisterSelection(
                        endpoint.selectionSnapshot(),
                        id,
                        100,
                        200,
                        50,
                        new AdmissionCapacity(0, 100));
        assertEquals(RegistrationStatus.REGISTERED, result.status());
        return result.value();
    }

    private static TaskInfo task(String id, TaskPhase phase) {
        var task = new TaskInfo();
        task.setRequestId(id);
        task.setPhase(phase);
        task.setInputLength(100);
        return task;
    }
}
