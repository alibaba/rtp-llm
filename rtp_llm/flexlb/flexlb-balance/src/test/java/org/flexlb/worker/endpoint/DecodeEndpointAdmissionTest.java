package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.DecodeEndpoint.CapacityFailure;
import org.flexlb.worker.endpoint.DecodeEndpoint.RequestRegistration;
import org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity;
import org.flexlb.worker.endpoint.DecodeResources.Capacity;
import org.flexlb.worker.endpoint.WorkerEndpoint.RegistrationStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Admission charges request slots and KV together; stale observations and repeated cleanup cannot oversubscribe them.
 */
class DecodeEndpointAdmissionTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "10,false",
        "50,false",
        "90,false",
        "10,true",
        "50,true",
        "90,true"
    })
    void admissionEvaluationClassifiesOccupancyWithoutCopyingEveryRequest(
            int occupiedPriority, boolean confirmed) {
        var endpoint = org.mockito.Mockito.spy(endpoint());
        var limit = new AdmissionCapacity(1, 100);
        assertEquals(
                RegistrationStatus.REGISTERED,
                endpoint.tryRegisterSelection(
                                "1",
                                endpoint.selectionSnapshot(),
                                100,
                                200,
                                occupiedPriority,
                                limit)
                        .status());
        if (confirmed) {
            var task = new org.flexlb.worker.status.TaskInfo();
            task.setRequestId("1");
            task.setInputLength(100L);
            task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
            var response = new WorkerStatusResponse();
            response.setTotalKvCacheTokens(10000L);
            response.setAvailableKvCacheTokens(9900L);
            response.setRunningTaskInfo(java.util.Map.of("1", task));
            EndpointTestSupport.applyStatus(endpoint, response).run();
        }
        var evaluation = endpoint.evaluateSelection(100, 200, 50, limit);
        assertEquals(new Capacity(1, 0, 0), evaluation.deficit());
        assertEquals(1L, evaluation.snapshot().usage().occupiedRequests());
        assertEquals(confirmed ? 100L : 200L, evaluation.snapshot().usage().expectedKvUsed());
        assertEquals(
                occupiedPriority < 50
                        ? CapacityFailure.INSUFFICIENT_CAPACITY
                        : occupiedPriority == 50
                                ? CapacityFailure.BLOCKED_BY_SAME_PRIORITY
                                : CapacityFailure.BLOCKED_BY_HIGHER_PRIORITY,
                evaluation.failure());
        org.mockito.Mockito.verify(endpoint, org.mockito.Mockito.never()).resourceSnapshot();
    }

    @Test
    void availableCapacityHasNoFailure() {
        var evaluation = endpoint().evaluateSelection(100, 200, 50, new AdmissionCapacity(1, 100));
        assertEquals(Capacity.NONE, evaluation.deficit());
        assertEquals(CapacityFailure.NONE, evaluation.failure());
    }

    @Test
    void workerOccupancyWithoutLocalPriorityHasUnknownBlocker() {
        var endpoint = endpoint();
        var response = new WorkerStatusResponse();
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(0L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        var evaluation = endpoint.evaluateSelection(100, 200, 50, new AdmissionCapacity(10, 100));
        assertFalse(evaluation.deficit().isZero());
        assertEquals(CapacityFailure.UNKNOWN_BLOCKER, evaluation.failure());
    }

    @Test
    void reservationImmediatelyConsumesRequestAndKvCapacity() {
        var endpoint = endpoint();
        var limit = new AdmissionCapacity(1, 100);
        var first = reserve(endpoint, "a", 500, 700, limit);
        assertEquals(RegistrationStatus.REGISTERED, first.status());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(
                9500L,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
        assertEquals(700L, endpoint.selectionSnapshot().realKvUsed());
        assertEquals(
                RegistrationStatus.CAPACITY_FULL, reserve(endpoint, "b", 100, 100, limit).status());
        assertTrue(endpoint.releaseRequest(first.value()));
        assertFalse(endpoint.releaseRequest(first.value()));
        assertEquals(
                RegistrationStatus.REGISTERED, reserve(endpoint, "b", 100, 100, limit).status());
    }

    @Test
    void staleRankingCannotReserveAgainstChangedCapacity() {
        var endpoint = endpoint();
        var before = endpoint.selectionSnapshot();
        var limit = new AdmissionCapacity(2, 100);
        assertEquals(
                RegistrationStatus.REGISTERED, reserve(endpoint, "a", 100, 100, limit).status());
        assertEquals(
                RegistrationStatus.VERSION_CONFLICT,
                endpoint.tryRegisterSelection("b", before, 100, 100, 50, limit).status());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void concurrentReservationsCannotBothConsumeTheLastSlot() throws Exception {
        var endpoint = endpoint();
        var before = endpoint.selectionSnapshot();
        var start = new CyclicBarrier(2);
        try (var workers = Executors.newFixedThreadPool(2)) {
            var one =
                    workers.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                return endpoint.tryRegisterSelection(
                                        "a", before, 100, 100, 50, new AdmissionCapacity(1, 100));
                            });
            var two =
                    workers.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                return endpoint.tryRegisterSelection(
                                        "b", before, 100, 100, 50, new AdmissionCapacity(1, 100));
                            });
            var first = one.get(5, TimeUnit.SECONDS);
            var second = two.get(5, TimeUnit.SECONDS);
            assertEquals(1, (first.value() == null ? 0 : 1) + (second.value() == null ? 0 : 1));
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
            assertEquals(
                    9900,
                    org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                            endpoint.selectionSnapshot().usage()));
        }
    }

    @Test
    void staleHandleCannotReleaseReplacementWithTheSameRequestId() {
        var endpoint = endpoint();
        var limit = new AdmissionCapacity(0, 100);
        var old = reserve(endpoint, "a", 100, 100, limit).value();
        assertTrue(endpoint.releaseRequest(old));
        var current = reserve(endpoint, "a", 200, 300, limit).value();
        assertNotSame(old, current);
        assertFalse(endpoint.releaseRequest(old));
        assertFalse(
                endpoint.releaseRequest(
                        new RequestRegistration(current.endpointGenerationId(), "a")));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(
                9800,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
        assertTrue(endpoint.releaseRequest(current));
    }

    @Test
    void stoppedAdmissionStillAllowsExistingReservationCleanup() {
        var endpoint = endpoint();
        var limit = new AdmissionCapacity(1, 100);
        var first = reserve(endpoint, "a", 100, 100, limit).value();
        endpoint.stopAcceptingRequests();
        assertEquals(
                RegistrationStatus.UNAVAILABLE, reserve(endpoint, "b", 100, 100, limit).status());
        assertTrue(endpoint.releaseRequest(first));
        assertEquals(0, endpoint.selectionSnapshot().usage().occupiedRequests());
    }

    @Test
    void invalidKvPercentageCannotReachAdmissionChecks() {
        for (long percent : new long[] {-1L, 0L, 101L}) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new DecodeResources.AdmissionCapacity(10L, percent));
        }
    }

    @Test
    void physicalAndExpectedDeficitsRequireDifferentVictimReleases() {
        var capacity = new DecodeResources.AdmissionCapacity(0L, 90L);
        var usage = new DecodeResources.CapacityUsage(2L, 1000L, 150L, 200L, 850L);
        assertEquals(
                new DecodeResources.Capacity(0L, 150L, 150L),
                capacity.evaluate(usage, 100L, 200L, Capacity.NONE));
        assertEquals(
                new DecodeResources.Capacity(0L, 100L, 0L),
                capacity.evaluate(usage, 100L, 200L, new DecodeResources.Capacity(1L, 50L, 200L)));
        assertTrue(
                capacity.evaluate(usage, 100L, 200L, new DecodeResources.Capacity(2L, 150L, 250L))
                        .isZero());
    }

    @Test
    void removingEveryVictimCannotMakeAnOversizedOutputAllowanceFit() {
        var capacity = new DecodeResources.AdmissionCapacity(0L, 90L);
        var usage = new DecodeResources.CapacityUsage(1L, 1000L, 500L, 0L, 500L);
        assertFalse(
                capacity.evaluate(usage, 100L, 901L, new DecodeResources.Capacity(1L, 500L, 500L))
                        .isZero());
    }

    @Test
    void overflowingUsageCannotDisappearIntoASaturatedMaximumBudget() {
        var capacity = new DecodeResources.AdmissionCapacity(Long.MAX_VALUE, 100L);
        var usage =
                new DecodeResources.CapacityUsage(
                        Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, 0L, Long.MAX_VALUE);
        assertEquals(
                new DecodeResources.Capacity(1L, 0L, 1L),
                capacity.evaluate(usage, 0L, 1L, Capacity.NONE));
        assertEquals(
                Long.MAX_VALUE / 100L * 90L + Long.MAX_VALUE % 100L * 90L / 100L,
                new DecodeResources.AdmissionCapacity(0L, 90L).kvBudget(Long.MAX_VALUE));
    }

    @Test
    void validKvPercentagePreservesUnknownCapacityBehavior() {
        var usage = new DecodeResources.CapacityUsage(0L, 0L, 0L, 0L, 0L);
        for (long percent : new long[] {1L, 90L, 100L}) {
            var capacity = new DecodeResources.AdmissionCapacity(0L, percent);
            assertTrue(capacity.evaluate(usage, 100L, 200L, Capacity.NONE).isZero());
            assertEquals(percent, capacity.kvBudget(100L));
        }
    }

    @Test
    void unconfirmedCleanupPreservesConfirmedRegistrationAndReportedCapacity() {
        var endpoint = endpoint();
        var registration = reserve(endpoint, "1", 100, 200, new AdmissionCapacity(0, 100)).value();
        confirm(endpoint, "1");
        assertFalse(endpoint.releaseUnconfirmedRequest(registration));
        assertEquals(
                DecodeResources.RegistrationState.CONFIRMED,
                endpoint.registrationState(registration));
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertThrows(IllegalStateException.class, () -> endpoint.rollbackSelection(registration));
        assertTrue(endpoint.releaseRequest(registration));
        assertEquals(
                1,
                endpoint.selectionSnapshot().usage().occupiedRequests(),
                "removing a confirmed registration must not erase reported Worker occupancy");
    }

    @Test
    void absentOldRegistrationCompletesCleanupWithoutReleasingReplacement() {
        var endpoint = endpoint();
        var limit = new AdmissionCapacity(0, 100);
        var old = reserve(endpoint, "1", 100, 200, limit).value();
        assertTrue(endpoint.releaseUnconfirmedRequest(old));
        assertTrue(endpoint.releaseUnconfirmedRequest(old));
        var replacement = reserve(endpoint, "1", 200, 300, limit).value();
        assertTrue(endpoint.releaseUnconfirmedRequest(old));
        assertEquals(
                DecodeResources.RegistrationState.UNCONFIRMED,
                endpoint.registrationState(replacement));
        assertEquals(
                9800,
                org.flexlb.worker.status.WorkerStatusTestSupport.hardKvAvailable(
                        endpoint.selectionSnapshot().usage()));
    }

    @org.junit.jupiter.api.RepeatedTest(20)
    void confirmationAndUnconfirmedCleanupRespectTheSameLock() throws Exception {
        var endpoint = endpoint();
        var registration = reserve(endpoint, "1", 100, 200, new AdmissionCapacity(0, 100)).value();
        var start = new CyclicBarrier(2);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var confirmation =
                    executor.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                confirm(endpoint, "1");
                                return null;
                            });
            var cleanup =
                    executor.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                return endpoint.releaseUnconfirmedRequest(registration);
                            });
            confirmation.get(5, TimeUnit.SECONDS);
            boolean removed = cleanup.get(5, TimeUnit.SECONDS);
            assertEquals(
                    removed
                            ? DecodeResources.RegistrationState.NOT_REGISTERED
                            : DecodeResources.RegistrationState.CONFIRMED,
                    endpoint.registrationState(registration));
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        }
    }

    private static DecodeEndpoint endpoint() {
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001);
        var endpoint = EndpointTestSupport.decodeEndpoint(status);
        var response = new WorkerStatusResponse();
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(10000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        return endpoint;
    }

    private static void confirm(DecodeEndpoint endpoint, String id) {
        var task = new org.flexlb.worker.status.TaskInfo();
        task.setRequestId(id);
        task.setInputLength(100L);
        task.setPhase(org.flexlb.worker.status.TaskPhase.RUNNING);
        var response = new WorkerStatusResponse();
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(9900L);
        response.setRunningTaskInfo(java.util.Map.of(id, task));
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    private static WorkerEndpoint.Registration<RequestRegistration> reserve(
            DecodeEndpoint endpoint,
            String id,
            long hard,
            long expected,
            AdmissionCapacity capacity) {
        return endpoint.tryRegisterSelection(
                id, endpoint.selectionSnapshot(), hard, expected, 50, capacity);
    }
}
