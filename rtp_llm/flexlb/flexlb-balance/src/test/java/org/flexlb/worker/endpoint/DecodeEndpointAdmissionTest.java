package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
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
    @Test
    void reservationImmediatelyConsumesRequestAndKvCapacity() {
        var endpoint = endpoint();
        var limit = new AdmissionCapacity(1, 100);
        var first = reserve(endpoint, "a", 500, 700, limit);
        assertEquals(RegistrationStatus.REGISTERED, first.status());
        assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
        assertEquals(9500L, endpoint.selectionSnapshot().usage().hardKvAvailable());
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
                endpoint.tryRegisterSelection(before, "b", 100, 100, 50, limit).status());
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
                                        before, "a", 100, 100, 50, new AdmissionCapacity(1, 100));
                            });
            var two =
                    workers.submit(
                            () -> {
                                start.await(5, TimeUnit.SECONDS);
                                return endpoint.tryRegisterSelection(
                                        before, "b", 100, 100, 50, new AdmissionCapacity(1, 100));
                            });
            var first = one.get(5, TimeUnit.SECONDS);
            var second = two.get(5, TimeUnit.SECONDS);
            assertEquals(1, (first.value() == null ? 0 : 1) + (second.value() == null ? 0 : 1));
            assertEquals(1, endpoint.selectionSnapshot().usage().occupiedRequests());
            assertEquals(9900, endpoint.selectionSnapshot().usage().hardKvAvailable());
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
        assertEquals(9800, endpoint.selectionSnapshot().usage().hardKvAvailable());
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

    private static DecodeEndpoint endpoint() {
        var status = EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001);
        var endpoint = EndpointTestSupport.decodeEndpoint(status);
        var response = new WorkerStatusResponse();
        response.setTotalKvCacheTokens(10000L);
        response.setAvailableKvCacheTokens(10000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
        return endpoint;
    }

    private static WorkerEndpoint.Registration<RequestRegistration> reserve(
            DecodeEndpoint endpoint,
            String id,
            long hard,
            long expected,
            AdmissionCapacity capacity) {
        return endpoint.tryRegisterSelection(
                endpoint.selectionSnapshot(), id, hard, expected, 50, capacity);
    }
}
