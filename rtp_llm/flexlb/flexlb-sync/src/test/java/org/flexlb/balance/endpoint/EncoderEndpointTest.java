package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RequestPhase;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class EncoderEndpointTest {

    @Test
    @SuppressWarnings("unchecked")
    void retirementWaitsForRegistrationAlreadyInsideTheGenerationGate() throws Exception {
        PlacementAvailability events = mock(PlacementAvailability.class);
        EncoderEndpoint endpoint = endpoint(events);
        assertTrue(endpoint.trackSelectedRequest("Aa", 10));
        ConcurrentMap<String, Object> requests =
                (ConcurrentMap<String, Object>)
                        ReflectionTestUtils.getField(endpoint, "requestObservations");
        CountDownLatch mapEntryLocked = new CountDownLatch(1);
        CountDownLatch releaseMapEntry = new CountDownLatch(1);
        AtomicReference<Throwable> error = new AtomicReference<>();
        Thread mapWriter =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try {
                                        requests.compute(
                                                "Aa",
                                                (id, current) -> {
                                                    mapEntryLocked.countDown();
                                                    try {
                                                        assertTrue(
                                                                releaseMapEntry.await(
                                                                        5, TimeUnit.SECONDS));
                                                    } catch (InterruptedException interrupted) {
                                                        Thread.currentThread().interrupt();
                                                        throw new AssertionError(interrupted);
                                                    }
                                                    return current;
                                                });
                                    } catch (Throwable failure) {
                                        error.compareAndSet(null, failure);
                                    }
                                });
        Thread registering = null;
        try {
            assertTrue(mapEntryLocked.await(2, TimeUnit.SECONDS));
            // These request IDs share a map bin, allowing registration to pause after admission.
            registering =
                    Thread.ofPlatform()
                            .start(
                                    () -> {
                                        try {
                                            assertTrue(endpoint.trackSelectedRequest("BB", 20));
                                        } catch (Throwable failure) {
                                            error.compareAndSet(null, failure);
                                        }
                                    });
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while (registering.getState() != Thread.State.BLOCKED && System.nanoTime() < deadline) {
                Thread.yield();
            }
            assertEquals(Thread.State.BLOCKED, registering.getState());

            endpoint.close();

            assertFalse(endpoint.trackSelectedRequest("late-request", 30));
            verify((RequestRepository) ReflectionTestUtils.getField(endpoint, "requests"), never())
                    .findActive(org.mockito.ArgumentMatchers.anyString(), eq(RequestPhase.ENCODER));
        } finally {
            releaseMapEntry.countDown();
            mapWriter.join(2_000);
            if (registering != null) {
                registering.join(2_000);
            }
        }
        endpoint.awaitRetirement();
        assertEquals(null, error.get());
        assertEquals(0, endpoint.pendingEncoderRequestCount());
        var repository = (RequestRepository) ReflectionTestUtils.getField(endpoint, "requests");
        verify(repository).findActive("Aa", RequestPhase.ENCODER);
        verify(repository).findActive("BB", RequestPhase.ENCODER);
        verify(events).changed(eq(endpoint.getStatus()), eq(endpoint.ipPort()));
    }

    @Test
    void closingSeveralPendingRequestsNotifiesCapacityOnce() {
        PlacementAvailability events = mock(PlacementAvailability.class);
        EncoderEndpoint endpoint = endpoint(events);
        assertTrue(endpoint.trackSelectedRequest("request-1", 10));
        assertTrue(endpoint.trackSelectedRequest("request-2", 20));
        assertFalse(endpoint.trackSelectedRequest("request-1", 10));

        endpoint.close();
        endpoint.awaitRetirement();

        assertEquals(0, endpoint.pendingEncoderRequestCount());
        assertFalse(endpoint.trackSelectedRequest("request-3", 30));
        verify(events).changed(eq(endpoint.getStatus()), eq(endpoint.ipPort()));
    }

    private static EncoderEndpoint endpoint(PlacementAvailability events) {
        return new EncoderEndpoint(
                WorkerStatus.createDiscovered(
                        RoleType.ENCODER, "default", "127.0.0.1", 8001, 8002, ""),
                mock(RequestRepository.class),
                events);
    }
}
