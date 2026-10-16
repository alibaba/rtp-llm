package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.EndpointEventProjector;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

class EncoderEndpointTest {

    @Test
    @SuppressWarnings("unchecked")
    void retirementWaitsForRegistrationAlreadyInsideTheGenerationGate() throws Exception {
        EndpointEventProjector events = mock(EndpointEventProjector.class);
        EncoderEndpoint endpoint = endpoint(events);
        assertTrue(endpoint.trackSelectedRequest("Aa", 10));
        ConcurrentMap<String, Object> requests = (ConcurrentMap<String, Object>)
                ReflectionTestUtils.getField(endpoint, "requestObservations");
        CountDownLatch mapEntryLocked = new CountDownLatch(1);
        CountDownLatch releaseMapEntry = new CountDownLatch(1);
        AtomicReference<Throwable> error = new AtomicReference<>();
        Thread mapWriter = Thread.ofPlatform().start(() -> {
            try {
                requests.compute("Aa", (id, current) -> {
                    mapEntryLocked.countDown();
                    try {
                        assertTrue(releaseMapEntry.await(5, TimeUnit.SECONDS));
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
            registering = Thread.ofPlatform().start(() -> {
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
            verify(events, never()).onEncoderGenerationRetired(eq(endpoint),
                    org.mockito.ArgumentMatchers.anyList());
        } finally {
            releaseMapEntry.countDown();
            mapWriter.join(2_000);
            if (registering != null) { registering.join(2_000); }
        }
        endpoint.awaitRetirement();
        assertEquals(null, error.get());
        assertEquals(0, endpoint.pendingEncoderRequestCount());
        verify(events).onEncoderGenerationRetired(eq(endpoint),
                argThat(ids -> ids.size() == 2 && ids.containsAll(List.of("Aa", "BB"))));
        verify(events).onEncoderCapacityChanged();
    }

    @Test
    void closingSeveralPendingRequestsNotifiesCapacityOnce() {
        EndpointEventProjector events = mock(EndpointEventProjector.class);
        EncoderEndpoint endpoint = endpoint(events);
        assertTrue(endpoint.trackSelectedRequest("request-1", 10));
        assertTrue(endpoint.trackSelectedRequest("request-2", 20));
        assertFalse(endpoint.trackSelectedRequest("request-1", 10));

        endpoint.close();
        endpoint.awaitRetirement();

        assertEquals(0, endpoint.pendingEncoderRequestCount());
        assertFalse(endpoint.trackSelectedRequest("request-3", 30));
        verify(events).onEncoderCapacityChanged();
    }

    private static EncoderEndpoint endpoint(EndpointEventProjector events) {
        return new EncoderEndpoint(WorkerStatus.createDiscovered(
                RoleType.ENCODER, "default", "127.0.0.1", 8001, 8002, ""), events);
    }
}
