package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Uses the real Decode ledger to verify priority competition and stale-wait cleanup.
 */
class ResourceWaitTest {
    @Test
    void lowerPriorityCannotTakeCapacityButHigherPriorityCan() {
        try (var fixture = new Fixture()) {
            var a = fixture.waiter(60, Long.MAX_VALUE);
            assertTrue(fixture.endpoint.registerResourceWait(a.resourceWait));
            assertNull(fixture.register("b", 20));
            var c = fixture.register("c", 90);
            assertNotNull(c);
            assertFalse(a.resourceWait.isDone());
            fixture.endpoint.rollbackSelection(c);
            fixture.endpoint.tryAcquire(a.resourceWait);
            assertNotNull(a.resourceWait.result().toCompletableFuture().join());
        }
    }

    @Test
    void revocationImmediatelyRemovesRestrictionEvenIfIndexCleanupWasMissed() {
        try (var fixture = new Fixture()) {
            var a = fixture.waiter(60, Long.MAX_VALUE);
            fixture.endpoint.registerResourceWait(a.resourceWait);
            a.resourceWait.revoke();
            assertNotNull(fixture.register("b", 20));
            fixture.endpoint.tryAcquire(a.resourceWait);
            assertNull(a.resourceWait.result().toCompletableFuture().join());
        }
    }

    @Test
    void multipleWaitersKeepTheirOwnPriorityUntilEachRequestStopsWaiting() {
        try (var fixture = new Fixture()) {
            var a = fixture.waiter(60, Long.MAX_VALUE);
            var c = fixture.waiter(90, Long.MAX_VALUE);
            fixture.endpoint.registerResourceWait(a.resourceWait);
            fixture.endpoint.registerResourceWait(c.resourceWait);
            assertNull(fixture.register("middle", 70));
            c.resourceWait.revoke();
            var middle = fixture.register("middle", 70);
            assertNotNull(middle);
            fixture.endpoint.rollbackSelection(middle);
            assertNull(fixture.register("low", 20));
            a.resourceWait.revoke();
            assertNotNull(fixture.register("low", 20));
        }
    }

    @Test
    void revocationRemovesProtectionWithoutCompletingTheResult() {
        try (var fixture = new Fixture()) {
            var a = fixture.waiter(60, Long.MAX_VALUE);
            fixture.endpoint.registerResourceWait(a.resourceWait);
            a.resourceWait.revoke();
            assertFalse(a.resourceWait.isDone());
            assertNotNull(fixture.register("b", 20));
        }
    }

    @Test
    void expiredWaitCannotBeInstalledAndClosingCompletesInstalledWaitOutsideLock() {
        try (var fixture = new Fixture()) {
            assertFalse(fixture.endpoint.registerResourceWait(fixture.waiter(60, 1).resourceWait));
            var a = fixture.waiter(60, Long.MAX_VALUE);
            fixture.endpoint.registerResourceWait(a.resourceWait);
            var called = new java.util.concurrent.atomic.AtomicBoolean();
            a.resourceWait
                    .result()
                    .thenAccept(
                            result -> {
                                assertFalse(fixture.endpoint.stateLock().isHeldByCurrentThread());
                                assertNull(result);
                                called.set(true);
                            });
            EndpointTestSupport.closeAndWait(fixture.endpoint);
            assertTrue(called.get());
        }
    }

    private record Waiter(WorkerRequest request, DecodeEndpoint.ResourceWait resourceWait) {}

    /**
     * Creates a Decode endpoint with controlled capacity and closes it after resource-wait checks.
     */
    private static final class Fixture implements AutoCloseable {
        private final DecodeResources.AdmissionCapacity capacity =
                new DecodeResources.AdmissionCapacity(1, 100);
        private final DecodeEndpoint endpoint =
                EndpointTestSupport.decode(
                        EndpointTestSupport.workerStatus(RoleType.DECODE, "127.0.0.1", 8000, 8001),
                        mock(org.flexlb.balance.schedule.RequestRepository.class));

        private int sequence;

        private Fixture() {
            var status = new WorkerStatusResponse();
            status.setTotalKvCacheTokens(1000L);
            status.setAvailableKvCacheTokens(1000L);
            EndpointTestSupport.applyStatus(endpoint, status).run();
        }

        @Override
        public void close() {
            EndpointTestSupport.closeAndWait(endpoint);
        }

        private Waiter waiter(int priority, long deadline) {
            var request = mock(WorkerRequest.class);
            when(request.getRequestId()).thenReturn("a-" + ++sequence);
            var wait =
                    endpoint.createResourceWait(
                            request, sequence, priority, deadline, 10, 10, capacity);
            return new Waiter(request, wait);
        }

        private DecodeEndpoint.RequestRegistration register(String id, int priority) {
            return endpoint.tryRegisterSelection(
                            id, endpoint.selectionSnapshot(), 10, 10, priority, capacity)
                    .value();
        }
    }
}
