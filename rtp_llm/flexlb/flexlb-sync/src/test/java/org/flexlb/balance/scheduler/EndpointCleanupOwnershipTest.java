package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Cleanup uses current directory membership, including terminal records, rather than live Slot state. */
class EndpointCleanupOwnershipTest {
    private FlexlbConfig config;
    private RequestRegistry registry;

    @BeforeEach void setup() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
    }

    @AfterEach void close() {
        if (registry.closeAdmissionAndAwaitMutations()) {
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @Test void currentDirectoryProtectsReservationButEarlierSnapshotCanEvictIt() {
        Set<Long> beforeRegistration = registry.snapshotSlots().stream()
                .map(RequestSlot::requestId).collect(java.util.stream.Collectors.toSet());
        long id = 7001L;
        registry.register(RequestLifecycleTestSupport.context(config, id));
        var endpoint = new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), mock(EndpointEventProjector.class));
        try (var pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            assertNotNull(endpoint.reservePinned(pin, id, 1L, 1L, 50));
        }
        assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
        assertEquals(1, endpoint.getInflightCount());
        // Models moving the ownership scan outside the endpoint lock without revalidation.
        assertEquals(1, endpoint.evictExpiredRequests(-1L, beforeRegistration::contains));
        assertEquals(0, endpoint.getInflightCount());
        assertTrue(registry.retainForSchedulerCleanup(id), "request is still registered despite premature eviction");
    }

    @Test void terminalRecordMembershipIsConservativeButNotEquivalentToLiveOwnership() throws Exception {
        long id = 7002L;
        var future = registry.register(RequestLifecycleTestSupport.context(config, id));
        RequestSlot original = registry.requestSlot(id);
        registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
        future.get(5, java.util.concurrent.TimeUnit.SECONDS);
        assertEquals(0, registry.liveRequestCount());
        assertTrue(registry.retainForSchedulerCleanup(id));
        assertSame(original, registry.requestSlot(id), "terminal record remains indexed");
        assertTrue(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));
        assertFalse(registry.retainForSchedulerCleanup(id));
        registry.register(RequestLifecycleTestSupport.context(config, id));
        assertNotSame(original, registry.requestSlot(id));
        assertTrue(registry.retainForSchedulerCleanup(id));
        assertFalse(registry.removeExactTerminalRecord(original, Long.MAX_VALUE));
        assertTrue(registry.retainForSchedulerCleanup(id), "old generation cleanup must preserve replacement");
    }

    @Test void registrationAfterAbsentCheckAcquiresFreshReservation() throws Exception {
        long id = 7003L;
        var endpoint = new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), mock(EndpointEventProjector.class));
        DecodeEndpoint.ReservationHandle oldReservation;
        try (var pin = endpoint.tryPinGeneration()) {
            assertNotNull(pin);
            oldReservation = endpoint.reservePinned(pin, id, 1L, 1L, 50);
        }
        assertNotNull(oldReservation);
        CountDownLatch checkedAbsent = new CountDownLatch(1);
        CountDownLatch registered = new CountDownLatch(1);
        var executor = Executors.newFixedThreadPool(2, task -> {
            Thread thread = new Thread(task, "cleanup-registration-race");
            thread.setDaemon(true);
            return thread;
        });
        try {
            var cleanup = executor.submit(() -> endpoint.evictExpiredRequests(-1L, requestId -> {
                boolean retain = registry.retainForSchedulerCleanup(requestId);
                assertFalse(retain);
                checkedAbsent.countDown();
                try {
                    assertTrue(registered.await(5, TimeUnit.SECONDS));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(interrupted);
                }
                return retain;
            }));
            var replacement = executor.submit(() -> {
                assertTrue(checkedAbsent.await(5, TimeUnit.SECONDS));
                registry.register(RequestLifecycleTestSupport.context(config, id));
                registered.countDown();
                try (var pin = endpoint.tryPinGeneration()) {
                    assertNotNull(pin);
                    return endpoint.reservePinned(pin, id, 1L, 1L, 50);
                }
            });
            assertEquals(1, cleanup.get(5, TimeUnit.SECONDS));
            var newReservation = replacement.get(5, TimeUnit.SECONDS);
            assertNotNull(newReservation);
            assertNotEquals(oldReservation.reservationToken(), newReservation.reservationToken());
            assertEquals(1, endpoint.getInflightCount());
            assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
            assertEquals(1, endpoint.getInflightCount());

            RequestSlot slot = registry.requestSlot(id);
            registry.cancelRequest(id, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
            assertTrue(registry.removeExactTerminalRecord(slot, Long.MAX_VALUE));
            assertEquals(1, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
            assertEquals(0, endpoint.evictExpiredRequests(-1L, registry::retainForSchedulerCleanup));
            assertEquals(0, endpoint.getInflightCount());
        } finally {
            executor.shutdownNow();
        }
    }

}
