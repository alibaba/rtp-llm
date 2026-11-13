package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Queued-route withdrawal protocol for hotspot rescue migrations. */
class HotspotRescueMigrationTest {
    private static final String SOURCE_ADDRESS = "10.0.0.1:8080";

    private FlexlbConfig config;
    private RequestRegistry registry;
    private GlobalQueueCoordinator queue;
    private DecodeEndpoint decode;
    private final DecodeEndpoint.AdmissionCapacity capacity = new DecodeEndpoint.AdmissionCapacity(1, 95);

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class), mock(RequestSchedulerReporter.class));
        queue = mock(GlobalQueueCoordinator.class);
        when(queue.requeue(any())).thenReturn(true);
        registry.attachGlobalQueue(queue);
        decode = new DecodeEndpoint(WorkerStatusTestFixtures.readyForScheduling(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8000, 8001, null)), mock(EndpointEventProjector.class));
    }

    @AfterEach
    void close() {
        if (queue == null) { return; }
        if (registry.closeAdmissionAndAwaitMutations()) { registry.closeOutstandingAndTerminalize(); }
        registry.closeExpiration();
        registry.closePublisher();
    }

    private ScheduledRequest queued(long id, long deadline) {
        return queued(id, deadline, source(true));
    }

    private ScheduledRequest queued(long id, long deadline, PrefillEndpoint prefill) {
        var context = RequestLifecycleTestSupport.context(config, id);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(30, deadline));
        var future = registry.register(context);
        context.setFuture(future);
        DecodeEndpoint.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.reserve(pin, id, 16, 16, 30, capacity);
        }
        assertNotNull(reservation);
        var item = new ScheduledRequest(context, future, new Response(), new ServerStatus(), new ServerStatus(),
                prefill, decode, reservation, System.currentTimeMillis());
        try (var admission = registry.claimAdmissionHandle(id, future)) {
            assertNotNull(admission);
            assertTrue(registry.commitItemForPublication(item, () -> true));
        }
        return item;
    }

    private PrefillEndpoint source(boolean removable) {
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.removeQueued(any(), anyString())).thenReturn(removable);
        when(prefill.ipPort()).thenReturn(SOURCE_ADDRESS);
        return prefill;
    }

    @Test
    void migrationWithdrawsRequeuesAndKeepsTheRequestAlive() {
        var item = queued(1, System.currentTimeMillis() + 60_000L);
        doAnswer(call -> {
            assertNull(decode.reservationHandle(1), "migration must release the queued Decode reservation");
            assertNull(registry.requestSlot(1).activeItem(), "the withdrawn route must detach from the slot");
            assertEquals(SOURCE_ADDRESS, item.ctx().getPrefillMigrationSource());
            return true;
        }).when(queue).requeue(item);
        assertTrue(registry.migrateQueuedPrefillRoute(item.prefillEp(), item));
        verify(item.prefillEp()).removeQueued(item, "HOTSPOT_RESCUE_MIGRATION");
        verify(queue).requeue(item);
        // A rescue re-queues the request instead of evicting it: the Future stays
        // open and the queue identity (sequence, enqueue time, deadline) survives.
        assertFalse(item.future().isDone());
        assertEquals(RequestState.Phase.QUEUED, registry.getRequestState(1, 0).state());
        var next = new ScheduledRequest(item.ctx(), item.future(), new Response(), item.prefill(), item.decode(),
                item.prefillEp(), null, null, item.enqueuedAtMs() + 1000);
        assertEquals(item.enqueuedAtMs(), next.enqueuedAtMs());
        assertEquals(item.enqueueSeq(), next.enqueueSeq());
        assertEquals(item.expiresAtMs(), next.expiresAtMs());
        assertEquals(SOURCE_ADDRESS, item.ctx().getPrefillMigrationSource());
    }

    @Test
    void aRequestMigratesAtMostOnce() {
        var item = queued(2, System.currentTimeMillis() + 60_000L);
        assertTrue(registry.migrateQueuedPrefillRoute(item.prefillEp(), item));
        assertFalse(registry.migrateQueuedPrefillRoute(item.prefillEp(), item));
        verify(item.prefillEp(), times(1)).removeQueued(any(), anyString());
        verify(queue, times(1)).requeue(any());
    }

    @Test
    void failedWithdrawalKeepsTheOriginalRouteDeliverable() {
        var item = queued(3, System.currentTimeMillis() + 60_000L, source(false));
        // The delivery race wins: the route is no longer queued on the source,
        // so the withdrawal is refused and the original route stays deliverable.
        assertFalse(registry.migrateQueuedPrefillRoute(item.prefillEp(), item));
        verify(queue, never()).requeue(any());
        assertFalse(item.future().isDone());
        assertSame(item.execution(), registry.requestSlot(3).activeItem());
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, item));
    }

    @Test
    @Timeout(10)
    void refusedMigrationReleasesTheAdmissionGate() {
        // The queued route belongs to a different worker: the withdrawal is
        // refused before any claim exists, and the admission gate must not leak
        // or scheduler shutdown would hang forever.
        var item = queued(4, System.currentTimeMillis() + 60_000L);
        PrefillEndpoint unrelated = source(true);
        when(unrelated.ipPort()).thenReturn("10.0.0.9:8080");
        assertFalse(registry.migrateQueuedPrefillRoute(unrelated, item));
        verify(item.prefillEp(), never()).removeQueued(any(), anyString());
        verify(unrelated, never()).removeQueued(any(), anyString());
        verify(queue, never()).requeue(any());
        assertTrue(registry.closeAdmissionAndAwaitMutations());
        registry.closeOutstandingAndTerminalize();
        registry.closeExpiration();
        registry.closePublisher();
        queue = null;
    }
}
