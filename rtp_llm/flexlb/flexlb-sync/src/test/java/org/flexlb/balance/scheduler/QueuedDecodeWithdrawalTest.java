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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class QueuedDecodeWithdrawalTest {
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
        decode = new DecodeEndpoint(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8000, 8001, null), mock(EndpointEventProjector.class));
    }

    @AfterEach
    void close() {
        if (registry.closeAdmissionAndAwaitMutations()) { registry.closeOutstandingAndTerminalize(); }
        registry.closeExpiration();
        registry.closePublisher();
    }

    private ScheduledRequest queued(long id) {
        var context = RequestLifecycleTestSupport.context(config, id);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(30, System.currentTimeMillis() + 60_000L));
        var future = registry.register(context);
        context.setFuture(future);
        DecodeEndpoint.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.reserve(pin, id, 16, 16, 30, capacity);
        }
        assertNotNull(reservation);
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.removeQueued(any(), anyString())).thenReturn(true);
        var item = new ScheduledRequest(context, future, new Response(), new ServerStatus(), new ServerStatus(),
                prefill, decode, reservation, System.currentTimeMillis());
        try (var admission = registry.claimAdmissionHandle(id, future)) {
            assertNotNull(admission);
            assertTrue(registry.commitItemForPublication(item, () -> true));
        }
        return item;
    }

    private boolean replace(ScheduledRequest item) {
        return registry.replaceQueuedDecodeReservations(decode, List.of(item.decodeReservation()),
                100, 16, 16, 80, capacity);
    }

    @Test
    void replacementKeepsRequestAliveAndTransfersCapacityBeforeRequeue() {
        var item = queued(1);
        doAnswer(call -> {
            assertNull(decode.reservationHandle(1));
            assertNotNull(decode.reservationHandle(100));
            assertNull(registry.requestSlot(1).activeItem());
            assertFalse(item.future().isDone());
            return true;
        }).when(queue).requeue(item);
        assertTrue(replace(item));
        assertEquals(1, decode.routingView().totalLoad());
        assertEquals(RequestState.Phase.QUEUED, registry.getRequestState(1, 0).state());
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
        try (var pin = decode.tryPinGeneration()) {
            assertNull(decode.reserve(pin, 1, 16, 16, 30, capacity),
                    "victim cannot reclaim capacity already assigned to the incoming request");
        }
        decode.release(decode.reservationHandle(100), DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
        DecodeEndpoint.ReservationHandle second;
        try (var pin = decode.tryPinGeneration()) { second = decode.reserve(pin, 1, 16, 16, 30, capacity); }
        var next = new ScheduledRequest(item.ctx(), item.future(), new Response(), item.prefill(), item.decode(),
                item.prefillEp(), decode, second, item.enqueuedAtMs() + 1000);
        assertEquals(item.enqueuedAtMs(), next.enqueuedAtMs());
        assertEquals(item.enqueueSeq(), next.enqueueSeq());
        assertEquals(item.expiresAtMs(), next.expiresAtMs());
        try (var admission = registry.claimAdmissionHandle(1, item.future())) {
            assertNotNull(admission);
            assertTrue(registry.commitItemForPublication(next, () -> true));
        }
        registry.processDecodeStatus(decode, DecodeEndpoint.WorkerStatusFact.terminal(item.decodeReservation(), 0));
        assertSame(next, registry.requestSlot(1).activeItem());
        assertFalse(item.future().isDone(), "old reservation evidence must not terminate the new route");
    }

    @Test
    void preparedPermitPreventsWithdrawalAndLeavesOriginalRouteUsable() {
        var item = queued(2);
        var permit = decode.acquireDispatchPermit(item.decodeReservation(), capacity).permit();
        assertNotNull(permit);
        assertFalse(replace(item));
        assertSame(item, registry.requestSlot(2).activeItem());
        assertFalse(item.future().isDone());
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, item));
        verify(queue, never()).requeue(any());
        assertTrue(permit.release());
    }

    @Test
    void equalPriorityCannotBeWithdrawn() {
        var item = queued(3);
        assertFalse(registry.replaceQueuedDecodeReservations(decode, List.of(item.decodeReservation()),
                100, 16, 16, 30, capacity));
        assertSame(item, registry.requestSlot(3).activeItem());
        assertNotNull(decode.reservationHandle(3));
        verify(queue, never()).requeue(any());
    }

    @ParameterizedTest
    @EnumSource(value = CancelReason.class, names = {"CLIENT_CANCELLED", "DEADLINE_EXCEEDED"})
    void cancellationDuringWithdrawalSettlesOriginalFutureWithoutRequeue(CancelReason reason) throws Exception {
        var item = queued(4);
        doAnswer(call -> {
            assertFalse(RequestLifecycleTestSupport.prepareMember(registry, item),
                    "withdrawal must fence batch preparation before releasing the old route");
            registry.cancelRequest(4, 0, reason);
            return true;
        }).when(item.prefillEp()).removeQueued(eq(item), anyString());
        assertTrue(replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(decode.reservationHandle(4));
        assertNotNull(decode.reservationHandle(100));
        verify(queue, never()).requeue(item);
    }

    @Test
    void failedReplacementDoesNotRemoveOrRequeueVictim() {
        var item = queued(5);
        var stale = new DecodeEndpoint.ReservationHandle(item.decodeReservation().endpointGenerationId(),
                5, item.decodeReservation().reservationToken() + 1);
        assertFalse(registry.replaceQueuedDecodeReservations(decode, List.of(stale), 100, 16, 16, 80, capacity));
        assertSame(item, registry.requestSlot(5).activeItem());
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, item));
        verify(item.prefillEp(), never()).removeQueued(any(), anyString());
    }

    @Test
    void queueShutdownDuringWithdrawalTerminatesVictimInsteadOfLeaking() throws Exception {
        var item = queued(6);
        when(queue.requeue(item)).thenReturn(false);
        assertTrue(replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, registry.liveRequestCount());
        assertNull(decode.reservationHandle(6));
    }
    @Test
    void laterVictimConflictReleasesEarlierWithdrawalClaim() {
        var item = queued(7);
        var missing = new DecodeEndpoint.ReservationHandle(
                item.decodeReservation().endpointGenerationId(), 999, 1);
        assertFalse(registry.replaceQueuedDecodeReservations(decode,
                List.of(item.decodeReservation(), missing), 100, 16, 16, 80, capacity));
        assertSame(item, registry.requestSlot(7).activeItem());
        assertNotNull(decode.reservationHandle(7));
        assertNull(decode.reservationHandle(100));
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, item),
                "an aborted multi-victim plan must not leave earlier victims fenced");
        verify(queue, never()).requeue(any());
    }

    @Test
    void requeueFailureTerminatesVictimAndReleasesIncomingReservation() throws Exception {
        var item = queued(8);
        when(queue.requeue(item)).thenThrow(new IllegalStateException("injected requeue failure"));
        assertThrows(IllegalStateException.class, () -> replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, registry.liveRequestCount());
        assertEquals(0, decode.routingView().totalLoad());
    }

    @Test
    void withdrawalBlocksConcurrentDispatchAndSettlesConcurrentCancellation() throws Exception {
        var item = queued(9);
        var removing = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        doAnswer(call -> {
            removing.countDown();
            assertTrue(resume.await(5, TimeUnit.SECONDS));
            return true;
        }).when(item.prefillEp()).removeQueued(eq(item), anyString());
        try (var executor = Executors.newSingleThreadExecutor()) {
            var replacement = executor.submit(() -> replace(item));
            try {
                assertTrue(removing.await(5, TimeUnit.SECONDS));
                assertFalse(RequestLifecycleTestSupport.prepareMember(registry, item));
                registry.cancelRequest(9, 0, CancelReason.CLIENT_CANCELLED);
                assertFalse(item.future().isDone(), "cancellation waits for withdrawal ownership to close");
            } finally {
                resume.countDown();
            }
            assertTrue(replacement.get(5, TimeUnit.SECONDS));
        }
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).requeue(item);
        assertNull(decode.reservationHandle(9));
        assertNotNull(decode.reservationHandle(100));
    }

}
