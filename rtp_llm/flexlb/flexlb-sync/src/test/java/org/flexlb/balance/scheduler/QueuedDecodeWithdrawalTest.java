package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

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
        return queued(id, item -> { });
    }

    private ScheduledRequest queued(long id, java.util.function.Consumer<ScheduledRequest> duringPublication) {
        var context = RequestLifecycleTestSupport.context(config, id);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(30, System.currentTimeMillis() + 60_000L));
        var future = registry.register(context);
        context.setFuture(future);
        DecodeEndpoint.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.reserve(pin, Long.toString(id), 16, 16, 30, capacity);
        }
        assertNotNull(reservation);
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.removeQueued(any(), anyString())).thenReturn(true);
        var item = new ScheduledRequest(context, future, new Response(), new ServerStatus(), new ServerStatus(),
                prefill, decode, reservation, System.currentTimeMillis());
        try (var admission = registry.claimAdmissionHandle(Long.toString(id), future)) {
            assertNotNull(admission);
            assertTrue(registry.commitItemForPublication(item, () -> true));
            duringPublication.accept(item);
        }
        return item;
    }

    @Test
    void lateDecodeFinishedAfterEmptyWorkerReportDrainsAcknowledgedRequest() {
        decode.close();
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8000, 8001, null);
        decode = new DecodeEndpoint(status, new EndpointEventProjector(registry));
        applyStatus(Map.of(), Map.of());
        var item = queued(42);
        var slot = registry.requestSlot("42");
        var delivery = slot.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 7L, () -> true);
        assertNotNull(delivery);
        delivery.complete(DeliveryResult.delivered());
        assertEquals(RequestState.Phase.ACKNOWLEDGED, slot.snapshot().state());
        TaskInfo task = new TaskInfo();
        task.setRequestId("42");
        task.setInputLength(16L);
        task.setPhase(TaskPhase.RUNNING);
        applyStatus(Map.of("42", task), Map.of());
        applyStatus(Map.of(), Map.of());
        applyStatus(Map.of(), Map.of());
        assertEquals(RequestState.Phase.ACKNOWLEDGED, slot.snapshot().state());
        assertEquals(1, registry.snapshotActiveRequests().size());
        assertEquals(0, decode.routingView().engineCapacityUsed());
        applyStatus(Map.of(), Map.of("42", task));
        assertEquals(RequestState.Phase.COMPLETED, slot.snapshot().state());
        assertTrue(registry.snapshotActiveRequests().isEmpty());
        assertFalse(decode.isAcceptedByEngine(item.decodeReservation()));
        applyStatus(Map.of(), Map.of("42", task));
        assertTrue(registry.snapshotActiveRequests().isEmpty());
        decode.close();
    }

    private void applyStatus(Map<String, TaskInfo> running, Map<String, TaskInfo> finished) {
        WorkerStatus status = decode.getStatus();
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.DECODE);
        response.setAlive(true);
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(10_000L);
        response.setRunningTaskInfo(running);
        response.setFinishedTaskInfo(finished);
        response.setStatusVersion(Math.max(1, status.appliedStatusCursor().statusVersion() + 1));
        response.setLatestFinishedVersion(Math.max(0, status.appliedStatusCursor().latestFinishedTaskVersion()) + finished.size());
        Runnable projection;
        status.lock.lock();
        try {
            projection = decode.applyPreparedStatus(status, status.prepareNewStatus(status.freezeStatusResponse(response)));
        } finally {
            status.lock.unlock();
        }
        projection.run();
    }

    @Test
    void stoppedWorkerReleasesOldRouteBeforeRequeueAndIgnoresRepeatedNotification() {
        var item = queued(20);
        doAnswer(call -> {
            assertNull(decode.reservationHandle("20"));
            assertNull(registry.requestSlot("20").activeItem());
            assertFalse(item.future().isDone());
            return true;
        }).when(queue).requeue(item);
        registry.onQueuedWorkerStopped(item);
        registry.onQueuedWorkerStopped(item);
        verify(queue, org.mockito.Mockito.times(1)).requeue(item);
        assertEquals(RequestState.Phase.QUEUED, registry.getRequestState("20", 0).state());
        assertEquals(0, decode.routingView().totalLoad());
    }

    @Test
    void stoppedWorkerDuringPublicationWaitsForAdmissionToClose() {
        var item = queued(21, publishing -> {
            registry.onQueuedWorkerStopped(publishing);
            verify(queue, never()).requeue(any());
            assertNotNull(decode.reservationHandle("21"));
        });
        verify(queue).requeue(item);
        assertNull(registry.requestSlot("21").activeItem());
        assertNull(decode.reservationHandle("21"));
        assertFalse(item.future().isDone());
    }

    @ParameterizedTest
    @EnumSource(value = CancelReason.class, names = {"CLIENT_CANCELLED", "DEADLINE_EXCEEDED"})
    void cancellationWinsOverDeferredWorkerRequeue(CancelReason reason) throws Exception {
        var item = queued(22, publishing -> {
            registry.onQueuedWorkerStopped(publishing);
            registry.cancelRequest("22", 0, reason);
        });
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).requeue(any());
        assertNull(decode.reservationHandle("22"));
    }

    @Test
    void stoppedWorkerCannotRequeueAfterDispatchHandoff() {
        var item = queued(23);
        assertNotNull(registry.requestSlot("23").claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 1L, () -> true));
        registry.onQueuedWorkerStopped(item);
        verify(queue, never()).requeue(any());
        assertSame(item, registry.requestSlot("23").activeItem());
        assertNotNull(decode.reservationHandle("23"));
    }

    @Test
    void stoppedWorkerRequeueIntoClosedGlobalQueueTerminatesRequest() throws Exception {
        var item = queued(24);
        when(queue.requeue(item)).thenReturn(false);
        registry.onQueuedWorkerStopped(item);
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(decode.reservationHandle("24"));
    }

    @Test
    void stoppedWorkerAfterDispatchPreparedButBeforeClaimRequeuesAndRejectsLateClaim() {
        var item = queued(25);
        var slot = registry.requestSlot("25");
        var dispatchPrepared = new AtomicBoolean();
        var prepared = slot.prepareDispatch(item, () -> {
            dispatchPrepared.set(true);
            return CapacityBoundary.Attempt.accepted(dispatchPrepared);
        });
        assertTrue(prepared.accepted());

        registry.onQueuedWorkerStopped(item);

        verify(queue, org.mockito.Mockito.times(1)).requeue(item);
        assertNull(slot.activeItem());
        assertNull(decode.reservationHandle("25"));
        var handedToEndpoint = new AtomicBoolean();
        assertNull(RequestLifecycleTestSupport.claimBatchWithoutPrediction(registry, item, 1L, () -> {
            handedToEndpoint.set(true);
            return true;
        }));
        assertNull(slot.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 1L, () -> {
            handedToEndpoint.set(true);
            return true;
        }));
        assertFalse(handedToEndpoint.get());
        assertFalse(item.future().isDone());
        assertEquals(RequestState.Phase.QUEUED, registry.getRequestState("25", 0).state());
    }

    private boolean replace(ScheduledRequest item) {
        return registry.replaceQueuedDecodeReservations(decode, List.of(item.decodeReservation()),
                "100", 16, 16, 80, capacity);
    }

    @Test
    void decodeRetirementReleasesBothEndpointsAndRequeuesOnlyOnce() {
        var item = queued(26);
        doAnswer(call -> {
            verify(item.prefillEp()).settleFailedRequest(item);
            assertNull(decode.reservationHandle("26"));
            assertNull(registry.requestSlot("26").activeItem());
            assertFalse(item.future().isDone());
            return true;
        }).when(queue).requeue(item);

        registry.projectDecodeRetirementReservation(decode, item.decodeReservation());
        registry.projectDecodeRetirementReservation(decode, item.decodeReservation());

        verify(queue, org.mockito.Mockito.times(1)).requeue(item);
        assertEquals(RequestState.Phase.QUEUED, registry.getRequestState("26", 0).state());
    }

    @Test
    void decodeRetirementBetweenRouteClaimAndPublishStillPublishesRoute() throws Exception {
        var item = queued(31);
        var slot = registry.requestSlot("31");
        var claim = slot.claimDelivery(item, DeliveryClaimKind.ROUTE_DECISION, 0L, () -> true);
        assertNotNull(claim);

        registry.projectDecodeRetirementReservation(decode, item.decodeReservation());

        assertFalse(item.future().isDone());
        verify(queue, never()).requeue(any());

        registry.publishRoute(claim, new WorkSnapshot(System.currentTimeMillis(),
                List.of(), List.of(), 0L), 30_000L);

        assertTrue(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).requeue(any());
    }

    @Test
    void decodeRetirementDuringPublicationWaitsForAdmissionToClose() {
        var item = queued(27, publishing -> {
            registry.projectDecodeRetirementReservation(decode, publishing.decodeReservation());
            registry.projectDecodeRetirementReservation(decode, publishing.decodeReservation());
            assertNull(registry.requestSlot("27").claimDelivery(publishing,
                    DeliveryClaimKind.BATCH_ENQUEUE, 1L, () -> {
                        throw new AssertionError("retired Decode must not receive the route");
                    }));
            verify(queue, never()).requeue(any());
            assertNotNull(registry.requestSlot("27").activeItem());
        });
        verify(queue).requeue(item);
        assertNull(registry.requestSlot("27").activeItem());
        assertFalse(item.future().isDone());
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {true, false})
    void bothEndpointRetirementsDuringPublicationUseOnePendingRequeue(boolean decodeFirst) {
        var item = queued(33, publishing -> {
            if (decodeFirst) {
                registry.projectDecodeRetirementReservation(decode, publishing.decodeReservation());
                registry.onQueuedWorkerStopped(publishing);
            } else {
                registry.onQueuedWorkerStopped(publishing);
                registry.projectDecodeRetirementReservation(decode, publishing.decodeReservation());
            }
            verify(queue, never()).requeue(any());
        });
        verify(queue, org.mockito.Mockito.times(1)).requeue(item);
        verify(item.prefillEp()).settleFailedRequest(item);
        assertNull(decode.reservationHandle("33"));
        assertNull(registry.requestSlot("33").activeItem());
        assertFalse(item.future().isDone());
    }

    @ParameterizedTest
    @EnumSource(value = CancelReason.class, names = {"CLIENT_CANCELLED", "DEADLINE_EXCEEDED"})
    void cancellationWinsOverDeferredDecodeRequeue(CancelReason reason) throws Exception {
        var item = queued(28, publishing -> {
            registry.projectDecodeRetirementReservation(decode, publishing.decodeReservation());
            registry.cancelRequest("28", 0, reason);
        });
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).requeue(any());
        assertNull(decode.reservationHandle("28"));
    }

    @Test
    void decodePreparationRetirementRequeuesInsteadOfPublishingFailure() {
        var item = queued(29);
        decode.close();
        var prepared = PrefillAdmissionResources.prepareMember(item);
        assertFalse(prepared.accepted());
        registry.failDeliveryPreparation(item, prepared.boundary().cause());
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
        assertNull(registry.requestSlot("29").activeItem());
    }

    @Test
    void decodePermitRetirementBeforeDeliveryClaimRequeuesInsteadOfPublishingFailure() {
        var item = queued(30);
        var member = PrefillAdmissionResources.prepareMember(item).value();
        var admission = PrefillAdmissionResources.createCommittedOwner(List.of(member));
        admission.bindPrefillHandoff(null);
        decode.close();
        var failure = assertThrows(EndpointRetiredException.class,
                () -> registry.claimRouteDelivery(item, admission));
        registry.failDeliveryPreparation(item, failure);
        admission.close();
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
    }

    @Test
    void prefillBatchRetirementRequeuesInsteadOfPublishingFailure() {
        var item = queued(34);
        var failure = PrefillAdmissionResources.rejectedPrefill(
                item, PrefillState.CapacityStatus.ENDPOINT_RETIRED, null);
        assertFalse(failure.accepted());
        registry.failDeliveryPreparation(item, failure.boundary().cause());
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
        assertNull(registry.requestSlot("34").activeItem());
    }

    @Test
    void decodeRetirementAfterDeliveryHandoffDoesNotRequeue() throws Exception {
        var item = queued(31);
        assertNotNull(registry.requestSlot("31").claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 1L, () -> true));
        registry.projectDecodeRetirementReservation(decode, item.decodeReservation());
        verify(queue, never()).requeue(any());
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void closedQueueEndsDecodeRetirementRequeue() throws Exception {
        var item = queued(32);
        when(queue.requeue(item)).thenReturn(false);
        registry.projectDecodeRetirementReservation(decode, item.decodeReservation());
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(decode.reservationHandle("32"));
        assertNull(registry.requestSlot("32").activeItem());
    }

    @Test
    void replacementKeepsRequestAliveAndTransfersCapacityBeforeRequeue() {
        var item = queued(1);
        doAnswer(call -> {
            assertNull(decode.reservationHandle("1"));
            assertNotNull(decode.reservationHandle("100"));
            assertNull(registry.requestSlot("1").activeItem());
            assertFalse(item.future().isDone());
            return true;
        }).when(queue).requeue(item);
        assertTrue(replace(item));
        assertEquals(1, decode.routingView().totalLoad());
        assertEquals(RequestState.Phase.QUEUED, registry.getRequestState("1", 0).state());
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
        try (var pin = decode.tryPinGeneration()) {
            assertNull(decode.reserve(pin, "1", 16, 16, 30, capacity),
                    "victim cannot reclaim capacity already assigned to the incoming request");
        }
        decode.release(decode.reservationHandle("100"), DecodeEndpoint.ReleaseReason.LOCAL_ROLLBACK);
        DecodeEndpoint.ReservationHandle second;
        try (var pin = decode.tryPinGeneration()) { second = decode.reserve(pin, "1", 16, 16, 30, capacity); }
        var next = new ScheduledRequest(item.ctx(), item.future(), new Response(), item.prefill(), item.decode(),
                item.prefillEp(), decode, second, item.enqueuedAtMs() + 1000);
        assertEquals(item.enqueuedAtMs(), next.enqueuedAtMs());
        assertEquals(item.enqueueSeq(), next.enqueueSeq());
        assertEquals(item.expiresAtMs(), next.expiresAtMs());
        try (var admission = registry.claimAdmissionHandle("1", item.future())) {
            assertNotNull(admission);
            assertTrue(registry.commitItemForPublication(next, () -> true));
        }
        registry.processDecodeStatus(decode, DecodeEndpoint.WorkerStatusFact.terminal(item.decodeReservation(), 0));
        assertSame(next, registry.requestSlot("1").activeItem());
        assertFalse(item.future().isDone(), "old reservation evidence must not terminate the new route");
    }

    @Test
    void preparedPermitPreventsWithdrawalAndLeavesOriginalRouteUsable() {
        var item = queued(2);
        var permit = decode.acquireDispatchPermit(item.decodeReservation(), capacity).permit();
        assertNotNull(permit);
        assertFalse(replace(item));
        assertSame(item, registry.requestSlot("2").activeItem());
        assertFalse(item.future().isDone());
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, item));
        verify(queue, never()).requeue(any());
        assertTrue(permit.release());
    }

    @Test
    void equalPriorityCannotBeWithdrawn() {
        var item = queued(3);
        assertFalse(registry.replaceQueuedDecodeReservations(decode, List.of(item.decodeReservation()),
                "100", 16, 16, 30, capacity));
        assertSame(item, registry.requestSlot("3").activeItem());
        assertNotNull(decode.reservationHandle("3"));
        verify(queue, never()).requeue(any());
    }

    @ParameterizedTest
    @EnumSource(value = CancelReason.class, names = {"CLIENT_CANCELLED", "DEADLINE_EXCEEDED"})
    void cancellationDuringWithdrawalSettlesOriginalFutureWithoutRequeue(CancelReason reason) throws Exception {
        var item = queued(4);
        doAnswer(call -> {
            assertFalse(RequestLifecycleTestSupport.prepareMember(registry, item),
                    "withdrawal must fence batch preparation before releasing the old route");
            registry.cancelRequest("4", 0, reason);
            return true;
        }).when(item.prefillEp()).removeQueued(eq(item), anyString());
        assertTrue(replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(decode.reservationHandle("4"));
        assertNotNull(decode.reservationHandle("100"));
        verify(queue, never()).requeue(item);
    }

    @Test
    void failedReplacementDoesNotRemoveOrRequeueVictim() {
        var item = queued(5);
        var stale = new DecodeEndpoint.ReservationHandle(item.decodeReservation().endpointGenerationId(),
                "5", item.decodeReservation().reservationToken() + 1);
        assertFalse(registry.replaceQueuedDecodeReservations(decode, List.of(stale), "100", 16, 16, 80, capacity));
        assertSame(item, registry.requestSlot("5").activeItem());
        assertTrue(RequestLifecycleTestSupport.prepareMember(registry, item));
        verify(item.prefillEp(), never()).removeQueued(any(), anyString());
    }

    @Test
    void queueShutdownDuringWithdrawalTerminatesVictimInsteadOfLeaking() throws Exception {
        var item = queued(6);
        when(queue.requeue(item)).thenReturn(false);
        assertTrue(replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, registry.trackedRequestCount());
        assertNull(decode.reservationHandle("6"));
    }
    @Test
    void laterVictimConflictReleasesEarlierWithdrawalClaim() {
        var item = queued(7);
        var missing = new DecodeEndpoint.ReservationHandle(
                item.decodeReservation().endpointGenerationId(), "999", 1);
        assertFalse(registry.replaceQueuedDecodeReservations(decode,
                List.of(item.decodeReservation(), missing), "100", 16, 16, 80, capacity));
        assertSame(item, registry.requestSlot("7").activeItem());
        assertNotNull(decode.reservationHandle("7"));
        assertNull(decode.reservationHandle("100"));
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
        assertEquals(0, registry.trackedRequestCount());
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
                registry.cancelRequest("9", 0, CancelReason.CLIENT_CANCELLED);
                assertFalse(item.future().isDone(), "cancellation waits for withdrawal ownership to close");
            } finally {
                resume.countDown();
            }
            assertTrue(replacement.get(5, TimeUnit.SECONDS));
        }
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).requeue(item);
        assertNull(decode.reservationHandle("9"));
        assertNotNull(decode.reservationHandle("100"));
    }

}
