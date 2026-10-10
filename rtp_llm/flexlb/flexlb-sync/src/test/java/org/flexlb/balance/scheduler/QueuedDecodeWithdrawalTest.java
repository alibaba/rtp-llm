package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class QueuedDecodeWithdrawalTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;
    private QueuedRequestScheduler queue;
    private DecodeEndpoint decode;
    private final DecodeResources.AdmissionCapacity capacity = new DecodeResources.AdmissionCapacity(1, 95);

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        queue = (QueuedRequestScheduler) registry;
        org.mockito.Mockito.doReturn(true).when(queue).requeue(any());
        decode = EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                "127.0.0.1", 8000, 8001, null), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(mock(AbstractRequestScheduler.class)));
    }

    @AfterEach
    void close() {
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) { registry.closeOutstandingAndTerminalize(); }
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
    }

    private RequestRoute queued(long id) {
        var context = RequestProtocolTestSupport.context(config, id);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(30, System.currentTimeMillis() + 60_000L));
        var future = RequestProtocolTestSupport.register(registry, context);
        context.setFuture(future);
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.tryReserveQueuedRequest(pin, id, 16, 16, 30, capacity);
        }
        assertNotNull(reservation);
        var prefill = mock(PrefillEndpoint.class);
        when(prefill.removeQueued(any(), anyString())).thenReturn(true);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), new ServerStatus(), new ServerStatus(),
                prefill, decode, reservation, System.currentTimeMillis());
        try (var admission = registry.claimAdmissionHandle(id, future); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue((registry.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
        return item;
    }

    private boolean replace(RequestRoute item) {
        var incoming = org.flexlb.balance.scheduler.SchedulerTestSupport.eviction(registry).replaceQueuedDecodeReservations(decode, List.of(item.decodeReservation()),
                100, 16, 16, 80, capacity);
        if (incoming != null) {
            assertEquals(EndpointTestSupport.decodeReservation(decode, 100), incoming);
        }
        return incoming != null;
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void evictionUsesSharedRouteCommitAndReleasesFailedPlacement(boolean accepted) throws Exception {
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        var victim = queued(40);
        var context = RequestProtocolTestSupport.context(config, 100);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(80, System.currentTimeMillis() + 60_000L));
        var future = RequestProtocolTestSupport.register(registry, context);
        context.setFuture(future);
        var prefill = mock(PrefillEndpoint.class);
        var selection = mock(WorkerAssignment.class);
        var pin = mock(WorkerEndpoint.GenerationPin.class);
        var prefillStatus = new ServerStatus();
        prefillStatus.setRole(RoleType.PREFILL);
        prefillStatus.setRequestId(100);
        prefillStatus.setSuccess(true);
        when(selection.endpoint()).thenReturn(prefill);
        when(pin.endpoint()).thenReturn(prefill);
        when(prefill.ipPort()).thenReturn("127.0.0.1:9000");
        when(selection.generationPin()).thenReturn(pin);
        when(selection.serverStatus()).thenReturn(prefillStatus);
            when(selection.requestId()).thenReturn(prefillStatus.getRequestId());
            when(selection.role()).thenReturn(prefillStatus.getRole());
            when(selection.group()).thenReturn(prefillStatus.getGroup());
        when(selection.prefillWorkMs()).thenReturn(1L);
        var decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setRequestId(100);
        decodeStatus.setServerIp("127.0.0.1");
        decodeStatus.setHttpPort(8000);
        var replacement = new java.util.concurrent.atomic.AtomicReference<DecodeResources.ReservationHandle>();
        doAnswer(call -> {
            replacement.set(EndpointTestSupport.decodeReservation(decode, 100));
            assertNotNull(replacement.get());
            return true;
        }).when(queue).requeue(victim);
        when(prefill.offerPinned(eq(pin), any(), org.mockito.ArgumentMatchers.any())).thenReturn(accepted);
        var manager = new DecodeCapacityAcquirer(mock(EngineCancelChannel.class), org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry), org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry), mock(RequestSchedulerReporter.class));
        try (var handle = registry.claimAdmissionHandle(100, future); var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(handle);
             var plan = RequestProtocolTestSupport.plan(context, RequestRoute.prepare(context,
                List.of(selection, WorkerAssignment.decode(decode.tryPinGeneration(), decodeStatus, decode.placementVersion()))))) {
            assertNotNull(handle);
            var reservation = manager.tryReclaim(context, context.getRequirements(), decode);
            assertNotNull(reservation);
            var result = reservation.join();
            assertEquals(replacement.get(), result.reservation());
            var scheduler = (QueuedRequestScheduler) context.scheduler();
            assertTrue(RequestProtocolTestSupport.adopt(scheduler, plan, result));
            var placement = scheduler.enqueueRoute(plan);
            if (placement.status() != org.flexlb.balance.PlacementResult.Status.SUCCESS) {
                handle.terminate(Response.buildErrorResponse(StrategyErrorType.RESOURCE_EXHAUSTED,
                        "selected Prefill capacity changed before canonical placement"));
            }
            var offered = org.mockito.ArgumentCaptor.forClass(RequestRoute.class);
            verify(prefill).offerPinned(eq(pin), offered.capture(), org.mockito.ArgumentMatchers.any());
            assertEquals(accepted ? org.flexlb.balance.PlacementResult.Status.SUCCESS
                    : org.flexlb.balance.PlacementResult.Status.BLOCKED, placement.status());
            assertEquals(replacement.get(), offered.getValue().decodeReservation());
            assertSame(future, offered.getValue().future());
            assertSame(context, offered.getValue().ctx());
            if (accepted) {
                assertSame(offered.getValue(), registry.findRequestContext(100).activeRoute());
                assertFalse(future.isDone());
                assertEquals(replacement.get(), EndpointTestSupport.decodeReservation(decode, 100));
            } else {
                assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
                assertEquals(replacement.get(), EndpointTestSupport.decodeReservation(decode, 100),
                        "the uncommitted route still owns its Decode reservation");
            }
        }
        assertEquals(accepted ? 1 : 0, decode.routingView().totalLoad());
        if (!accepted) { assertNull(EndpointTestSupport.decodeReservation(decode, 100)); }
        assertNull(EndpointTestSupport.decodeReservation(decode, 40));
        assertFalse(victim.future().isDone());
        verify(queue).requeue(victim);
        verify(selection).close();
    }

    @Test
    void replacementKeepsRequestAliveAndTransfersCapacityBeforeRequeue() {
        var item = queued(1);
        doAnswer(call -> {
            assertNull(EndpointTestSupport.decodeReservation(decode, 1));
            assertNotNull(EndpointTestSupport.decodeReservation(decode, 100));
            assertNull(registry.findRequestContext(1).activeRoute());
            assertFalse(item.future().isDone());
            return true;
        }).when(queue).requeue(item);
        assertTrue(replace(item));
        assertEquals(1, decode.routingView().totalLoad());
        assertEquals(RequestState.Phase.QUEUED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(1, 0).state());
        verify(queue).requeue(item);
        assertFalse(item.future().isDone());
        try (var pin = decode.tryPinGeneration()) {
            assertNull(decode.tryReserveQueuedRequest(pin, 1, 16, 16, 30, capacity),
                    "victim cannot reclaim capacity already assigned to the incoming request");
        }
        decode.release(EndpointTestSupport.decodeReservation(decode, 100), DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        DecodeResources.ReservationHandle second;
        try (var pin = decode.tryPinGeneration()) { second = decode.tryReserveQueuedRequest(pin, 1, 16, 16, 30, capacity); }
        var next = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(item.ctx()), new Response(), item.prefill(), item.decode(),
                item.prefillEp(), decode, second, item.enqueuedAtMs() + 1000);
        assertEquals(item.enqueuedAtMs(), next.enqueuedAtMs());
        assertEquals(item.enqueueSeq(), next.enqueueSeq());
        assertEquals(item.expiresAtMs(), next.expiresAtMs());
        try (var admission = registry.claimAdmissionHandle(1, item.future()); var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue((registry.commitRoute(next, RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
        RequestProtocolTestSupport.applyDecodeStatus(registry, decode, DecodeResources.DecodeRequestStatus.terminal(item.decodeReservation(), 0));
        assertSame(next, registry.findRequestContext(1).activeRoute());
        assertFalse(item.future().isDone(), "old reservation evidence must not terminate the new route");
    }

    @Test
    void preparedPermitPreventsWithdrawalAndLeavesOriginalRouteUsable() {
        var item = queued(2);
        clearInvocations(item.prefillEp());
        var permit = decode.acquireDispatchPermit(item.decodeReservation(), capacity).permit();
        assertNotNull(permit);
        assertFalse(replace(item));
        verify(item.prefillEp()).signalRouteReady();
        assertSame(item, registry.findRequestContext(2).activeRoute());
        assertFalse(item.future().isDone());
        assertTrue(RequestProtocolTestSupport.prepareMember(registry, item));
        verify(queue, never()).requeue(any());
        assertTrue(permit.release());
    }

    @Test
    void equalPriorityCannotBeWithdrawn() {
        var item = queued(3);
        assertNull(org.flexlb.balance.scheduler.SchedulerTestSupport.eviction(registry).replaceQueuedDecodeReservations(decode, List.of(item.decodeReservation()),
                100, 16, 16, 30, capacity));
        assertSame(item, registry.findRequestContext(3).activeRoute());
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 3));
        verify(queue, never()).requeue(any());
    }

    @ParameterizedTest
    @EnumSource(value = CancelReason.class, names = {"CLIENT_CANCELLED", "DEADLINE_EXCEEDED"})
    void cancellationDuringWithdrawalSettlesOriginalFutureWithoutRequeue(CancelReason reason) throws Exception {
        var item = queued(4);
        var prefill = item.prefillEp();
        doAnswer(call -> {
            assertFalse(RequestProtocolTestSupport.prepareMember(registry, item),
                    "withdrawal must fence batch preparation before releasing the old route");
            registry.cancel(4, 0, reason);
            return true;
        }).when(prefill).removeQueued(eq(item), anyString());
        assertTrue(replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(EndpointTestSupport.decodeReservation(decode, 4));
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 100));
        verify(queue, never()).requeue(item);
    }

    @Test
    void failedReplacementDoesNotRemoveOrRequeueVictim() {
        var item = queued(5);
        var stale = new DecodeResources.ReservationHandle(item.decodeReservation().endpointGenerationId(),
                5, item.decodeReservation().reservationToken() + 1);
        assertNull(org.flexlb.balance.scheduler.SchedulerTestSupport.eviction(registry).replaceQueuedDecodeReservations(decode, List.of(stale), 100, 16, 16, 80, capacity));
        assertSame(item, registry.findRequestContext(5).activeRoute());
        assertTrue(RequestProtocolTestSupport.prepareMember(registry, item));
        verify(item.prefillEp(), never()).removeQueued(any(), anyString());
    }

    @Test
    void queueShutdownDuringWithdrawalTerminatesVictimInsteadOfLeaking() throws Exception {
        var item = queued(6);
        when(queue.requeue(item)).thenReturn(false);
        assertTrue(replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        assertNull(EndpointTestSupport.decodeReservation(decode, 6));
    }
    @Test
    void laterVictimConflictReleasesEarlierWithdrawalClaim() {
        var item = queued(7);
        var missing = new DecodeResources.ReservationHandle(
                item.decodeReservation().endpointGenerationId(), 999, 1);
        assertNull(org.flexlb.balance.scheduler.SchedulerTestSupport.eviction(registry).replaceQueuedDecodeReservations(decode,
                List.of(item.decodeReservation(), missing), 100, 16, 16, 80, capacity));
        assertSame(item, registry.findRequestContext(7).activeRoute());
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 7));
        assertNull(EndpointTestSupport.decodeReservation(decode, 100));
        assertTrue(RequestProtocolTestSupport.prepareMember(registry, item),
                "an aborted multi-victim plan must not leave earlier victims fenced");
        verify(queue, never()).requeue(any());
    }

    @Test
    void requeueFailureTerminatesVictimAndReleasesIncomingReservation() throws Exception {
        var item = queued(8);
        when(queue.requeue(item)).thenThrow(new IllegalStateException("injected requeue failure"));
        assertThrows(IllegalStateException.class, () -> replace(item));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).liveRequestCount());
        assertEquals(0, decode.routingView().totalLoad());
    }

    @Test
    void failedWithdrawalDoesNotReleaseReusedRequestId() throws Exception {
        var item = queued(8);
        var replacement = new java.util.concurrent.atomic.AtomicReference<DecodeResources.ReservationHandle>();
        var failure = new IllegalStateException("injected requeue failure after reservation replacement");
        doAnswer(call -> {
            var original = EndpointTestSupport.decodeReservation(decode, 100);
            assertNotNull(original);
            decode.release(original, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            try (var pin = decode.tryPinGeneration()) {
                replacement.set(decode.tryReserveQueuedRequest(pin, 100, 16, 16, 80, capacity));
            }
            assertNotNull(replacement.get());
            assertNotEquals(original.reservationToken(), replacement.get().reservationToken());
            throw failure;
        }).when(queue).requeue(item);

        assertSame(failure, assertThrows(IllegalStateException.class, () -> replace(item)));
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(replacement.get(), EndpointTestSupport.decodeReservation(decode, 100));
        assertEquals(1, decode.routingView().totalLoad());
        decode.release(replacement.get(), DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }

    @Test
    void withdrawalBlocksConcurrentDispatchAndSettlesConcurrentCancellation() throws Exception {
        var item = queued(9);
        var prefill = item.prefillEp();
        var removing = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        doAnswer(call -> {
            removing.countDown();
            assertTrue(resume.await(5, TimeUnit.SECONDS));
            return true;
        }).when(prefill).removeQueued(eq(item), anyString());
        try (var executor = Executors.newSingleThreadExecutor()) {
            var replacement = executor.submit(() -> replace(item));
            try {
                assertTrue(removing.await(5, TimeUnit.SECONDS));
                assertFalse(RequestProtocolTestSupport.prepareMember(registry, item));
                registry.cancel(9, 0, CancelReason.CLIENT_CANCELLED);
                assertFalse(item.future().isDone(), "cancellation waits for withdrawal ownership to close");
            } finally {
                resume.countDown();
            }
            assertTrue(replacement.get(5, TimeUnit.SECONDS));
        }
        assertFalse(item.future().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).requeue(item);
        assertNull(EndpointTestSupport.decodeReservation(decode, 9));
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 100));
    }

    @ParameterizedTest
    @ValueSource(strings = {"foreignEndpoint", "foreignRequest", "foreignGeneration", "foreignToken", "equalPriority", "lowerPriority"})
    void withdrawalEligibilityRejectsEveryForeignIdentityWithoutTouchingResources(String mismatch) {
        RequestRoute item = queued(91L);
        DecodeEndpoint source = mismatch.equals("foreignEndpoint") ? RequestProtocolTestSupport.decodeEndpoint() : decode;
        var reservation = item.decodeReservation();
        var attempted = switch (mismatch) {
            case "foreignRequest" -> new DecodeResources.ReservationHandle(reservation.endpointGenerationId(), 92L, reservation.reservationToken());
            case "foreignGeneration" -> new DecodeResources.ReservationHandle(reservation.endpointGenerationId() + 1, 91L, reservation.reservationToken());
            case "foreignToken" -> new DecodeResources.ReservationHandle(reservation.endpointGenerationId(), 91L, reservation.reservationToken() + 1);
            default -> reservation;
        };
        int priority = mismatch.equals("equalPriority") ? 30 : mismatch.equals("lowerPriority") ? 29 : 80;
        assertNull(registry.claimQueuedRoute(source, attempted, priority));
        assertSame(item, item.ctx().activeRoute());
        assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, item.ctx().stage());
        assertEquals(reservation, EndpointTestSupport.decodeReservation(decode, 91L));
        assertEquals(1, decode.routingView().totalLoad());
        assertFalse(item.future().isDone());
        verify(item.prefillEp(), never()).removeQueued(any(), anyString());
        verify(queue, never()).requeue(any());
        try (var valid = registry.claimQueuedRoute(decode, reservation, 80)) {
            assertNotNull(valid, "rejected attempts must leave the valid withdrawal qualification available");
            registry.completeWithdrawal(valid, false);
        }
        assertEquals(RequestContext.RequestStage.READY_TO_DELIVER, item.ctx().stage());
        assertEquals(reservation, EndpointTestSupport.decodeReservation(decode, 91L));
    }

}
