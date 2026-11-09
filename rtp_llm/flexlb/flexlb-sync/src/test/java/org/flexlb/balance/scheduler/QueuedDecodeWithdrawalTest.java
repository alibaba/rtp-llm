package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.eviction.DefaultDecodePreemption;
import org.flexlb.balance.eviction.PriorityDecodePreemptionPolicy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Exercises queued victim withdrawal, incoming capacity transfer, and rollback under concurrent cancellation.
 */
class QueuedDecodeWithdrawalTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;
    private QueuedRequestScheduler queue;
    private DecodePreemption eviction;
    private DecodeEndpoint decode;
    private DecodeResources.AdmissionCapacity capacity =
            new DecodeResources.AdmissionCapacity(1, 95);

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        config.getRouter()
                .getRoles()
                .getDecode()
                .getAvailability()
                .setMaxEngineRequests(capacity.maxEngineRequests());
        config.getRouter()
                .getRoles()
                .getDecode()
                .getAvailability()
                .setMaxKvUsagePercent(capacity.maxKvUsagePercent());
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        queue =
                (QueuedRequestScheduler)
                        org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(registry);
        eviction =
                new DefaultDecodePreemption(
                        SchedulerTestSupport.dispatcher(SchedulerTestSupport.runtime(registry)),
                        SchedulerTestSupport.repository(registry),
                        SchedulerTestSupport.runtime(registry).tasks(),
                        SchedulerTestSupport.runtime(registry).taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        service,
                        registry.dispatcher.decodeAdmissionCapacity,
                        new PriorityDecodePreemptionPolicy());
        org.mockito.Mockito.doReturn(true).when(queue).enqueue(any());
        decode =
                EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, null, "127.0.0.1", 8000, 8001, null),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                mock(AbstractRequestScheduler.class)));
    }

    @AfterEach
    void close() {
        if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(registry)) {
            registry.closeOutstandingAndTerminalize();
        }
        SchedulerTestSupport.timer(
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry))
                .close();
        SchedulerTestSupport.closeRequestExecutors(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void evictionUsesSharedRouteCommitAndReleasesFailedPlacement(boolean accepted)
            throws Exception {
        decode = org.mockito.Mockito.spy(decode);
        var successfulProducerRollbacks = new java.util.concurrent.atomic.AtomicInteger();
        doAnswer(
                        call -> {
                            var result = (DecodeResources.ResourceChange<?>) call.callRealMethod();
                            if (result.status()
                                    == DecodeResources.ReservationReleaseResult.RELEASED) {
                                successfulProducerRollbacks.incrementAndGet();
                            }
                            return result;
                        })
                .when(decode)
                .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        SchedulingTestConfig.allowVictim(config, VictimStage.DECODE_RESERVED);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        var victim = queued(40);
        var context = RequestProtocolTestSupport.context(config, 100);
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(80, System.currentTimeMillis() + 60_000L));
        var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
        context.data().setFuture(future);
        var prefill = mock(PrefillEndpoint.class);
        var prefillWorker = mock(org.flexlb.dao.master.WorkerStatus.class);
        when(prefillWorker.isAvailableForScheduling()).thenReturn(true);
        when(prefill.getStatus()).thenReturn(prefillWorker);
        when(prefill.acceptsNewRequests()).thenReturn(true);
        var selectedEndpoint = prefill;
        var prefillStatus = new ServerStatus();
        prefillStatus.setRole(RoleType.PREFILL);
        prefillStatus.setRequestId(String.valueOf(100));
        prefillStatus.setSuccess(true);
        prefillStatus.setServerIp("127.0.0.1");
        prefillStatus.setHttpPort(9000);
        when(prefill.getIp()).thenReturn("127.0.0.1");
        when(prefill.getHttpPort()).thenReturn(9000);

        when(selectedEndpoint.isRetired()).thenReturn(false);
        when(prefill.ipPort()).thenReturn("127.0.0.1:9000");

        var decodeStatus = new ServerStatus();
        decodeStatus.setSuccess(true);
        decodeStatus.setRole(RoleType.DECODE);
        decodeStatus.setRequestId(String.valueOf(100));
        decodeStatus.setServerIp("127.0.0.1");
        decodeStatus.setHttpPort(8000);
        var replacement =
                new java.util.concurrent.atomic.AtomicReference<
                        DecodeResources.RequestReservation>();
        doAnswer(
                        call -> {
                            replacement.set(EndpointTestSupport.decodeReservation(decode, 100));
                            assertNotNull(replacement.get());
                            return true;
                        })
                .when(queue)
                .enqueue(victim.ctx());
        when(prefill.enqueueRoute(any(), org.mockito.ArgumentMatchers.any()))
                .thenAnswer(
                        call ->
                                accepted
                                        && RequestProtocolTestSupport.recordPrefillReservation(
                                                call.getArgument(0)));
        var manager =
                new DefaultDecodePreemption(
                        SchedulerTestSupport.dispatcher(SchedulerTestSupport.runtime(registry)),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).tasks(),
                        org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)
                                .taskTimer(),
                        mock(RequestSchedulerReporter.class),
                        SchedulingTestConfig.configService(config),
                        SchedulingTestConfig.capacity(config),
                        new PriorityDecodePreemptionPolicy());
        var handle = SchedulerTestSupport.claim(registry, "100", future);
        try (var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(handle)) {
            assertNotNull(handle);
            var reservation = manager.tryReclaim(context, decode);
            assertNotNull(reservation);
            var result = reservation.join();
            assertEquals(replacement.get(), result.reservation());
            var prefillRecord = mock(PrefillEndpoint.RequestRecord.class);
            var completedSelection =
                    new RouteResult(
                            context,
                            java.util.List.of(
                                    new RoleSelection.Prefill(
                                            prefill, prefillStatus, 1L, prefillRecord),
                                    new RoleSelection.Decode(
                                            decode, decodeStatus, result.reservation())));
            when(prefillRecord.belongsTo(completedSelection)).thenReturn(true);
            try (var plan = RequestProtocolTestSupport.plan(context, completedSelection)) {
                var scheduler =
                        (QueuedRequestScheduler)
                                SchedulerTestServices.scheduler(
                                        context.data().getRequestScheduler());
                assertTrue(result.committed());
                var placement = SchedulerTestSupport.enqueueRoute(scheduler, plan);
                var offered = org.mockito.ArgumentCaptor.forClass(RouteResult.class);
                verify(prefill).enqueueRoute(offered.capture(), org.mockito.ArgumentMatchers.any());
                assertEquals(
                        accepted
                                ? org.flexlb.balance.PlacementResult.Status.SUCCESS
                                : org.flexlb.balance.PlacementResult.Status.BLOCKED,
                        placement.status());
                assertEquals(replacement.get(), offered.getValue().decodeReservation());
                assertSame(future, offered.getValue().ctx().data().getFuture());
                assertSame(context, offered.getValue().ctx());
                if (accepted) {
                    assertSame(
                            offered.getValue(),
                            SchedulerTestSupport.repository(registry)
                                    .findActive("100")
                                    .snapshot()
                                    .route());
                    assertFalse(future.isDone());
                    assertEquals(
                            replacement.get(), EndpointTestSupport.decodeReservation(decode, 100));
                } else {
                    assertEquals(
                            replacement.get(),
                            EndpointTestSupport.decodeReservation(decode, 100),
                            "the uncommitted route still owns its Decode reservation");
                    assertFalse(
                            future.isDone(),
                            "the producer must finish its real rollback before terminating admission");
                    plan.close();
                    assertNull(EndpointTestSupport.decodeReservation(decode, 100));
                    registry.publish(
                            context,
                            Response.buildErrorResponse(
                                    StrategyErrorType.RESOURCE_EXHAUSTED,
                                    "selected Prefill capacity changed before canonical placement"),
                            false);
                    assertFalse(future.get(2, TimeUnit.SECONDS).isSuccess());
                }
            }
            assertEquals(
                    accepted ? 0 : 1,
                    successfulProducerRollbacks.get(),
                    "closing the already resolved plan cannot release the replacement reservation"
                            + " twice");
            assertEquals(accepted ? 1 : 0, decode.routingView().totalLoad());
            if (!accepted) {
                assertNull(EndpointTestSupport.decodeReservation(decode, 100));
            }
            assertNull(EndpointTestSupport.decodeReservation(decode, 40));
            assertFalse(victim.ctx().data().getFuture().isDone());
            verify(queue).enqueue(victim.ctx());
            verify(selectedEndpoint, org.mockito.Mockito.never()).close();
        }
    }

    @Test
    void replacementKeepsRequestAliveAndTransfersCapacityBeforeRequeue() {
        var item = queued(1);
        doAnswer(
                        call -> {
                            assertNull(EndpointTestSupport.decodeReservation(decode, 1));
                            assertNotNull(EndpointTestSupport.decodeReservation(decode, 100));
                            assertNull(
                                    SchedulerTestSupport.repository(registry)
                                            .findActive("1")
                                            .snapshot()
                                            .route());
                            assertFalse(item.ctx().data().getFuture().isDone());
                            return true;
                        })
                .when(queue)
                .enqueue(item.ctx());
        assertTrue(replace(item));
        assertEquals(1, decode.routingView().totalLoad());
        assertEquals(
                RequestState.Phase.QUEUED,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                        .getRequestState("1", 0)
                        .state());
        verify(queue).enqueue(item.ctx());
        assertFalse(item.ctx().data().getFuture().isDone());
        {
            var selectedEndpoint = decode;

            assertNull(
                    decode.tryRegisterSelection(
                                    decode.routingView(), String.valueOf(1), 16, 16, 30, capacity)
                            .value(),
                    "victim cannot reclaim capacity already assigned to the incoming request");
        }
        decode.release(
                        EndpointTestSupport.decodeReservation(decode, 100),
                        DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                .status();
        DecodeResources.RequestReservation second;
        {
            var selectedEndpoint = decode;

            second =
                    decode.tryRegisterSelection(
                                    decode.routingView(), String.valueOf(1), 16, 16, 30, capacity)
                            .value();
        }
        var next =
                SchedulingTestConfig.createRoute(
                        item.ctx(),
                        new Response(),
                        item.role(org.flexlb.dao.route.RoleType.PREFILL)
                                .toServerStatus(item.requestId()),
                        item.role(org.flexlb.dao.route.RoleType.DECODE)
                                .toServerStatus(item.requestId()),
                        item.prefillEndpoint(),
                        decode,
                        second,
                        item.enqueuedAtMs() + 1000);
        assertEquals(item.enqueuedAtMs(), next.enqueuedAtMs());
        assertEquals(item.enqueueSeq(), next.enqueueSeq());
        assertEquals(item.expiresAtMs(), next.expiresAtMs());
        var admission = SchedulerTestSupport.claim(registry, "1", item.ctx().data().getFuture());
        try (var admissionCompletion3 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue(
                    (RequestProtocolTestSupport.commitRoute(
                                    registry,
                                    next,
                                    RequestProtocolTestSupport.publication(() -> true))
                            == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
        RequestProtocolTestSupport.applyDecodeStatus(
                registry,
                decode,
                DecodeResources.DecodeRequestStatus.terminal(item.decodeReservation(), 0));
        assertSame(
                next, SchedulerTestSupport.repository(registry).findActive("1").snapshot().route());
        assertFalse(
                item.ctx().data().getFuture().isDone(),
                "old reservation evidence must not terminate the new route");
    }

    @Test
    void preparedPermitPreventsWithdrawalAndLeavesOriginalRouteUsable() {
        var item = queued(2);
        clearInvocations(item.prefillEndpoint());
        var permit = decode.acquireDispatchPermit(item.decodeReservation(), capacity).value();
        assertNotNull(permit);
        assertFalse(replace(item));
        verify(item.prefillEndpoint(), never())
                .signal(org.flexlb.balance.endpoint.PrefillEndpoint.Signal.INPUT);
        assertEquals(ContextResult.Phase.OPEN, item.ctx().snapshot().phase());
        assertNotNull(item.ctx().snapshot().route());
        assertNull(item.ctx().snapshot().delivery());
        assertSame(
                item, SchedulerTestSupport.repository(registry).findActive("2").snapshot().route());
        assertFalse(item.ctx().data().getFuture().isDone());
        assertTrue(RequestProtocolTestSupport.prepareMember(registry, item));
        verify(queue, never()).enqueue(any());
        assertTrue(permit.release());
    }

    @Test
    void equalPriorityCannotBeWithdrawn() {
        var item = queued(3);
        assertNull(replace(List.of(item.decodeReservation()), 30));
        assertSame(
                item, SchedulerTestSupport.repository(registry).findActive("3").snapshot().route());
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 3));
        verify(queue, never()).enqueue(any());
    }

    @ParameterizedTest
    @EnumSource(
            value = CancelReason.class,
            names = {"CLIENT_CANCELLED", "DEADLINE_EXCEEDED"})
    void cancellationDuringWithdrawalSettlesOriginalFutureWithoutRequeue(CancelReason reason)
            throws Exception {
        var item = queued(4);
        var prefill = item.prefillEndpoint();
        doAnswer(
                        call -> {
                            assertFalse(
                                    RequestProtocolTestSupport.prepareMember(registry, item),
                                    "withdrawal must fence batch preparation before releasing the"
                                            + " old route");
                            registry.cancel("4", 0, reason);
                            return true;
                        })
                .when(prefill)
                .removeQueued(eq(item));
        assertTrue(replace(item));
        assertFalse(item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(EndpointTestSupport.decodeReservation(decode, 4));
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 100));
        verify(queue, never()).enqueue(item.ctx());
    }

    @Test
    void terminalFailureDuringWithdrawalClosesWithoutCancellationOrRequeue() throws Exception {
        var item = queued(11);
        var prefill = item.prefillEndpoint();
        doAnswer(
                        call -> {
                            registry.closeRequest(
                                    item.ctx(),
                                    new ContextResult.EndEvidence(
                                            item,
                                            ContextResult.EndKind.DELIVERY_FAILURE,
                                            null,
                                            false,
                                            0L,
                                            StrategyErrorType.DISPATCH_FAILED,
                                            "terminal failure during withdrawal",
                                            false,
                                            false));
                            var closing = item.ctx().snapshot();
                            assertEquals(ContextResult.Phase.ENDING, closing.phase());
                            assertNull(closing.cancellation());
                            assertNotNull(
                                    closing.placement(),
                                    "physical withdrawal still owns its placement scope");
                            return true;
                        })
                .when(prefill)
                .removeQueued(eq(item));

        assertTrue(replace(item));
        var response = item.ctx().data().getFuture().get(2, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(), response.getCode());
        var closed = item.ctx().snapshot();
        assertEquals(ContextResult.Phase.CLOSED, closed.phase());
        assertNull(closed.cancellation());
        assertEquals("terminal failure during withdrawal", closed.outcome().detail());
        assertNull(EndpointTestSupport.decodeReservation(decode, 11));
        assertNotNull(
                EndpointTestSupport.decodeReservation(decode, 100),
                "closing the withdrawn victim must preserve the incoming reservation");
        verify(queue, never()).enqueue(item.ctx());
        verify(prefill).removeQueued(eq(item));
    }

    @Test
    void failedReplacementDoesNotRemoveOrRequeueVictim() {
        var item = queued(5);
        var stale =
                new DecodeResources.RequestReservation(
                        item.decodeReservation().endpointGenerationId(),
                        String.valueOf(5),
                        item.decodeReservation().reservationToken() + 1);
        assertNull(replace(List.of(stale), 80));
        assertSame(
                item, SchedulerTestSupport.repository(registry).findActive("5").snapshot().route());
        assertTrue(RequestProtocolTestSupport.prepareMember(registry, item));
        verify(item.prefillEndpoint(), never()).removeQueued(any());
    }

    @Test
    void queueShutdownDuringWithdrawalTerminatesVictimInsteadOfLeaking() throws Exception {
        var item = queued(6);
        when(queue.enqueue(item.ctx())).thenReturn(false);
        assertTrue(replace(item));
        assertFalse(item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                        .liveRequestCount());
        assertNull(EndpointTestSupport.decodeReservation(decode, 6));
    }

    @Test
    void laterVictimConflictReleasesEarlierWithdrawalClaim() {
        var item = queued(7);
        var missing =
                new DecodeResources.RequestReservation(
                        item.decodeReservation().endpointGenerationId(), String.valueOf(999), 1);
        assertNull(replace(List.of(item.decodeReservation(), missing), 80));
        assertSame(
                item, SchedulerTestSupport.repository(registry).findActive("7").snapshot().route());
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 7));
        assertNull(EndpointTestSupport.decodeReservation(decode, 100));
        assertTrue(
                RequestProtocolTestSupport.prepareMember(registry, item),
                "an aborted multi-victim plan must not leave earlier victims fenced");
        verify(queue, never()).enqueue(any());
    }

    @Test
    void requeueFailureTerminatesVictimAndReleasesIncomingReservation() throws Exception {
        var item = queued(8);
        when(queue.enqueue(item.ctx()))
                .thenThrow(new IllegalStateException("injected requeue failure"));
        assertThrows(IllegalStateException.class, () -> replace(item));
        assertFalse(item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(
                0,
                org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry)
                        .liveRequestCount());
        assertEquals(0, decode.routingView().totalLoad());
    }

    @Test
    void prefillRemovalFailureStillFinishesOtherVictimsAndReleasesIncoming() throws Exception {
        capacity = new DecodeResources.AdmissionCapacity(2, 95);
        decode = org.mockito.Mockito.spy(decode);
        var releaseCount = new java.util.concurrent.atomic.AtomicInteger();
        doAnswer(
                        call -> {
                            releaseCount.incrementAndGet();
                            return call.callRealMethod();
                        })
                .when(decode)
                .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        var first = queued(81);
        var second = queued(82);
        when(first.prefillEndpoint().removeQueued(first))
                .thenThrow(new IllegalStateException("injected Prefill removal failure"));
        assertThrows(
                IllegalStateException.class,
                () -> replace(List.of(first.decodeReservation(), second.decodeReservation()), 80));
        assertFalse(first.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertNull(EndpointTestSupport.decodeReservation(decode, 100));
        assertNull(EndpointTestSupport.decodeReservation(decode, 81));
        assertNull(EndpointTestSupport.decodeReservation(decode, 82));
        assertNull(second.ctx().snapshot().placement());
        assertNull(second.ctx().snapshot().route());
        verify(queue).enqueue(second.ctx());
        verify(queue, never()).enqueue(first.ctx());
        assertFalse(RequestProtocolTestSupport.prepareMember(registry, first));
        assertEquals(
                1,
                releaseCount.get(),
                "Scheduler failure must release the incoming reservation once");
    }

    @Test
    void multipleWithdrawalFailuresReleaseIncomingOnceAndKeepCleanupEvidence() throws Exception {
        capacity = new DecodeResources.AdmissionCapacity(2, 95);
        decode = org.mockito.Mockito.spy(decode);
        var first = queued(83);
        var second = queued(84);
        var firstFailure = new IllegalStateException("first withdrawal failed");
        var secondFailure = new IllegalStateException("second withdrawal failed");
        var cleanupFailure = new IllegalStateException("release notification failed");
        when(first.prefillEndpoint().removeQueued(first)).thenThrow(firstFailure);
        when(second.prefillEndpoint().removeQueued(second)).thenThrow(secondFailure);
        doAnswer(
                        call -> {
                            call.callRealMethod();
                            throw cleanupFailure;
                        })
                .when(decode)
                .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));

        var failure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                replace(
                                        List.of(
                                                first.decodeReservation(),
                                                second.decodeReservation()),
                                        80));

        assertSame(firstFailure, failure);
        assertTrue(List.of(failure.getSuppressed()).contains(secondFailure));
        assertTrue(List.of(failure.getSuppressed()).contains(cleanupFailure));
        verify(decode).release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        assertNull(first.ctx().snapshot().placement());
        assertNull(second.ctx().snapshot().placement());
        assertNull(EndpointTestSupport.decodeReservation(decode, 100));
        assertFalse(first.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertFalse(second.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(0, decode.routingView().totalLoad());
    }

    @Test
    void failedWithdrawalDoesNotReleaseReusedRequestId() throws Exception {
        var item = queued(8);
        var replacement =
                new java.util.concurrent.atomic.AtomicReference<
                        DecodeResources.RequestReservation>();
        var failure =
                new IllegalStateException("injected requeue failure after reservation replacement");
        doAnswer(
                        call -> {
                            var original = EndpointTestSupport.decodeReservation(decode, 100);
                            assertNotNull(original);
                            decode.release(original, DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                                    .status();
                            {
                                var selectedEndpoint = decode;

                                replacement.set(
                                        decode.tryRegisterSelection(
                                                        decode.routingView(),
                                                        String.valueOf(100),
                                                        16,
                                                        16,
                                                        80,
                                                        capacity)
                                                .value());
                            }
                            assertNotNull(replacement.get());
                            assertNotEquals(
                                    original.reservationToken(),
                                    replacement.get().reservationToken());
                            throw failure;
                        })
                .when(queue)
                .enqueue(item.ctx());

        assertSame(failure, assertThrows(IllegalStateException.class, () -> replace(item)));
        assertFalse(item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        assertEquals(replacement.get(), EndpointTestSupport.decodeReservation(decode, 100));
        assertEquals(1, decode.routingView().totalLoad());
        decode.release(replacement.get(), DecodeResources.ReleaseReason.LOCAL_ROLLBACK).status();
    }

    @Test
    void withdrawalBlocksConcurrentDispatchAndSettlesConcurrentCancellation() throws Exception {
        var item = queued(9);
        var prefill = item.prefillEndpoint();
        var removing = new CountDownLatch(1);
        var resume = new CountDownLatch(1);
        doAnswer(
                        call -> {
                            removing.countDown();
                            assertTrue(resume.await(5, TimeUnit.SECONDS));
                            return true;
                        })
                .when(prefill)
                .removeQueued(eq(item));
        try (var executor = Executors.newSingleThreadExecutor()) {
            var replacement = executor.submit(() -> replace(item));
            try {
                assertTrue(removing.await(5, TimeUnit.SECONDS));
                assertFalse(RequestProtocolTestSupport.prepareMember(registry, item));
                registry.cancel("9", 0, CancelReason.CLIENT_CANCELLED);
                assertFalse(
                        item.ctx().data().getFuture().isDone(),
                        "cancellation waits for withdrawal ownership to close");
            } finally {
                resume.countDown();
            }
            assertTrue(replacement.get(5, TimeUnit.SECONDS));
        }
        assertFalse(item.ctx().data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        verify(queue, never()).enqueue(item.ctx());
        assertNull(EndpointTestSupport.decodeReservation(decode, 9));
        assertNotNull(EndpointTestSupport.decodeReservation(decode, 100));
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "foreignEndpoint",
                "foreignRequest",
                "foreignGeneration",
                "foreignToken",
                "equalPriority",
                "lowerPriority"
            })
    void withdrawalEligibilityRejectsEveryForeignIdentityWithoutTouchingResources(String mismatch) {
        RouteResult item = queued(91L);
        DecodeEndpoint source =
                mismatch.equals("foreignEndpoint")
                        ? RequestProtocolTestSupport.decodeEndpoint()
                        : decode;
        var reservation = item.decodeReservation();
        var attempted =
                switch (mismatch) {
                    case "foreignRequest" ->
                            new DecodeResources.RequestReservation(
                                    reservation.endpointGenerationId(),
                                    String.valueOf(92L),
                                    reservation.reservationToken());
                    case "foreignGeneration" ->
                            new DecodeResources.RequestReservation(
                                    reservation.endpointGenerationId() + 1,
                                    String.valueOf(91L),
                                    reservation.reservationToken());
                    case "foreignToken" ->
                            new DecodeResources.RequestReservation(
                                    reservation.endpointGenerationId(),
                                    String.valueOf(91L),
                                    reservation.reservationToken() + 1);
                    default -> reservation;
                };
        int priority =
                mismatch.equals("equalPriority") ? 30 : mismatch.equals("lowerPriority") ? 29 : 80;
        assertNull(registry.beginPreemption(source, attempted, priority));
        assertSame(item, item.ctx().snapshot().route());
        assertEquals(ContextResult.Phase.OPEN, item.ctx().snapshot().phase());
        assertNotNull(item.ctx().snapshot().route());
        assertNull(item.ctx().snapshot().delivery());
        assertEquals(reservation, EndpointTestSupport.decodeReservation(decode, 91L));
        assertEquals(1, decode.routingView().totalLoad());
        assertFalse(item.ctx().data().getFuture().isDone());
        verify(item.prefillEndpoint(), never()).removeQueued(any());
        verify(queue, never()).enqueue(any());
        var valid = registry.beginPreemption(decode, reservation, 80);
        {
            assertNotNull(
                    valid,
                    "rejected attempts must leave the valid withdrawal qualification available");
            AbstractRequestScheduler.completePlacement(
                    valid, ContextResult.PublicationResult.RESTORED);
        }
        assertEquals(ContextResult.Phase.OPEN, item.ctx().snapshot().phase());
        assertNotNull(item.ctx().snapshot().route());
        assertNull(item.ctx().snapshot().delivery());
        assertEquals(reservation, EndpointTestSupport.decodeReservation(decode, 91L));
    }

    private RouteResult queued(long id) {
        var context = RequestProtocolTestSupport.context(config, id);
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(30, System.currentTimeMillis() + 60_000L));
        var future = RequestProtocolTestSupport.submitForManualPlacement(registry, context);
        context.data().setFuture(future);
        DecodeResources.RequestReservation reservation;
        {
            var selectedEndpoint = decode;

            reservation =
                    decode.tryRegisterSelection(
                                    decode.routingView(), String.valueOf(id), 16, 16, 30, capacity)
                            .value();
        }
        assertNotNull(reservation);
        var prefill = mock(PrefillEndpoint.class);
        var prefillWorker = mock(org.flexlb.dao.master.WorkerStatus.class);
        when(prefillWorker.isAvailableForScheduling()).thenReturn(true);
        when(prefill.getStatus()).thenReturn(prefillWorker);
        when(prefill.acceptsNewRequests()).thenReturn(true);
        when(prefill.removeQueued(any())).thenReturn(true);
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        new ServerStatus(),
                        new ServerStatus(),
                        prefill,
                        decode,
                        reservation,
                        System.currentTimeMillis());
        var admission = SchedulerTestSupport.claim(registry, String.valueOf(id), future);
        try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue(
                    (RequestProtocolTestSupport.commitRoute(
                                    registry,
                                    item,
                                    RequestProtocolTestSupport.publication(() -> true))
                            == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
        return item;
    }

    /**
     * Inject exact victim identities to exercise withdrawal conflicts independently of planning.
     */
    private DecodeResources.RequestReservation replace(
            List<DecodeResources.RequestReservation> victims, int priority) {
        var incoming =
                org.flexlb.dao.loadbalance.Request.builder()
                        .requestId("100")
                        .seqLen(16L)
                        .maxNewTokens(0)
                        .build();
        return org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                eviction, "replaceQueuedDecodeReservations", decode, victims, incoming, priority);
    }

    private boolean replace(RouteResult item) {
        var incoming = replace(List.of(item.decodeReservation()), 80);
        if (incoming != null) {
            assertEquals(EndpointTestSupport.decodeReservation(decode, 100), incoming);
        }
        return incoming != null;
    }
}
