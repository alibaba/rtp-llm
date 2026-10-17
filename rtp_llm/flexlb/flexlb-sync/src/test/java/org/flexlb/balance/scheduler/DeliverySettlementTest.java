package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CancelRequestPB;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DeliverySettlementTest {
    private FlexlbConfig config;
    private AbstractRequestScheduler registry;
    private DeliverySettlementTestSupport ledger;
    private PrefillEndpoint prefill;
    private final java.util.Map<Long, java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>> cancellations = new java.util.concurrent.ConcurrentHashMap<>();

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        EngineGrpcClient channel = mock(EngineGrpcClient.class);
        org.springframework.test.util.ReflectionTestUtils.setField(registry.runtime, "engineClient", channel);
        when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call ->
                cancellations.computeIfAbsent(call.<CancelRequestPB>getArgument(2).getRequestId(), ignored -> new java.util.concurrent.CompletableFuture<>()));
        ledger = new DeliverySettlementTestSupport();
        prefill = mock(PrefillEndpoint.class);
        when(prefill.release(org.mockito.ArgumentMatchers.isNull(), any())).thenAnswer(invocation -> {
            RequestRoute item = invocation.getArgument(1);
            RequestContext requestContext = registry.requests.findActive(item.requestId());
            assertTrue(requestContext == null || !Thread.holdsLock(requestContext),
                    "endpoint accounting must run outside the context monitor");
            return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(ledger.prefill, item);
        });
    }

    @AfterEach
    void close() {
        // Mock RPCs have no network deadline; finish them before draining accepted callbacks.
        org.mockito.Mockito.doReturn(java.util.concurrent.CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.FAILED))
                .when(SchedulerTestSupport.engineClient(registry)).cancelAsync(any(), anyInt(), any(), anyLong());
        cancellations.values().forEach(reply -> reply.complete(EngineGrpcClient.CancelReply.FAILED));
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
            registry.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(SchedulerTestSupport.runtime(registry)).close();
            SchedulerTestSupport.closeRequestExecutors(SchedulerTestSupport.runtime(registry));
        }
    }

    @Test
    void rejectionRetainsBothLedgersUntilExactRemoteCleanup() throws Exception {
        Member member = member(1L, 11L);
        ledger.commit(11L, List.of(member.item()));
        reject(member);
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(1, 1);
        assertTrue(registry.requests.isCurrent(member.requestContext()));
        cleaned(member);
        assertOccupancy(0, 0);
        assertFalse(registry.requests.isCurrent(member.requestContext()));
        verify(member.item().decodeEp()).release(member.item().decodeReservation(), DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        verify(member.item().decodeEp(), never()).release(any(), eq(DecodeResources.ReleaseReason.NOT_SENT));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void prefillFenceCannotReleaseARealDecodeReservation(boolean observed) throws Exception {
        var decode = spy(EndpointTestSupport.decode(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), registry.requests));
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) { reservation = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, 2L, 1L, 1L, 50); }
        Member member = member(2L, 12L, decode, reservation);
        ledger.commit(12L, List.of(member.item()));
        DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
        if (observed) { DeliverySettlementTestSupport.decodeStatus(decode, 2L, false); }
        reject(member);
        cancellations.computeIfAbsent(2L, ignored -> new java.util.concurrent.CompletableFuture<>()).complete(EngineGrpcClient.CancelReply.REQUEST_FENCED);
        assertEquals(1, decode.routingView().engineCapacityUsed());
        assertOccupancy(1, 1);
        assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone());
        DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        registry.runtime.signals().delivery(member.claim()).toCompletableFuture().get(2, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertOccupancy(0, 0);
    }

    @Test
    void sixteenMemberBatchReleasesOnlyAfterRejectedCleanupAndEverySuccessfulMemberFinish() throws Exception {
        List<Member> members = new ArrayList<>();
        for (long id = 10L; id < 26L; id++) { members.add(member(id, 20L)); }
        ledger.commit(20L, members.stream().map(Member::item).toList());
        reject(members.getFirst());
        assertOccupancy(1, 16);
        cleaned(members.getFirst());
        assertOccupancy(1, 15);
        for (int i = 1; i < members.size(); i++) {
            Member member = members.get(i);
            member.claim().item().ctx().data().getScheduler().onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.delivered(), System.currentTimeMillis()));
            ledger.finish(20L, member.item()).forEach(status -> registry.onEngineEvidence(status.route().ctx(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
            assertOccupancy(i == 15 ? 0 : 1, 15 - i);
        }
        assertTrue(ledger.prefill.batchCapacityAvailable(2));
    }

    @Test
    void repeatedPartialRejectionsReturnBatchPermitsAfterProof() throws Exception {
        for (int i = 0; i < 4; i++) {
            Member rejected = member(30L + 2L * i, 30L + i);
            Member successful = member(31L + 2L * i, 30L + i);
            ledger.commit(30L + i, List.of(rejected.item(), successful.item()));
            reject(rejected);
            successful.claim().item().ctx().data().getScheduler().onDeliveryEvidence(successful.claim(), new ContextResult.TransportResult(successful.claim(), DeliveryResult.delivered(), System.currentTimeMillis()));
            ledger.finish(30L + i, successful.item());
            assertOccupancy(1, 1);
            cleaned(rejected);
            assertOccupancy(0, 0);
            assertTrue(ledger.prefill.batchCapacityAvailable(1));
        }
    }

    @Test
    void concurrentEndpointFinishAndRemoteCleanupSettleOneMemberOnce() throws Exception {
        Member member = member(40L, 40L);
        ledger.commit(40L, List.of(member.item()));
        reject(member);
        CountDownLatch start = new CountDownLatch(1);
        try (var threads = Executors.newFixedThreadPool(2)) {
            var cleanup = threads.submit(() -> { await(start); cancellations.computeIfAbsent(40L, ignored -> new java.util.concurrent.CompletableFuture<>()).complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED); });
            var finished = threads.submit(() -> { await(start); ledger.finish(40L, member.item()); });
            start.countDown();
            cleanup.get(2, TimeUnit.SECONDS);
            finished.get(2, TimeUnit.SECONDS);
        }
        registry.runtime.signals().delivery(member.claim()).toCompletableFuture().get(2, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertStaleDelivery(member, DeliveryResult.prefillRejected(new IllegalStateException("prepare rejected")));
        ledger.finish(40L, member.item());
        assertOccupancy(0, 0);
    }

    @Test
    void delayedFactsCannotReleaseAReusedRequestIdentity() throws Exception {
        Member old = member(50L, 50L);
        ledger.commit(50L, List.of(old.item()));
        reject(old);
        cleaned(old);
        var terminal = registry.requests.findTerminal(50L);
        assertTrue(registry.requests.removeExactTerminal(terminal, Long.MAX_VALUE));
        Member replacement = member(50L, 51L);
        ledger.commit(51L, List.of(replacement.item()));
        decodeFinished(old);
        ledger.finish(50L, old.item()).forEach(status -> registry.onEngineEvidence(status.route().ctx(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(1, 1);
        assertSame(replacement.requestContext(), registry.requests.findActive(50L));
        assertFalse(replacement.item().future().isDone());
    }

    @Test
    void failedLocalCleanupRetainsOwnershipAfterBoundedRetries() throws Exception {
        Member member = member(60L, 60L);
        ledger.commit(60L, List.of(member.item()));
        var failedAttempts = new CountDownLatch(3);
        doAnswer(call -> {
            failedAttempts.countDown();
            throw new IllegalStateException("injected endpoint failure");
        }).when(prefill).release(null, member.item());
        reject(member);
        // A CLEANED ACK proves release is allowed; it does not prove local resource release succeeded.
        cancellations.computeIfAbsent(member.item().requestId(), ignored -> new java.util.concurrent.CompletableFuture<>())
                .complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
        await(failedAttempts);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(1, 1);
        assertTrue(registry.requests.isCurrent(member.requestContext()));
        var failure = SchedulerTestSupport.failure(registry);
        assertNotNull(failure);
        assertTrue(failure.getMessage().contains("injected endpoint failure"));
        var physicalSettlement = registry.runtime.signals().delivery(member.claim()).toCompletableFuture();
        assertFalse(physicalSettlement.isDone(), "physical settlement must retain the failed Prefill release obligation");
        assertFalse(member.requestContext().snapshot().close().prefillSettled());
        assertTrue(member.requestContext().snapshot().close().decodeSettled(), "the successfully released side remains settled");
        verify(prefill, after(150).times(3)).release(null, member.item());
        verify(member.item().decodeEp(), times(1)).release(member.item().decodeReservation(), DecodeResources.ReleaseReason.REMOTE_CLEANUP);

        // Restore the real release operation and resume the same request's remaining resource obligation.
        doAnswer(call -> EndpointTestSupport.releaseRequest(ledger.prefill, call.getArgument(1)))
                .when(prefill).release(null, member.item());
        registry.requestCleanup(member.requestContext());
        physicalSettlement.get(2, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
        assertFalse(registry.requests.isCurrent(member.requestContext()));
        assertNotNull(registry.requests.findTerminal(member.item().requestId()));
        assertTrue(member.requestContext().snapshot().close().prefillSettled());
        assertTrue(member.requestContext().snapshot().close().decodeSettled());
        assertSame(failure, SchedulerTestSupport.failure(registry), "actual recovery must retain the recorded facility failure");
        verify(prefill, times(4)).release(null, member.item());
        verify(member.item().decodeEp(), times(1)).release(member.item().decodeReservation(), DecodeResources.ReleaseReason.REMOTE_CLEANUP);
    }

    @Test
    void cleanedRouteCancellationReleasesAcceptedDecodeWithoutAbandonmentOrRetirement() throws Exception {
        var decode = spy(EndpointTestSupport.decode(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8180, 8181, null), registry.requests));
        try {
            DecodeResources.ReservationHandle reservation;
            try (var pin = decode.tryPinGeneration()) {
                reservation = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, 131L, 1L, 2L, 50);
            }
            var member = member(131L, 131L, decode, reservation, DeliveryClaimKind.ROUTE_DECISION);
            ledger.commit(131L, List.of(member.item()));
            DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
            DeliverySettlementTestSupport.decodeStatus(decode, 131L, false);
            registry.onDeliveryEvidence(member.claim(), new ContextResult.RouteDelivered(member.claim(), 0L, System.currentTimeMillis()));
            assertTrue(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
            assertTrue(decode.ownership(reservation).confirmed());
            assertEquals(1, decode.routingView().engineCapacityUsed());
            registry.cancel(131L, 0L, CancelReason.CLIENT_CANCELLED);
            assertNull(member.requestContext().snapshot().delivery().abandonment(), "route cancellation does not need batch abandonment");
            assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone());
            cleaned(member);
            verify(SchedulerTestSupport.engineClient(registry)).cancelAsync(any(), anyInt(),
                    eq(CancelReason.CLIENT_CANCELLED.toRequest(131L)), eq(500L));
            verify(decode).release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            assertFalse(decode.isRetired(), "actual cleanup must finish before endpoint retirement");
            assertFalse(decode.ownership(reservation).owned());
            assertEquals(0, decode.routingView().engineCapacityUsed());
            assertEquals(0, decode.routingView().inflightHardKv());
            assertOccupancy(0, 0);
            assertFalse(registry.requests.isCurrent(member.requestContext()));
            assertNotNull(registry.requests.findTerminal(131L));
            assertTrue(member.item().future().join().isSuccess(), "cancellation must preserve the first frontend ACK");
            assertNull(SchedulerTestSupport.failure(registry));
        } finally { decode.close(); decode.awaitRetirement(); }
    }

    @ParameterizedTest
    @CsvSource({"NOT_SENT,false", "PREFILL_REJECTED,false", "NOT_SENT,true", "PREFILL_REJECTED,true"})
    void failedDeliveryReleasesOnlyProvedUnacceptedDecodeOwnership(DeliveryResult.Status source, boolean accepted) {
        var decode = spy(EndpointTestSupport.decode(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8180, 8181, null), registry.requests));
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, 91L, 1L, 2L, 50);
        }
        var context = RequestProtocolTestSupport.context(config, 91L);
        var item = SchedulingTestConfig.createRoute(context, new Response(), null, null,
                prefill, decode, reservation, System.currentTimeMillis());
        DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
        if (accepted) { DeliverySettlementTestSupport.decodeStatus(decode, 91L, false); }
        try {
            assertEquals(1, decode.routingView().engineCapacityUsed());
            var settlement = AbstractRequestScheduler.releaseResources(new ContextResult.ClosePermit(context, 1L, item, source == DeliveryResult.Status.NOT_SENT ? DecodeResources.ReleaseReason.NOT_SENT : null, true, false, false, false));
            boolean released = source == DeliveryResult.Status.NOT_SENT && !accepted;
            assertEquals(released, settlement.decodeSettled());
            assertTrue(settlement.prefillSettled());
            org.junit.jupiter.api.Assertions.assertNull(settlement.failure());
            assertEquals(released ? 0 : 1, decode.routingView().engineCapacityUsed());
            assertEquals(!released, decode.ownership(reservation).owned());
            verify(decode, source == DeliveryResult.Status.NOT_SENT ? times(1) : never())
                    .release(reservation, DecodeResources.ReleaseReason.NOT_SENT);
        } finally {
            decode.release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP).status();
            item.close();
        }
    }

    @Test
    void notSentReleasesLocalReservationsOutsideTheRequestLock() throws Exception {
        Member member = member(80L, 80L);
        ledger.commit(80L, List.of(member.item()));
        when(member.item().decodeEp().ownership(member.item().decodeReservation()))
                .thenAnswer(invocation -> { assertFalse(Thread.holdsLock(member.requestContext())); return DecodeResources.Ownership.NONE; });
        member.claim().item().ctx().data().getScheduler().onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.notSent(new IllegalStateException("serialization failed")), System.currentTimeMillis()));
        registry.runtime.signals().delivery(member.claim()).toCompletableFuture().get(2, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(0, 0);
        assertTrue(cancellations.isEmpty());
        verify(member.item().decodeEp()).release(member.item().decodeReservation(), DecodeResources.ReleaseReason.NOT_SENT);
    }

    @Test
    void unknownTransportOutcomeRetainsResourcesAndDoesNotFabricateAcknowledgement() {
        Member member = member(70L, 70L);
        ledger.commit(70L, List.of(member.item()));
        member.claim().item().ctx().data().getScheduler().onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
        assertOccupancy(1, 1);
        assertFalse(member.item().future().isDone());
        assertTrue(cancellations.isEmpty());
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class, names = {"CANCEL_IN_FLIGHT", "NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void rejectionCannotFinishPreemptionBeforeCleanupProof(PreemptionCancelPhase phase) throws Exception {
        Member member = member(90L, 90L);
        ledger.commit(90L, List.of(member.item()));
        PreemptionRegistration preemption = registry.tryClaim(member.item().decodeReservation(), 91L, "priority victim").orElseThrow();
        assertTrue(registry.onPreemption(preemption, new AbstractRequestScheduler.PreemptionEvent(PreemptionCancelPhase.CANCEL_IN_FLIGHT, false, null, null, null)));
        if (phase != PreemptionCancelPhase.CANCEL_IN_FLIGHT) { assertTrue(registry.onPreemption(preemption, new AbstractRequestScheduler.PreemptionEvent(phase, false, null, null, null))); }
        reject(member);
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertFalse(registry.runtime.signals().preemption(preemption).toCompletableFuture().isDone());
        assertOccupancy(1, 1);
        cleaned(member);
        assertTrue(registry.runtime.signals().preemption(preemption).toCompletableFuture().isDone());
        assertOccupancy(0, 0);
    }

    @Test
    void callbackReturnsWhileRemoteProofAndLocalCleanupRunElsewhere() throws Exception {
        Member member = member(99112L, 99112L);
        ledger.commit(99112L, List.of(member.item()));
        CountDownLatch cleanupEntered = new CountDownLatch(1);
        CountDownLatch releaseCleanup = new CountDownLatch(1);
        doAnswer(call -> { cleanupEntered.countDown(); await(releaseCleanup); return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(ledger.prefill, call.getArgument(1)); })
                .when(prefill).release(null, member.item());
        var callback = Executors.newSingleThreadExecutor();
        try {
            callback.submit(() -> member.claim().item().ctx().data().getScheduler().onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.prefillRejected(new IllegalStateException("prepare rejected")), System.currentTimeMillis()))).get(1, TimeUnit.SECONDS);
            assertOccupancy(1, 1);
            RequestProtocolTestSupport.awaitCondition(() -> cancellations.containsKey(member.item().requestId()));
            cancellations.get(member.item().requestId()).complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            assertTrue(cleanupEntered.await(2, TimeUnit.SECONDS));
            assertTrue(registry.requests.isCurrent(member.requestContext()));
        } finally {
            releaseCleanup.countDown();
            callback.shutdownNow();
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
    }

    @ParameterizedTest
    @ValueSource(strings = {"APD", "ADP", "PAD", "PDA", "DAP", "DPA"})
    void normalAckPrefillAndDecodePermutationsSettleRealLedgersExactlyOnce(String order) throws Exception {
        var decode = spy(EndpointTestSupport.decode(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8180, 8181, null), registry.requests));
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) { reservation = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, 70L, 1L, 2L, 50); }
        Member member = member(70L, 70L, decode, reservation);
        ledger.commit(70L, List.of(member.item()));
        DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
        boolean decodeEnded = false;
        for (int index = 0; index < order.length(); index++) {
            switch (order.charAt(index)) {
                case 'A' -> member.claim().item().ctx().data().getScheduler().onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.delivered(), System.currentTimeMillis()));
                case 'P' -> ledger.finish(70L, member.item()).forEach(status -> registry.onEngineEvidence(status.route().ctx(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
                case 'D' -> {
                    DeliverySettlementTestSupport.decodeStatus(decode, 70L, true);
                    decodeEnded = true;
                }
                default -> throw new AssertionError(order);
            }
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
            assertEquals(decodeEnded ? 0 : 1, decode.routingView().engineCapacityUsed());
            if (index == 0 && order.charAt(index) == 'A') {
                assertOccupancy(1, 1);
                assertTrue(registry.requests.isCurrent(member.requestContext()), "ACK alone cannot archive execution ownership");
            }
        }
        var response = member.item().future().get(5, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertOccupancy(0, 0);
        assertEquals(0, decode.routingView().engineCapacityUsed());
        assertEquals(0, decode.routingView().inflightHardKv());
        assertEquals(0, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        assertFalse(registry.requests.isCurrent(member.requestContext()));
        assertTrue(cancellations.isEmpty(), "normal completion must not start Engine cancel");
        DeliverySettlementTestSupport.decodeStatus(decode, 70L, true);
        ledger.finish(70L, member.item()).forEach(status -> registry.onEngineEvidence(status.route().ctx(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertOccupancy(0, 0);
        assertSame(response, member.item().future().join());
        assertStaleDelivery(member, DeliveryResult.delivered());
    }

    @ParameterizedTest(name = "decodeStatusFirst={0}")
    @ValueSource(booleans = {true, false})
    void uncertainDeliveryIsAcknowledgedRegardlessOfDecodeStatusOrder(boolean decodeStatusFirst) {
        Member member = dispatchedMember(122L);
        DeliveryResult unknown = DeliveryResult.uncertain(new IllegalStateException("reply lost"));
        try {
            if (decodeStatusFirst) {
                DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 122L, false);
            } else {
                registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), unknown, System.currentTimeMillis()));
            }
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
            assertFalse(member.item().future().isDone(),
                    "neither Decode acceptance alone nor UNKNOWN alone may acknowledge delivery");

            if (decodeStatusFirst) {
                registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), unknown, System.currentTimeMillis()));
            } else {
                DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 122L, false);
            }
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
            synchronized (member.requestContext()) {
                assertTrue(member.requestContext().snapshot().engine().decodeAccepted(), "the exact Decode status was accepted");
            }

            Response response = assertDoesNotThrow(() -> member.item().future().get(2, TimeUnit.SECONDS),
                    "UNKNOWN plus exact Decode ACTIVE must acknowledge delivery in either event order");
            assertTrue(response.isSuccess());
            assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone(),
                    "acknowledgement must retain execution ownership until Worker terminal");
            assertOccupancy(1, 1);
            assertEquals(1, member.item().decodeEp().routingView().engineCapacityUsed());
            assertTrue(cancellations.isEmpty(), "delivery confirmation must not cancel execution");
        } finally {
            ledger.finish(122L, member.item()).forEach(status ->
                    registry.onEngineEvidence(member.requestContext(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
            DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 122L, true);
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class, names = {"UNSUPPORTED", "FAILED"})
    void cancelledBatchSettlesFromWorkerTerminalsWithoutRemoteCleanupAck(EngineGrpcClient.CancelReply ack)
            throws Exception {
        Member member = dispatchedMember(120L);
        try {
            registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.delivered(), System.currentTimeMillis()));
            assertTrue(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
            registry.cancel(120L, 0L, CancelReason.CLIENT_CANCELLED);
            cancellations.computeIfAbsent(120L, ignored -> new java.util.concurrent.CompletableFuture<>()).complete(ack);
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
            assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone(),
                    "a cancel outcome alone cannot prove Worker cleanup");

            ledger.finish(120L, member.item()).forEach(status ->
                    registry.onEngineEvidence(member.requestContext(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
            DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 120L, true);
            SchedulerTestSupport.awaitRequestWork(registry.runtime);

            assertTerminalsSettledAndArchived(member);
        } finally {
            registry.onDeliveryEvidence(member.claim(), new ContextResult.CleanupAcknowledgement(member.claim(), ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"PDS", "PSD", "DPS", "DSP", "SPD", "SDP"})
    void cancelledBatchRequiresBothTerminalsAndSenderExitInEveryOrder(String order) throws Exception {
        Member member = dispatchedMember(123L);
        try {
            registry.cancel(123L, 0L, CancelReason.CLIENT_CANCELLED);
            cancellations.computeIfAbsent(123L, ignored -> new java.util.concurrent.CompletableFuture<>()).complete(EngineGrpcClient.CancelReply.UNSUPPORTED);
            for (int i = 0; i < order.length(); i++) {
                switch (order.charAt(i)) {
                    case 'P' -> ledger.finish(123L, member.item()).forEach(status ->
                            registry.onEngineEvidence(member.requestContext(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
                    case 'D' -> DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 123L, true);
                    case 'S' -> registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
                    default -> throw new AssertionError(order);
                }
                SchedulerTestSupport.awaitRequestWork(registry.runtime);
                if (i < 2) {
                    assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone(),
                            "partial evidence must retain delivery ownership: " + order.substring(0, i + 1));
                    assertTrue(registry.requests.isCurrent(member.requestContext()));
                }
            }
            assertTerminalsSettledAndArchived(member);
            assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        } finally {
            registry.onDeliveryEvidence(member.claim(), new ContextResult.CleanupAcknowledgement(member.claim(), ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        }
    }

    @org.junit.jupiter.api.RepeatedTest(10)
    void concurrentUnknownAndDecodeActiveConfirmOnceWithoutReleasingExecution() throws Exception {
        Member member = dispatchedMember(124L);
        var published = new java.util.concurrent.atomic.AtomicInteger();
        member.item().future().thenRun(() -> {
            assertFalse(Thread.holdsLock(member.requestContext()));
            published.incrementAndGet();
        });
        var start = new CountDownLatch(1);
        try (var threads = Executors.newFixedThreadPool(2)) {
            var sender = threads.submit(() -> {
                await(start);
                registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            });
            var worker = threads.submit(() -> {
                await(start);
                DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 124L, false);
                DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 124L, false);
            });
            start.countDown();
            sender.get(5, TimeUnit.SECONDS);
            worker.get(5, TimeUnit.SECONDS);
            assertTrue(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> published.get() == 1);
            assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone());
            assertOccupancy(1, 1);
            assertEquals(1, member.item().decodeEp().routingView().engineCapacityUsed());
            assertTrue(cancellations.isEmpty());
        } finally {
            ledger.finish(124L, member.item()).forEach(status ->
                    registry.onEngineEvidence(member.requestContext(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
            DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 124L, true);
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        }
        assertTerminalsSettledAndArchived(member);
        assertEquals(1, published.get());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeFailureBeforeSenderExitSettlesWithoutClientCancellation(boolean uncertainDelivery) throws Exception {
        Member member = dispatchedMember(121L);
        try {
            ledger.finish(121L, member.item()).forEach(status ->
                    registry.onEngineEvidence(member.requestContext(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
            DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 121L, true, 42L);
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
            synchronized (member.requestContext()) {
                assertNull((member.requestContext().snapshot().cancellation() == null ? null : member.requestContext().snapshot().cancellation().reason()), "no client cancellation was recorded");
            }
            assertFalse(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone(),
                    "Worker terminals cannot substitute for sender exit");

            assertFalse(cancellations.containsKey(121L),
                    "both Worker terminals already prove remote cleanup; no Cancel ACK is required");
            registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), uncertainDelivery
                    ? DeliveryResult.uncertain(new IllegalStateException("reply lost")) : DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(registry.runtime);

            assertTerminalsSettledAndArchived(member);
        } finally {
            registry.onDeliveryEvidence(member.claim(), new ContextResult.CleanupAcknowledgement(member.claim(), ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
        }
    }

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void batchAckTimestampBelongsToDeliveryConfirmation(boolean senderFirst, boolean workerSuccessful) throws Exception {
        Member member = dispatchedMember(125L);
        if (senderFirst) {
            registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.delivered(), System.currentTimeMillis()));
            assertTrue(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        }
        ledger.finish(125L, member.item()).forEach(status ->
                registry.onEngineEvidence(member.requestContext(), new ContextResult.EngineEvidence((status.route().prefillEp() == prefill ? status.route() : null), null, RoleType.PREFILL, switch (status.kind()) { case ACTIVE -> ContextResult.EngineKind.ACTIVE; case COMPLETED -> ContextResult.EngineKind.COMPLETED; case FAILED -> ContextResult.EngineKind.FAILED; case PRIORITY_CANCELED -> ContextResult.EngineKind.PRIORITY_CANCELLED; }, status.errorCode(), null, System.currentTimeMillis())));
        DeliverySettlementTestSupport.decodeStatus(member.item().decodeEp(), 125L, true,
                workerSuccessful ? 0L : 42L);
        member.item().future().get(5, TimeUnit.SECONDS);
        long responseObservedAtNanos = System.nanoTime();
        if (!senderFirst) {
            assertEquals(0L, member.requestContext().data().getAckAtNanos());
            registry.onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.delivered(), System.currentTimeMillis()));
        }
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
        assertTerminalsSettledAndArchived(member);
        if (senderFirst) {
            assertTrue(member.requestContext().data().getAckAtMs() > 0L);
            assertTrue(member.requestContext().data().getAckAtNanos() > 0L);
            assertTrue(member.requestContext().data().getAckAtNanos() <= responseObservedAtNanos);
        } else {
            assertEquals(0L, member.requestContext().data().getAckAtMs());
            assertEquals(0L, member.requestContext().data().getAckAtNanos(),
                    "a late sender cannot add an ACK to an already completed response");
        }
    }

    private Member dispatchedMember(long id) {
        DecodeEndpoint decode = EndpointTestSupport.decode(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8180, 8181, null), registry.requests);
        DecodeResources.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = EndpointTestSupport.reserveUnqueuedDecode(decode, pin, id, 1L, 2L, 50);
        }
        Member member = member(id, id, decode, reservation);
        ledger.commit(id, List.of(member.item()));
        DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
        return member;
    }

    private void assertTerminalsSettledAndArchived(Member member) {
        assertOccupancy(0, 0);
        assertEquals(0, member.item().decodeEp().routingView().engineCapacityUsed());
        assertFalse(member.item().decodeEp().ownership(member.item().decodeReservation()).owned());
        assertAll("both Worker terminals and sender exit must finish request cleanup",
                () -> assertTrue(registry.runtime.signals().delivery(member.claim()).toCompletableFuture().isDone(),
                        "delivery settlement still waits after both Worker terminals"),
                () -> assertFalse(registry.requests.isCurrent(member.requestContext()),
                        "request remains active after both resource ledgers are empty"));
    }

    private Member member(long id, long batchId) {
        return member(id, batchId, RequestProtocolTestSupport.decodeEndpoint(), new DecodeResources.ReservationHandle(1L, id, id));
    }

    private Member member(long id, long batchId, DecodeEndpoint decode, DecodeResources.ReservationHandle reservation) {
        return member(id, batchId, decode, reservation, DeliveryClaimKind.BATCH_ENQUEUE);
    }

    private Member member(long id, long batchId, DecodeEndpoint decode, DecodeResources.ReservationHandle reservation,
            DeliveryClaimKind kind) {
        var context = RequestProtocolTestSupport.context(config, id);
        var future = RequestProtocolTestSupport.register(registry, context);
        context.data().setFuture(future);
        var prefillServer = new org.flexlb.dao.loadbalance.ServerStatus();
        prefillServer.setServerIp("127.0.0.1");
        prefillServer.setGrpcPort(8090);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, new Response(), prefillServer, null,
                prefill, decode, reservation, System.currentTimeMillis());
        RequestProtocolTestSupport.beginRoutePublication(registry, new RequestProtocolTestSupport.Registered(item, future));
        var claim = kind == DeliveryClaimKind.BATCH_ENQUEUE ? RequestProtocolTestSupport.claimBatch(registry, item, batchId, () -> true)
                : RequestProtocolTestSupport.claimRouteWithoutPrediction(registry, item, () -> true);
        assertNotNull(claim);
        if (kind == DeliveryClaimKind.BATCH_ENQUEUE) { assertTrue(claim.item().ctx().data().getScheduler().tryStartSend(claim)); }
        return new Member(item, registry.requests.findActive(id), claim);
    }

    private void cleaned(Member member) throws Exception {
        cancellations.computeIfAbsent(member.item().requestId(), ignored -> new java.util.concurrent.CompletableFuture<>())
                .complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
        registry.runtime.signals().delivery(member.claim()).toCompletableFuture().get(2, TimeUnit.SECONDS);
        SchedulerTestSupport.awaitRequestWork(registry.runtime);
    }

    private void reject(Member member) {
        member.claim().item().ctx().data().getScheduler().onDeliveryEvidence(member.claim(), new ContextResult.TransportResult(member.claim(), DeliveryResult.prefillRejected(new IllegalStateException("prepare rejected")), System.currentTimeMillis()));
    }

    private void assertStaleDelivery(Member member, DeliveryResult result) {
        synchronized (member.requestContext()) {
            var before = member.requestContext().snapshot();
            var update = member.requestContext().recordDeliveryResult(
                    new ContextResult.TransportResult(member.claim(), result, System.currentTimeMillis()));
            assertEquals(ContextResult.UpdateKind.STALE, update.kind());
            assertSame(before, update.before());
            assertSame(before, update.after());
            assertSame(before, member.requestContext().snapshot());
        }
    }

    private void decodeFinished(Member member) {
        RequestProtocolTestSupport.applyDecodeStatus(registry, member.requestContext(), member.item().decodeEp(), DecodeResources.DecodeRequestStatus.terminal(member.item().decodeReservation(), 601L));
    }

    private void assertOccupancy(int batches, int members) {
        assertEquals(batches, ledger.prefill.stats().batchCount());
        assertEquals(members, ledger.prefill.stats().locallyOwnedRequests());
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(5, TimeUnit.SECONDS));
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
    }

    private record Member(RequestRoute item, RequestContext requestContext, DeliveryClaim claim) { }
}
