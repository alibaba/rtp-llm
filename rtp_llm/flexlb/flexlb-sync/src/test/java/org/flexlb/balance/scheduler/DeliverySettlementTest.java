package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.RequestContext.PreemptionRegistration;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
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
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.argThat;
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
    private final java.util.Map<Long, java.util.concurrent.CompletableFuture<EngineCancelChannel.CancelAck>> cancellations = new java.util.concurrent.ConcurrentHashMap<>();

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        EngineCancelChannel channel = mock(EngineCancelChannel.class);
        org.springframework.test.util.ReflectionTestUtils.setField(registry.runtime, "cancelChannel", channel);
        when(channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call ->
                cancellations.computeIfAbsent(call.getArgument(1), ignored -> new java.util.concurrent.CompletableFuture<>()));
        ledger = new DeliverySettlementTestSupport();
        prefill = mock(PrefillEndpoint.class);
        when(prefill.releaseRequest(any())).thenAnswer(invocation -> {
            RequestRoute item = invocation.getArgument(0);
            RequestContext requestContext = registry.findRequestContext(item.requestId());
            assertTrue(requestContext == null || !Thread.holdsLock(requestContext),
                    "endpoint accounting must run outside the context monitor");
            return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(ledger.prefill, item);
        });
    }

    @AfterEach
    void close() {
        if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
            registry.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
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
        cancellations.get(2L).complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
        assertEquals(1, decode.routingView().engineCapacityUsed());
        assertOccupancy(1, 1);
        assertFalse(member.claim().settlement().toCompletableFuture().isDone());
        DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        member.claim().settlement().toCompletableFuture().get(2, TimeUnit.SECONDS);
        registry.runtime.continuations().awaitIdle();
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
            member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.delivered());
            ledger.finish(20L, member.item()).forEach(status -> registry.onPrefillStatus(status.route().ctx(), prefill, RoleType.PREFILL, status));
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
            successful.claim().item.ctx().scheduler().completeDelivery(successful.claim(), DeliveryResult.delivered());
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
            var cleanup = threads.submit(() -> { await(start); cancellations.get(40L).complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED); });
            var finished = threads.submit(() -> { await(start); ledger.finish(40L, member.item()); });
            start.countDown();
            cleanup.get(2, TimeUnit.SECONDS);
            finished.get(2, TimeUnit.SECONDS);
        }
        member.claim().settlement().toCompletableFuture().get(2, TimeUnit.SECONDS);
        registry.runtime.continuations().awaitIdle();
        assertThrows(IllegalStateException.class, () -> reject(member));
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
        ledger.finish(50L, old.item()).forEach(status -> registry.onPrefillStatus(status.route().ctx(), prefill, RoleType.PREFILL, status));
        registry.runtime.continuations().awaitIdle();
        assertOccupancy(1, 1);
        assertSame(replacement.requestContext(), registry.requests.findActive(50L));
        assertFalse(replacement.item().future().isDone());
    }

    @Test
    void failedLocalCleanupRetainsOwnershipAndRecordsFailureWithoutBusyRetry() throws Exception {
        Member member = member(60L, 60L);
        ledger.commit(60L, List.of(member.item()));
        doThrow(new IllegalStateException("injected endpoint failure")).when(prefill).releaseRequest(member.item());
        reject(member);
        cleaned(member);
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(1, 1);
        assertTrue(registry.requests.isCurrent(member.requestContext()));
        org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(registry));
        verify(prefill, after(150).times(1)).releaseRequest(member.item());
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
        var item = SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                prefill, decode, reservation, System.currentTimeMillis());
        DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
        if (accepted) { DeliverySettlementTestSupport.decodeStatus(decode, 91L, false); }
        try {
            assertEquals(1, decode.routingView().engineCapacityUsed());
            var settlement = AbstractRequestScheduler.releaseResources(item, null, source, true, false);
            boolean released = source == DeliveryResult.Status.NOT_SENT && !accepted;
            assertEquals(released, settlement.decodeSettled());
            assertTrue(settlement.prefillSettled());
            org.junit.jupiter.api.Assertions.assertNull(settlement.failure());
            assertEquals(released ? 0 : 1, decode.routingView().engineCapacityUsed());
            assertEquals(!released, decode.hasOwnedResources(reservation));
            verify(decode, source == DeliveryResult.Status.NOT_SENT ? times(1) : never())
                    .release(reservation, DecodeResources.ReleaseReason.NOT_SENT);
        } finally {
            decode.release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            item.close();
        }
    }

    @Test
    void notSentReleasesLocalReservationsOutsideTheRequestLock() throws Exception {
        Member member = member(80L, 80L);
        ledger.commit(80L, List.of(member.item()));
        when(member.item().decodeEp().hasOwnedResources(member.item().decodeReservation()))
                .thenAnswer(invocation -> { assertFalse(Thread.holdsLock(member.requestContext())); return false; });
        member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.notSent(new IllegalStateException("serialization failed")));
        member.claim().settlement().toCompletableFuture().get(2, TimeUnit.SECONDS);
        registry.runtime.continuations().awaitIdle();
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(0, 0);
        assertTrue(cancellations.isEmpty());
        verify(member.item().decodeEp()).release(member.item().decodeReservation(), DecodeResources.ReleaseReason.NOT_SENT);
    }

    @Test
    void unknownTransportOutcomeRetainsResourcesAndDoesNotFabricateAcknowledgement() {
        Member member = member(70L, 70L);
        ledger.commit(70L, List.of(member.item()));
        member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.uncertain(new IllegalStateException("reply lost")));
        assertOccupancy(1, 1);
        assertFalse(member.item().future().isDone());
        assertTrue(cancellations.isEmpty());
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class, names = {"CANCEL_IN_FLIGHT", "NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void rejectionCannotFinishPreemptionBeforeCleanupProof(PreemptionCancelPhase phase) throws Exception {
        Member member = member(90L, 90L);
        ledger.commit(90L, List.of(member.item()));
        PreemptionRegistration preemption = member.requestContext().tryInstallPreemption(member.item().decodeReservation(), 91L, "priority victim");
        assertTrue(registry.updatePreemption(preemption, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        if (phase != PreemptionCancelPhase.CANCEL_IN_FLIGHT) { assertTrue(registry.updatePreemption(preemption, phase)); }
        reject(member);
        assertFalse(member.item().future().get(2, TimeUnit.SECONDS).isSuccess());
        assertFalse(preemption.requestResolution().toCompletableFuture().isDone());
        assertOccupancy(1, 1);
        cleaned(member);
        assertTrue(preemption.requestResolution().toCompletableFuture().isDone());
        assertOccupancy(0, 0);
    }

    @Test
    void callbackReturnsWhileRemoteProofAndLocalCleanupRunElsewhere() throws Exception {
        Member member = member(99112L, 99112L);
        ledger.commit(99112L, List.of(member.item()));
        CountDownLatch cleanupEntered = new CountDownLatch(1);
        CountDownLatch releaseCleanup = new CountDownLatch(1);
        doAnswer(call -> { cleanupEntered.countDown(); await(releaseCleanup); return org.flexlb.balance.endpoint.EndpointTestSupport.releaseRequest(ledger.prefill, call.getArgument(0)); })
                .when(prefill).releaseRequest(member.item());
        var callback = Executors.newSingleThreadExecutor();
        try {
            callback.submit(() -> member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.prefillRejected(new IllegalStateException("prepare rejected")))).get(1, TimeUnit.SECONDS);
            assertOccupancy(1, 1);
            RequestProtocolTestSupport.awaitCondition(() -> cancellations.containsKey(member.item().requestId()));
            cancellations.get(member.item().requestId()).complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            assertTrue(cleanupEntered.await(2, TimeUnit.SECONDS));
            assertTrue(registry.requests.isCurrent(member.requestContext()));
        } finally {
            releaseCleanup.countDown();
            callback.shutdownNow();
        }
        registry.runtime.continuations().awaitIdle();
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
                case 'A' -> member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.delivered());
                case 'P' -> ledger.finish(70L, member.item()).forEach(status -> registry.onPrefillStatus(status.route().ctx(), prefill, RoleType.PREFILL, status));
                case 'D' -> {
                    DeliverySettlementTestSupport.decodeStatus(decode, 70L, true);
                    decodeEnded = true;
                }
                default -> throw new AssertionError(order);
            }
            registry.runtime.continuations().awaitIdle();
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
        ledger.finish(70L, member.item()).forEach(status -> registry.onPrefillStatus(status.route().ctx(), prefill, RoleType.PREFILL, status));
        registry.runtime.continuations().awaitIdle();
        assertOccupancy(0, 0);
        assertSame(response, member.item().future().join());
        assertThrows(IllegalStateException.class, () -> member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.delivered()));
    }

    private Member member(long id, long batchId) {
        return member(id, batchId, RequestProtocolTestSupport.decodeEndpoint(), new DecodeResources.ReservationHandle(1L, id, id));
    }

    private Member member(long id, long batchId, DecodeEndpoint decode, DecodeResources.ReservationHandle reservation) {
        var context = RequestProtocolTestSupport.context(config, id);
        var future = RequestProtocolTestSupport.register(registry, context);
        context.setFuture(future);
        var prefillServer = new org.flexlb.dao.loadbalance.ServerStatus();
        prefillServer.setServerIp("127.0.0.1");
        prefillServer.setGrpcPort(8090);
        var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), prefillServer, null,
                prefill, decode, reservation, System.currentTimeMillis());
        RequestProtocolTestSupport.bindRoute(registry, new RequestProtocolTestSupport.Registered(item, future));
        var claim = RequestProtocolTestSupport.claimBatch(registry, item, batchId, () -> true);
        assertNotNull(claim);
        assertTrue(claim.item.ctx().scheduler().tryStartSend(claim));
        return new Member(item, registry.findRequestContext(id), claim);
    }

    private void cleaned(Member member) throws Exception {
        cancellations.computeIfAbsent(member.item().requestId(), ignored -> new java.util.concurrent.CompletableFuture<>())
                .complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
        member.claim().settlement().toCompletableFuture().get(2, TimeUnit.SECONDS);
        registry.runtime.continuations().awaitIdle();
    }

    private void reject(Member member) {
        member.claim().item.ctx().scheduler().completeDelivery(member.claim(), DeliveryResult.prefillRejected(new IllegalStateException("prepare rejected")));
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

    private record Member(RequestRoute item, RequestContext requestContext, RequestContext.DeliveryClaim claim) { }
}
