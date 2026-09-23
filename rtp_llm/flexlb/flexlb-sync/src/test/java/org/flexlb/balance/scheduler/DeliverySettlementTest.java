package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DeliverySettlementTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DeliverySettlementTest {
    private FlexlbConfig config;
    private RequestRegistry registry;
    private DeliverySettlementTestSupport ledger;
    private PrefillEndpoint prefill;

    @BeforeEach
    void setUp() {
        config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class));
        ledger = new DeliverySettlementTestSupport();
        prefill = mock(PrefillEndpoint.class);
        when(prefill.releaseCommittedItem(any())).thenAnswer(invocation -> {
            ScheduledRequest item = invocation.getArgument(0);
            RequestSlot slot = registry.requestSlot(item.requestId());
            assertTrue(slot == null || !Thread.holdsLock(slot),
                    "endpoint accounting must run outside the Slot monitor");
            return ledger.prefill.terminalizeCommittedItem(item);
        });
        doAnswer(invocation -> {
            prefill.releaseCommittedItem(invocation.getArgument(0));
            return null;
        }).when(prefill).settleFailedRequest(any());
        when(prefill.expireCommittedItem(any())).thenAnswer(invocation ->
                ledger.prefill.terminalizeCommittedItem(invocation.getArgument(0)));
    }

    @AfterEach
    void close() {
        if (registry.closeAdmissionAndAwaitMutations()) {
            registry.closeOutstandingAndTerminalize();
            registry.closeExpiration();
            registry.closePublisher();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void decodeActivityCannotTurnPrefillRejectionIntoAck(boolean accepted) throws Exception {
        Member member = member(1L, 11L);
        ledger.commit(11L, List.of(member.item()));
        member.slot().processDecodeStatus(member.item().decodeEp(), accepted
                ? DecodeEndpoint.WorkerStatusFact.accepted(member.item().decodeReservation())
                : DecodeEndpoint.WorkerStatusFact.active(member.item().decodeReservation()));

        reject(member);

        assertFalse(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestState.Phase.FAILED, member.slot().snapshot().state());
        assertOccupancy(0, 0);
        verify(member.item().decodeEp(), never()).release(any(), eq(DecodeEndpoint.ReleaseReason.NOT_SENT));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void realDecodeReservationSurvivesRejectionUntilItsOwnFinished(boolean observed) throws Exception {
        var decode = spy(new DecodeEndpoint(WorkerStatus.createDiscovered(
                RoleType.DECODE, null, "127.0.0.1", 8080, 8081, null), new EndpointEventProjector(registry)));
        DecodeEndpoint.ReservationHandle reservation;
        try (var pin = decode.tryPinGeneration()) {
            reservation = decode.reserveUnqueued(pin, 2L, 1L, 1L, 50);
        }
        assertNotNull(reservation);
        Member member = member(2L, 12L, decode, reservation);
        ledger.commit(12L, List.of(member.item()));
        DeliverySettlementTestSupport.dispatchDecode(decode, reservation);
        if (observed) {
            DeliverySettlementTestSupport.decodeStatus(decode, 2L, false);
        }
        assertEquals(1, decode.routingView().engineCapacityUsed());

        reject(member);

        assertFalse(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(0, 0);
        assertEquals(1, decode.routingView().engineCapacityUsed());
        assertEquals(observed, decode.isAcceptedByEngine(reservation));
        verify(decode, never()).release(any(), eq(DecodeEndpoint.ReleaseReason.NOT_SENT));
        DeliverySettlementTestSupport.decodeStatus(decode, 2L, true);
        assertFalse(decode.isAcceptedByEngine(reservation));
        assertEquals(0, decode.routingView().engineCapacityUsed());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateRejectionSettlesOriginalMemberAfterDecodeTerminalOrSlotRemoval(boolean remove) throws Exception {
        Member member = member(3L, 13L);
        ledger.commit(13L, List.of(member.item()));
        decodeFinished(member);
        Response first = member.item().future().get(5, TimeUnit.SECONDS);
        if (remove) {
            assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
        }

        reject(member);

        assertOccupancy(0, 0);
        assertSame(first, member.item().future().join());
    }

    @Test
    void sixteenMemberBatchReleasesOnlyAfterAllFifteenSuccessfulMembersFinish() throws Exception {
        List<Member> members = new ArrayList<>();
        for (long id = 10L; id < 26L; id++) {
            members.add(member(id, 20L));
        }
        ledger.commit(20L, members.stream().map(Member::item).toList());
        reject(members.getFirst());
        assertFalse(members.getFirst().item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(1, 15);
        for (int i = 1; i < members.size(); i++) {
            Member member = members.get(i);
            member.claim().complete(DeliveryResult.delivered());
            assertTrue(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
            new EndpointEventProjector(registry).onPrefillStatus(prefill, RoleType.PREFILL,
                    ledger.finish(20L, member.item()));
            assertOccupancy(i == 15 ? 0 : 1, 15 - i);
        }
        assertTrue(ledger.prefill.batchAvailability(2).isAvailable());
    }

    @Test
    void repeatedPartialRejectionsCannotOccupyBothBatchPermits() {
        for (int i = 0; i < 4; i++) {
            Member rejected = member(30L + 2L * i, 30L + i);
            Member successful = member(31L + 2L * i, 30L + i);
            ledger.commit(30L + i, List.of(rejected.item(), successful.item()));
            reject(rejected);
            ledger.finish(30L + i, successful.item());
            assertOccupancy(0, 0);
            assertTrue(ledger.prefill.batchAvailability(1).isAvailable());
        }
    }

    @Test
    void concurrentFinishedAndRejectionSettleOneMemberOnce() throws Exception {
        Member member = member(40L, 40L);
        ledger.commit(40L, List.of(member.item()));
        CountDownLatch start = new CountDownLatch(1);
        try (var threads = Executors.newFixedThreadPool(2)) {
            var rejection = threads.submit(() -> { await(start); reject(member); });
            var finished = threads.submit(() -> { await(start); ledger.finish(40L, member.item()); });
            start.countDown();
            rejection.get(5, TimeUnit.SECONDS);
            finished.get(5, TimeUnit.SECONDS);
        }
        reject(member);
        ledger.finish(40L, member.item());
        assertOccupancy(0, 0);
    }

    @Test
    void delayedSettlementCannotReleaseReusedRequestId() throws Exception {
        Member old = member(50L, 50L);
        ledger.commit(50L, List.of(old.item()));
        decodeFinished(old);
        old.item().future().get(5, TimeUnit.SECONDS);
        assertTrue(registry.removeExactTerminalRecord(old.slot(), Long.MAX_VALUE));
        ledger.finish(50L, old.item());
        Member replacement = member(50L, 51L);
        ledger.commit(51L, List.of(replacement.item()));

        reject(old);

        assertOccupancy(1, 1);
        assertFalse(replacement.item().future().isDone());
    }

    @Test
    void prefillSettlementFailureReturnsFailureAndRetainsCleanupWithoutBackgroundRetry() throws Exception {
        Member member = member(60L, 60L);
        ledger.commit(60L, List.of(member.item()));
        var failure = new IllegalStateException("injected endpoint failure");
        doThrow(failure).when(prefill).releaseCommittedItem(member.item());

        assertSame(failure, assertThrows(IllegalStateException.class, () -> reject(member)));
        assertFalse(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(1, 1);
        decodeFinished(member);
        assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
        verify(member.item().decodeEp()).settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.PREFILL_REJECTED);

        verify(prefill, after(1200).times(1)).releaseCommittedItem(member.item());
        assertOccupancy(1, 1);
    }

    @Test
    void terminalCleanupFailureDoesNotScheduleBackgroundRetry() throws Exception {
        Member member = member(61L, 61L);
        ledger.commit(61L, List.of(member.item()));
        doThrow(new IllegalStateException("injected expiry cleanup failure"))
                .when(prefill).expireCommittedItem(member.item());
        member.slot().expireInactiveRequest(System.currentTimeMillis()
                + config.getRequestLifecycle().getRequest().getTimeoutMs() + 1L);
        assertFalse(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(1, 1);
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));

        verify(prefill, after(1200).times(1)).expireCommittedItem(member.item());
        verify(member.item().decodeEp()).release(member.item().decodeReservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
        assertOccupancy(1, 1);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void prefillRetirementCompletesFailedCleanupRegardlessOfDecodeFinishOrder(boolean decodeFirst) throws Exception {
        Member member = member(63L, 63L);
        when(prefill.getStatus()).thenReturn(WorkerStatus.createDiscovered(
                RoleType.PREFILL, null, "127.0.0.1", 8080, 8081, null));
        ledger.commit(63L, List.of(member.item()));
        doThrow(new IllegalStateException("Prefill cleanup failed"))
                .when(prefill).releaseCommittedItem(member.item());
        assertThrows(IllegalStateException.class, () -> reject(member));
        Response failure = member.item().future().get(5, TimeUnit.SECONDS);
        assertFalse(failure.isSuccess());

        if (decodeFirst) { decodeFinished(member); }
        // Retirement drains endpoint ownership before projecting its exact items.
        assertTrue(ledger.prefill.terminalizeCommittedItem(member.item()));
        registry.projectPrefillRetirementItem(prefill, member.item());
        if (!decodeFirst) {
            assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
            decodeFinished(member);
        }

        assertSame(failure, member.item().future().join());
        assertOccupancy(0, 0);
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE),
                "retirement must reach a failed Slot that still owns cleanup");
    }

    @Test
    void decodeReservationReleaseFailurePropagatesWithPrefillAlreadySettled() throws Exception {
        Member member = member(62L, 62L);
        ledger.commit(62L, List.of(member.item()));
        var failure = new IllegalStateException("injected Decode reservation release failure");
        when(member.item().decodeEp().settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.NOT_SENT))
                .thenThrow(failure);

        assertSame(failure, assertThrows(IllegalStateException.class, () ->
                member.claim().complete(DeliveryResult.notSent(new IllegalStateException("local dispatch failed")))));
        assertFalse(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
        verify(member.item().decodeEp(), after(1200).times(1))
                .settleFailedRequest(member.item().decodeReservation(), DeliveryResult.Status.NOT_SENT);
        assertOccupancy(0, 0);
    }

    @ParameterizedTest
    @EnumSource(value = DeliveryResult.Status.class, names = {"TIMED_OUT", "UNCERTAIN"})
    void unknownTransportOutcomeRetainsPrefill(DeliveryResult.Status status) {
        Member member = member(70L, 70L);
        ledger.commit(70L, List.of(member.item()));
        member.claim().complete(new DeliveryResult(status, new IllegalStateException("lost response")));
        assertOccupancy(1, 1);
        assertFalse(member.item().future().isDone());
        verify(member.item().decodeEp(), never()).release(any(), eq(DecodeEndpoint.ReleaseReason.NOT_SENT));
    }

    @Test
    void unsentRequestReleasesDecodeReservationOutsideSlotLock() throws Exception {
        Member member = member(80L, 80L);
        ledger.commit(80L, List.of(member.item()));
        when(member.item().decodeEp().settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.NOT_SENT))
                .thenAnswer(invocation -> {
                    assertFalse(Thread.holdsLock(member.slot()));
                    return true;
                });
        member.claim().complete(DeliveryResult.notSent(new IllegalStateException("request build failed")));
        assertFalse(member.item().future().get(5, TimeUnit.SECONDS).isSuccess());
        assertOccupancy(0, 0);
        verify(member.item().decodeEp()).settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.NOT_SENT);
    }

    @ParameterizedTest
    @EnumSource(value = PreemptionCancelPhase.class,
            names = {"CANCEL_IN_FLIGHT", "NOT_FOUND_STALE", "CANCEL_UNKNOWN"})
    void preemptionCannotUsePrefillRejectionAsDecodeCompletion(PreemptionCancelPhase phase) {
        Member member = member(90L, 90L);
        ledger.commit(90L, List.of(member.item()));
        PreemptionRegistration preemption;
        synchronized (member.slot()) {
            preemption = member.slot().tryInstallPreemption(90L, 91L, "priority victim");
            assertNotNull(preemption);
            assertTrue(member.slot().updatePreemption(preemption, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
            if (phase != PreemptionCancelPhase.CANCEL_IN_FLIGHT) {
                assertTrue(member.slot().updatePreemption(preemption, phase));
            }
        }
        reject(member);
        assertOccupancy(0, 0);
        assertFalse(preemption.terminalObservation().toCompletableFuture().isDone());
        verify(member.item().decodeEp(), never()).updatePreemption(
                anyLong(),
                argThat(update -> update.kind() == DecodeEndpoint.PreemptionUpdate.Kind.FINISHED));
        decodeFinished(member);
        assertTrue(preemption.terminalObservation().toCompletableFuture().isDone());
    }

    @Test
    void rejectionRespondsWhilePreemptionStillWaitsForDecode() throws Exception {
        Member member = member(101L, 101L);
        ledger.commit(101L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = member.slot().tryInstallPreemption(101L, 102L, "priority victim");
            assertTrue(member.slot().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
            assertTrue(member.slot().updatePreemption(claim, PreemptionCancelPhase.CANCEL_UNKNOWN));
        }

        reject(member);

        Response failure = member.item().future().get(1, TimeUnit.SECONDS);
        assertFalse(failure.isSuccess());
        assertEquals(RequestState.Phase.FAILED, member.slot().snapshot().state());
        assertFalse(claim.terminalObservation().toCompletableFuture().isDone());
        assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
        member.slot().processDecodeStatus(member.item().decodeEp(),
                DecodeEndpoint.WorkerStatusFact.terminal(member.item().decodeReservation(), 0L));
        assertSame(failure, member.item().future().join());
        assertTrue(claim.terminalObservation().toCompletableFuture().isDone());
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void workerCompletionCannotDiscardFailedCleanup(boolean preempting) throws Exception {
        Member member = member(103L, 103L);
        ledger.commit(103L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = preempting ? member.slot().tryInstallPreemption(103L, 104L, "priority victim") : null;
            if (claim != null) { claim.applyPhase(PreemptionCancelPhase.CANCEL_IN_FLIGHT); }
        }
        CountDownLatch cleaning = new CountDownLatch(1);
        CountDownLatch resume = new CountDownLatch(1);
        var cleanupFailure = new IllegalStateException("Prefill cleanup failed");
        doAnswer(invocation -> {
            cleaning.countDown();
            await(resume);
            throw cleanupFailure;
        }).when(prefill).releaseCommittedItem(member.item());
        try (var executor = Executors.newSingleThreadExecutor()) {
            var callback = executor.submit(() -> reject(member));
            try {
                await(cleaning);
                Response failure = member.item().future().get(1, TimeUnit.SECONDS);
                assertFalse(failure.isSuccess());
                member.slot().processDecodeStatus(member.item().decodeEp(),
                        DecodeEndpoint.WorkerStatusFact.terminal(member.item().decodeReservation(), 0L));
                assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
            } finally {
                resume.countDown();
            }
            assertSame(cleanupFailure,
                    assertThrows(java.util.concurrent.ExecutionException.class,
                            () -> callback.get(5, TimeUnit.SECONDS)).getCause());
        }
        assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
        assertOccupancy(1, 1);
        if (claim != null) {
            assertFalse(claim.terminalObservation().toCompletableFuture().isDone());
            assertFalse(claim.applyPhase(PreemptionCancelPhase.NOT_FOUND_STALE),
                    "Decode completion ends the protocol even while Prefill cleanup is pending");
        }
        member.slot().expireInactiveRequest(System.currentTimeMillis()
                + config.getRequestLifecycle().getRequest().getTimeoutMs() + 1L);
        assertOccupancy(0, 0);
        assertEquals(RequestState.Phase.FAILED, member.slot().snapshot().state());
        if (claim != null) { assertTrue(claim.terminalObservation().toCompletableFuture().isDone()); }
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @Test
    void preemptionReleaseDuringCleanupRechecksTheEarlierPendingResult() throws Exception {
        Member member = member(108L, 108L);
        ledger.commit(108L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = member.slot().tryInstallPreemption(108L, 109L, "priority victim");
        }
        when(member.item().decodeEp().settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.PREFILL_REJECTED))
                .thenAnswer(invocation -> {
                    assertTrue(claim.release());
                    assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
                    return false; // Ownership changed after this pass took its snapshot.
                }).thenReturn(true);

        reject(member);

        assertFalse(member.item().future().get(1, TimeUnit.SECONDS).isSuccess());
        verify(member.item().decodeEp(), times(2)).settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.PREFILL_REJECTED);
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @Test
    void expiryDuringCleanupSurvivesALaterActiveObservation() throws Exception {
        Member member = member(109L, 109L);
        ledger.commit(109L, List.of(member.item()));
        when(member.item().decodeEp().settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.PREFILL_REJECTED))
                .thenAnswer(invocation -> {
                    long expiredAt = System.currentTimeMillis()
                            + config.getRequestLifecycle().getRequest().getTimeoutMs() + 1L;
                    member.slot().expireInactiveRequest(expiredAt);
                    synchronized (member.slot()) {
                        org.springframework.test.util.ReflectionTestUtils.<RequestSlot.EngineObservation>invokeMethod(member.slot(), "applyDecodeStatusLocked", member.item().decodeEp(),
                                DecodeEndpoint.WorkerStatusFact.active(member.item().decodeReservation()), expiredAt + 1L);
                    }
                    assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
                    return false;
                });

        reject(member);

        assertFalse(member.item().future().get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestState.Phase.FAILED, member.slot().snapshot().state());
        verify(member.item().decodeEp()).release(member.item().decodeReservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @Test
    void prefillCompletionCannotFinishPreemptionBeforeDecodeSettlement() throws Exception {
        Member member = member(110L, 110L);
        ledger.commit(110L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = member.slot().tryInstallPreemption(110L, 111L, "priority victim");
        }
        assertTrue(claim.applyPhase(PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        new EndpointEventProjector(registry).onPrefillStatus(prefill, RoleType.PDFUSION,
                ledger.finish(110L, member.item()));
        assertFalse(member.item().future().isDone());
        assertTrue(claim.applyPhase(PreemptionCancelPhase.CANCEL_UNKNOWN));

        reject(member);

        Response response = member.item().future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertFalse(claim.terminalObservation().toCompletableFuture().isDone());
        decodeFinished(member);
        assertSame(response, member.item().future().join());
        assertTrue(claim.terminalObservation().toCompletableFuture().isDone());
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @Test
    void latePriorityCancellationFinishesPrefillCleanupAfterDecodeHasFinished() throws Exception {
        Member member = member(111L, 111L);
        ledger.commit(111L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = member.slot().tryInstallPreemption(111L, 112L, "priority victim");
        }
        assertTrue(claim.applyPhase(PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        doThrow(new IllegalStateException("Prefill cleanup failed"))
                .when(prefill).releaseCommittedItem(member.item());
        assertThrows(IllegalStateException.class, () -> reject(member));
        Response response = member.item().future().get(1, TimeUnit.SECONDS);
        decodeFinished(member);
        assertFalse(claim.terminalObservation().toCompletableFuture().isDone());
        assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));

        ledger.finish(111L, member.item());
        member.slot().processPrefillStatus(prefill, RoleType.PREFILL, PrefillState.WorkerStatusFact.terminal(
                member.item(), PrefillState.WorkerStatusFact.Kind.PRIORITY_CANCELED,
                StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()));

        assertSame(response, member.item().future().join());
        assertFalse(response.isSuccess());
        assertOccupancy(0, 0);
        assertTrue(claim.terminalObservation().toCompletableFuture().isDone());
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @Test
    void latePendingSettlementCannotUndoDecodeCompletion() throws Exception {
        Member member = member(104L, 104L);
        ledger.commit(104L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = member.slot().tryInstallPreemption(104L, 105L, "priority victim");
            assertTrue(member.slot().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        }
        when(member.item().decodeEp().settleFailedRequest(
                member.item().decodeReservation(), DeliveryResult.Status.PREFILL_REJECTED))
                .thenAnswer(invocation -> {
                    member.slot().processDecodeStatus(member.item().decodeEp(),
                            DecodeEndpoint.WorkerStatusFact.terminal(member.item().decodeReservation(), 0L));
                    assertFalse(claim.terminalObservation().toCompletableFuture().isDone(),
                            "cleanup is still executing outside the Slot lock");
                    return false; // Snapshot taken before the above completion.
                });

        reject(member);

        assertFalse(member.item().future().get(1, TimeUnit.SECONDS).isSuccess());
        assertTrue(claim.terminalObservation().toCompletableFuture().isDone());
        assertTrue(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
    }

    @Test
    void inactivityStillCleansResourcesAfterFailureWasPublished() throws Exception {
        config.getRequestLifecycle().getRequest().setTimeoutMs(200L);
        Member member = member(106L, 106L);
        ledger.commit(106L, List.of(member.item()));
        reject(member);
        Response response = member.item().future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        RequestLifecycleTestSupport.awaitCondition(() -> registry.liveRequestCount() == 0);
        assertSame(response, member.item().future().join());
        assertEquals(RequestState.Phase.FAILED, member.slot().snapshot().state());
        verify(member.item().decodeEp()).release(member.item().decodeReservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
    }

    @Test
    void preemptionPhaseAfterFailureCannotReopenResponseOrDropCleanup() throws Exception {
        Member member = member(107L, 107L);
        ledger.commit(107L, List.of(member.item()));
        PreemptionRegistration claim;
        synchronized (member.slot()) {
            claim = member.slot().tryInstallPreemption(107L, 108L, "priority victim");
            assertTrue(member.slot().updatePreemption(claim, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        }
        reject(member);
        Response response = member.item().future().get(1, TimeUnit.SECONDS);
        assertTrue(claim.applyPhase(PreemptionCancelPhase.NOT_FOUND_STALE));
        assertFalse(claim.terminalObservation().toCompletableFuture().isDone());
        assertFalse(registry.removeExactTerminalRecord(member.slot(), Long.MAX_VALUE));
        decodeFinished(member);
        assertSame(response, member.item().future().join());
        assertFalse(response.isSuccess());
        assertTrue(claim.terminalObservation().toCompletableFuture().isDone());
    }

    private Member member(long id, long batchId) {
        return member(id, batchId, mock(DecodeEndpoint.class), new DecodeEndpoint.ReservationHandle(1L, id, id));
    }

    private Member member(long id, long batchId, DecodeEndpoint decode, DecodeEndpoint.ReservationHandle reservation) {
        var context = RequestLifecycleTestSupport.context(config, id);
        var future = registry.register(context);
        var item = new ScheduledRequest(context, future, new Response(), null, null,
                prefill, decode, reservation, System.currentTimeMillis());
        RequestLifecycleTestSupport.bindRoute(registry, new RequestLifecycleTestSupport.Registered(item, future));
        var claim = RequestLifecycleTestSupport.claimBatch(registry, item, batchId, () -> true);
        assertNotNull(claim);
        return new Member(item, registry.requestSlot(id), claim);
    }

    private void reject(Member member) {
        member.claim().complete(DeliveryResult.prefillRejected(new IllegalStateException("prepare rejected")));
    }

    private void decodeFinished(Member member) {
        member.slot().processDecodeStatus(member.item().decodeEp(),
                DecodeEndpoint.WorkerStatusFact.terminal(member.item().decodeReservation(), 601L));
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

    private record Member(ScheduledRequest item, RequestSlot slot, RequestSlot.DeliveryClaim claim) { }
}
