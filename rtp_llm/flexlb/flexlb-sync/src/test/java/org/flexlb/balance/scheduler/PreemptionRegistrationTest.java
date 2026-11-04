package org.flexlb.balance.scheduler;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.ResponseCompletionExecutor.RequestFuture;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.EngineEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EngineKind;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class PreemptionRegistrationTest {
    @Test
    void acceptedCancelFollowsTheSingleLegalPath() {
        var registration = registration();
        var context = registration.owner();
        assertFalse(context.snapshot().preemption().phase().acceptsPriorityTerminal());
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(context.snapshot().preemption().phase().acceptsPriorityTerminal());
        assertFalse(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(advance(registration, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(advance(registration, PreemptionCancelPhase.NOT_FOUND_STALE));
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertEquals(PreemptionCancelPhase.CANCEL_UNKNOWN, context.snapshot().preemption().phase());
        assertTrue(
                context.snapshot().preemption().cancelAcknowledged(),
                "later UNKNOWN preserves the accepted cancellation");
        assertTrue(finish(registration));
        assertFalse(finish(registration));
        assertTrue(context.snapshot().preemption().finished());
        assertFalse(advance(registration, PreemptionCancelPhase.CANCEL_REQUESTED));
    }

    @Test
    void notFoundRetainsTheAttemptUntilEvidenceOrRequestExpiryFinishesIt() {
        var registration = registration();
        var context = registration.owner();
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(advance(registration, PreemptionCancelPhase.NOT_FOUND_STALE));
        assertFalse(advance(registration, PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertEquals(
                PreemptionCancelPhase.NOT_FOUND_STALE, context.snapshot().preemption().phase());
        assertFalse(context.snapshot().preemption().phase().isLocallyReleasable());
        org.junit.jupiter.api.Assertions.assertFalse(context.canReconcilePreemption(registration));
        assertTrue(finish(registration));
        assertTrue(context.snapshot().preemption().finished());
        assertFalse(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
    }

    @Test
    void aClaimCanBeReleasedBeforeCancelIsAccepted() {
        var claimed = registration();
        var inFlight = registration();
        assertTrue(claimed.owner().snapshot().preemption().phase().isLocallyReleasable());
        assertTrue(advance(inFlight, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(inFlight.owner().snapshot().preemption().phase().isLocallyReleasable());
        assertTrue(advance(inFlight, PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(inFlight.owner().snapshot().preemption().phase().isLocallyReleasable());
    }

    @Test
    void registeredInputCannotBeReplacedWhilePreemptionOwnsIdentity() {
        var registration = registration();
        var context = registration.owner();
        context.data().setRequest(Request.builder().requestId(7L).seqLen(16L).build());
        context.data()
                .setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50, 100_000L));
        context.data()
                .register(
                        mock(RequestLifecycle.class),
                        new RequestFuture((kind, response, failure) -> false));
        assertThrows(
                IllegalStateException.class,
                () ->
                        context.data()
                                .setRequest(
                                        context.data().getRequest().toBuilder()
                                                .requestId(99L)
                                                .build()));
        assertEquals(7L, context.snapshot().key().requestId());
    }

    @Test
    void unknownCancelRequiresWorkerProofAndCannotDowngradeItToLocalExpiry() {
        var registration = registration();
        var context = registration.owner();
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_UNKNOWN));
        var route = context.snapshot().route();
        var failure =
                new EndEvidence(
                        route,
                        EndKind.FAILURE,
                        null,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "local failure",
                        false,
                        false);
        assertNull(context.beginClose(failure, 1002L).outcome());
        assertSame(failure, context.snapshot().preemption().pendingTerminal());
        org.junit.jupiter.api.Assertions.assertFalse(
                context.canReconcilePreemption(registration),
                "UNKNOWN cannot turn local failure into Decode proof");
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        route,
                                        route.decodeReservation(),
                                        RoleType.DECODE,
                                        EngineKind.COMPLETED,
                                        0L,
                                        "worker complete",
                                        1003L)));
        var worker = context.snapshot().preemption().pendingTerminal();
        assertEquals(EndKind.WORKER, worker.kind());
        var token = context.snapshot().deadlines().inactivity();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest -> observedRequest.recordDeadline(token, token.atMs(), null));
        assertSame(
                worker,
                context.snapshot().preemption().pendingTerminal(),
                "authoritative Worker completion takes precedence over later local expiry");
        var beforeReconciliation = context.snapshot();
        org.junit.jupiter.api.Assertions.assertTrue(context.canReconcilePreemption(registration));
        assertSame(
                beforeReconciliation,
                context.snapshot(),
                "synchronous ledger preparation must not publish an in-progress request state");
        assertSame(registration, context.snapshot().preemption().identity());
        assertSame(worker, context.snapshot().preemption().pendingTerminal());
    }

    @Test
    void prefillHandoffTerminalRequiresIndependentDecodeProofBeforeNotFoundCanFinish() {
        var registration = registration();
        var context = registration.owner();
        var route = context.snapshot().route();
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertNull(
                context.beginClose(
                                new EndEvidence(
                                        route,
                                        EndKind.WORKER,
                                        RoleType.PREFILL,
                                        true,
                                        0L,
                                        null,
                                        "Prefill completed",
                                        true,
                                        false),
                                1002L)
                        .outcome());
        assertTrue(advance(registration, PreemptionCancelPhase.NOT_FOUND_STALE));
        org.junit.jupiter.api.Assertions.assertFalse(
                context.canReconcilePreemption(registration),
                "Prefill handoff plus NOT_FOUND is not Decode release proof");
        assertFalse(context.snapshot().preemption().finished());
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordEngineEvidence(
                                new EngineEvidence(
                                        route,
                                        route.decodeReservation(),
                                        RoleType.DECODE,
                                        EngineKind.ALLOCATED,
                                        0L,
                                        null,
                                        1003L)));
        org.junit.jupiter.api.Assertions.assertTrue(context.canReconcilePreemption(registration));
        context.completePreemption(
                registration,
                new PreemptionReport(
                        null, true, true, false, context.snapshot().preemption().pendingTerminal()),
                1004L);
        assertTrue(context.snapshot().preemption().finished());
        assertTrue(context.snapshot().close().prefillSettled());
        assertTrue(context.snapshot().close().decodeSettled());
    }

    private static boolean advance(PreemptionRegistration exact, PreemptionCancelPhase phase) {
        return exact.owner()
                .completePreemption(
                        exact, new PreemptionReport(phase, false, false, false, null), 1001L);
    }

    private static boolean finish(PreemptionRegistration exact) {
        var context = exact.owner();
        if (context.snapshot().preemption().finished()) {
            return false;
        }
        var evidence =
                new EndEvidence(
                        context.snapshot().route(),
                        EndKind.PRIORITY,
                        RoleType.PREFILL,
                        false,
                        0L,
                        StrategyErrorType.PRIORITY_PREEMPTED,
                        "matching fenced proof",
                        true,
                        true);
        context.completePreemption(
                exact, new PreemptionReport(null, true, true, false, evidence), 1002L);
        return context.snapshot().preemption().finished();
    }

    private static PreemptionRegistration registration() {
        var context = RequestCleanupPassTest.openContext(7L);
        return RequestCleanupPassTest.requireValue(
                context.beginPreemption(
                        context.snapshot().route().decodeReservation(),
                        11L,
                        new CancelTarget("127.0.0.1", 8090)));
    }
}
