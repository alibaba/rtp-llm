package org.flexlb.balance.scheduler;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

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
        assertTrue(context.snapshot().preemption().cancelAcknowledged(), "later UNKNOWN preserves the accepted cancellation");
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
        assertEquals(PreemptionCancelPhase.NOT_FOUND_STALE, context.snapshot().preemption().phase());
        assertFalse(context.snapshot().preemption().phase().isLocallyReleasable());
        assertInstanceOf(Blocked.class, context.beginReconciliation(registration));
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
        context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(50, 100_000L));
        context.data().register(mock(AbstractRequestScheduler.class), new RequestFuture((kind, response, failure) -> false));
        assertThrows(IllegalStateException.class,
                () -> context.data().setRequest(context.data().getRequest().toBuilder().requestId(99L).build()));
        assertEquals(7L, context.snapshot().key().requestId());
    }

    @Test
    void unknownCancelRequiresWorkerProofAndCannotDowngradeItToLocalExpiry() {
        var registration = registration();
        var context = registration.owner();
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_UNKNOWN));
        var route = context.snapshot().route();
        var failure = new EndEvidence(route, EndKind.FAILURE, null, false, 0L,
                StrategyErrorType.DISPATCH_FAILED, "local failure", false, false);
        assertInstanceOf(Blocked.class, context.beginClose(failure, 1002L));
        assertSame(failure, context.snapshot().preemption().pendingTerminal());
        assertInstanceOf(Blocked.class, context.beginReconciliation(registration),
                "UNKNOWN cannot turn local failure into Decode proof");
        context.recordEngineEvidence(new EngineEvidence(route, route.decodeReservation(), RoleType.DECODE,
                EngineKind.COMPLETED, 0L, "worker complete", 1003L));
        var worker = context.snapshot().preemption().pendingTerminal();
        assertEquals(EndKind.WORKER, worker.kind());
        var token = context.snapshot().deadlines().inactivity();
        context.recordDeadline(token, token.atMs(), null);
        assertSame(worker, context.snapshot().preemption().pendingTerminal(),
                "authoritative Worker completion takes precedence over later local expiry");
        var reconcile = RequestCleanupPassTest.granted(context.beginReconciliation(registration));
        assertSame(registration, reconcile.attempt());
        assertSame(worker, reconcile.terminal());
    }

    @Test
    void prefillHandoffTerminalRequiresIndependentDecodeProofBeforeNotFoundCanFinish() {
        var registration = registration();
        var context = registration.owner();
        var route = context.snapshot().route();
        assertTrue(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertInstanceOf(Blocked.class, context.beginClose(new EndEvidence(route, EndKind.WORKER, RoleType.PREFILL,
                true, 0L, null, "Prefill completed", true, false), 1002L));
        assertTrue(advance(registration, PreemptionCancelPhase.NOT_FOUND_STALE));
        assertInstanceOf(Blocked.class, context.beginReconciliation(registration),
                "Prefill handoff plus NOT_FOUND is not Decode release proof");
        assertFalse(context.snapshot().preemption().finished());
        context.recordEngineEvidence(new EngineEvidence(route, route.decodeReservation(), RoleType.DECODE,
                EngineKind.ALLOCATED, 0L, null, 1003L));
        var permission = RequestCleanupPassTest.granted(context.beginReconciliation(registration));
        context.completePreemption(permission, new PreemptionReport(null, true, true, false, permission.terminal()), 1004L);
        assertTrue(context.snapshot().preemption().finished());
        assertTrue(context.snapshot().close().prefillSettled());
        assertTrue(context.snapshot().close().decodeSettled());
    }

    private static boolean advance(PreemptionRegistration exact, PreemptionCancelPhase phase) {
        return exact.owner().completePreemption(exact, new PreemptionReport(phase, false, false, false, null), 1001L).kind()
                == UpdateKind.APPLIED;
    }

    private static boolean finish(PreemptionRegistration exact) {
        var context = exact.owner();
        if (context.snapshot().preemption().finished()) { return false; }
        var evidence = new EndEvidence(context.snapshot().route(), EndKind.PRIORITY, RoleType.PREFILL, false, 0L,
                StrategyErrorType.PRIORITY_PREEMPTED, "matching fenced proof", true, true);
        context.completePreemption(exact, new PreemptionReport(null, true, true, false, evidence), 1002L);
        return context.snapshot().preemption().finished();
    }

    private static PreemptionRegistration registration() {
        var context = RequestCleanupPassTest.openContext(7L);
        return RequestCleanupPassTest.granted(context.beginPreemption(context.snapshot().route().decodeReservation(),
                11L, new CancelTarget("127.0.0.1", 8090)));
    }
}
