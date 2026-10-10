package org.flexlb.balance.scheduler;

import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.scheduler.RequestContext.PreemptionRegistration;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreemptionRegistrationTest {

    @Test
    void acceptedCancelFollowsTheSingleLegalPath() {
        PreemptionRegistration registration = registration();

        assertFalse(registration.canAcceptPriorityTerminal());
        assertTrue(advance(registration,
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(registration.canAcceptPriorityTerminal());
        assertFalse(advance(registration,
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(advance(registration,
                PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(advance(registration,
                PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(advance(registration,
                PreemptionCancelPhase.NOT_FOUND_STALE));
        assertTrue(advance(registration,
                PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertTrue(registration.isUnknown());
        assertTrue(registration.canCompletePreemption());
        assertTrue(finish(registration));
        assertFalse(finish(registration));
        assertTrue(registration.isFinished());
        assertFalse(registration.canAcceptPriorityTerminal());
    }

    @Test
    void notFoundRetainsTheAttemptUntilEvidenceOrRequestExpiryFinishesIt() {
        PreemptionRegistration registration = registration();

        assertTrue(advance(registration,
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(advance(registration,
                PreemptionCancelPhase.NOT_FOUND_STALE));
        assertFalse(advance(registration,
                PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertTrue(registration.isNotFound());
        assertFalse(registration.isReleasable());
        assertTrue(registration.canCompletePreemption());
        assertTrue(finish(registration));
        assertFalse(registration.canCompletePreemption());
        assertFalse(advance(registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
    }

    @Test
    void aClaimCanBeReleasedBeforeCancelIsAccepted() {
        PreemptionRegistration claimed = registration();
        PreemptionRegistration inFlight = registration();

        assertTrue(claimed.isReleasable());
        assertTrue(advance(inFlight,
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(inFlight.isReleasable());
        assertTrue(advance(inFlight,
                PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(inFlight.isReleasable());
    }

    @Test
    void requestIdentitySurvivesChangesToTheOriginalInput() {
        PreemptionRegistration registration = registration();
        registration.owner.getRequest().setRequestId(99L);

        assertEquals(7L, registration.requestId());
    }

    private static boolean advance(PreemptionRegistration claim, PreemptionCancelPhase next) {
        synchronized (claim.owner) {
            return Boolean.TRUE.equals(org.springframework.test.util.ReflectionTestUtils.invokeMethod(claim, "advanceTo", next));
        }
    }

    private static boolean finish(PreemptionRegistration claim) {
        synchronized (claim.owner) {
            return Boolean.TRUE.equals(org.springframework.test.util.ReflectionTestUtils.invokeMethod(claim, "tryFinish"));
        }
    }

    private static PreemptionRegistration registration() {
        RequestContext context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 7L);
        context.activate(new RequestContext.RequestFuture((completion, response, failure, interrupt) -> false));
        return new PreemptionRegistration(context, 11L, "test preemption", new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 8090));
    }
}
