package org.flexlb.balance.scheduler;

import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreemptionRegistrationTest {

    @Test
    void acceptedCancelFollowsTheSingleLegalPath() {
        PreemptionRegistration registration = registration();

        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertFalse(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(registration.advanceTo(
                PreemptionCancelPhase.NOT_FOUND_STALE));
        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertTrue(registration.isUnknown());
        assertTrue(registration.canCompletePreemption());
        assertTrue(registration.tryFinish());
        assertFalse(registration.tryFinish());
        assertTrue(registration.isFinished());
    }

    @Test
    void notFoundRetainsTheAttemptUntilEvidenceOrRequestExpiryFinishesIt() {
        PreemptionRegistration registration = registration();

        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.NOT_FOUND_STALE));
        assertFalse(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertTrue(registration.isNotFound());
        assertFalse(registration.isReleasable());
        assertTrue(registration.canCompletePreemption());
        assertTrue(registration.tryFinish());
        assertFalse(registration.canCompletePreemption());
        assertFalse(registration.advanceTo(PreemptionCancelPhase.CANCEL_IN_FLIGHT));
    }

    @Test
    void aClaimCanBeReleasedBeforeCancelIsAccepted() {
        PreemptionRegistration claimed = registration();
        PreemptionRegistration inFlight = registration();

        assertTrue(claimed.isReleasable());
        assertTrue(inFlight.advanceTo(
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(inFlight.isReleasable());
        assertTrue(inFlight.advanceTo(
                PreemptionCancelPhase.CANCEL_REQUESTED));
        assertFalse(inFlight.isReleasable());
    }

    private static PreemptionRegistration registration() {
        return new PreemptionRegistration(null, "7", 11L, "test preemption");
    }
}
