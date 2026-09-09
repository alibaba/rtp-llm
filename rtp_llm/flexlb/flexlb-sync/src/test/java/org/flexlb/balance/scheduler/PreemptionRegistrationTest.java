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
        assertTrue(registration.canSettleTombstone());
        assertTrue(registration.settle());
        assertFalse(registration.settle());
        assertTrue(registration.isSettled());
    }

    @Test
    void notFoundCanTransferToAnEngineFenceOrSettle() {
        PreemptionRegistration registration = registration();

        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        assertTrue(registration.advanceTo(
                PreemptionCancelPhase.NOT_FOUND_STALE));
        assertFalse(registration.advanceTo(
                PreemptionCancelPhase.CANCEL_UNKNOWN));
        assertTrue(registration.isNotFound());
        assertTrue(registration.isFenceTransferable());
        assertTrue(registration.canSettleTombstone());
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
        return new PreemptionRegistration(7L, 11L, "test preemption");
    }
}
