package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PreemptionRegistrationTest {

    @Test
    void acceptedCancelFollowsTheSingleLegalPath() {
        PreemptionRegistration registration = registration();

        assertTrue(registration.beginCancel());
        assertFalse(registration.beginCancel());
        assertTrue(registration.acceptCancel());
        assertFalse(registration.acceptCancel());
        assertTrue(registration.markUnknown());
        assertTrue(registration.isUnknown());
        assertTrue(registration.canSettleTombstone());
        assertTrue(registration.settle());
        assertFalse(registration.settle());
        assertTrue(registration.isSettled());
    }

    @Test
    void notFoundCanTransferToAnEngineFenceOrSettle() {
        PreemptionRegistration registration = registration();

        assertTrue(registration.beginCancel());
        assertTrue(registration.markNotFound());
        assertTrue(registration.isNotFound());
        assertTrue(registration.isFenceTransferable());
        assertTrue(registration.canSettleTombstone());
    }

    @Test
    void aClaimCanBeReleasedBeforeCancelIsAccepted() {
        PreemptionRegistration claimed = registration();
        PreemptionRegistration inFlight = registration();

        assertTrue(claimed.isReleasable());
        assertTrue(inFlight.beginCancel());
        assertTrue(inFlight.isReleasable());
        assertTrue(inFlight.acceptCancel());
        assertFalse(inFlight.isReleasable());
    }

    private static PreemptionRegistration registration() {
        return new PreemptionRegistration(7L, 11L, "test preemption");
    }
}
