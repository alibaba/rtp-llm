package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Query records expose request state without becoming executable worker routes.
 */
class RequestStateBoundaryTest {
    @Test
    void queryRecordIsNotARequestRoute() {
        Object record =
                new RequestState(
                        "1",
                        1,
                        RequestPhase.GENERATION,
                        RequestState.State.COMPLETED,
                        10,
                        20,
                        "completed");
        assertFalse(record instanceof RequestRoute);
    }
}
