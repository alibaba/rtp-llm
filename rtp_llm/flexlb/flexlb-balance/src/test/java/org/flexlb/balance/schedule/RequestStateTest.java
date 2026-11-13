package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contracts of the immutable request view exposed by RequestRepository.
 */
class RequestStateTest {

    @Test
    void onlyTerminalPhasesAreTerminal() {
        assertFalse(RequestResult.State.QUEUED.isTerminal());
        assertFalse(RequestResult.State.CANCEL_REQUESTED.isTerminal());
        assertTrue(RequestResult.State.CANCELLED.isTerminal());
        assertTrue(RequestResult.State.TIMED_OUT.isTerminal());
        assertTrue(RequestResult.State.FAILED.isTerminal());
        assertTrue(RequestResult.State.COMPLETED.isTerminal());
    }
}
