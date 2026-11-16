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
        assertFalse(RequestState.State.QUEUED.isTerminal());
        assertFalse(RequestState.State.CANCEL_REQUESTED.isTerminal());
        assertTrue(RequestState.State.CANCELLED.isTerminal());
        assertTrue(RequestState.State.TIMED_OUT.isTerminal());
        assertTrue(RequestState.State.FAILED.isTerminal());
        assertTrue(RequestState.State.COMPLETED.isTerminal());
    }
}
