package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PrefillResourceLockContractTest {

    @Test
    void everyLockedEntryRejectsCallsBeforeTouchingArgumentsOrState() {
        PrefillEndpoint state = state();
        List<Runnable> unguardedCalls =
                List.of(
                        () -> state.enableQueueLocked(null),
                        () -> state.addQueuedLocked(null),
                        () -> state.removeQueuedLocked(null),
                        state::schedulingInputsChangedLocked,
                        () -> state.ownsSelectionLocked(List.of(), 0L),
                        state::queueHeadLocked,
                        () -> state.commitLocked(null),
                        () -> state.commitStatusLocked(null, null));

        for (Runnable call : unguardedCalls) {
            IllegalStateException failure = assertThrows(IllegalStateException.class, call::run);
            assertEquals("Prefill ownership requires queue lock", failure.getMessage());
        }
        assertEquals(0L, state.progress().ownershipVersion());
    }

    private static PrefillEndpoint state() {
        return org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                new ReentrantLock(),
                PrefillActiveIndex.ordered(
                        16,
                        Comparator.comparingLong(
                                org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem
                                        ::requestId)),
                () -> 0L);
    }
}
