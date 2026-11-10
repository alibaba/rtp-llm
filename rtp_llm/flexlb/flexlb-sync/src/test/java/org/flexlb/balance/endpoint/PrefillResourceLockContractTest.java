package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks resource locks remain internal and locked operations reject unguarded calls.
 */
class PrefillResourceLockContractTest {

    @Test
    void endpointLocksAndCallerLockedOperationsAreNotPublicApi() throws Exception {
        int modifiers = WorkerEndpoint.class.getDeclaredMethod("stateLock").getModifiers();
        assertFalse(java.lang.reflect.Modifier.isPublic(modifiers));
        assertFalse(java.lang.reflect.Modifier.isProtected(modifiers));
        assertFalse(
                java.util.Arrays.stream(PrefillEndpoint.class.getMethods())
                        .anyMatch(method -> method.getName().endsWith("Locked")));
    }

    @Test
    void everyLockedEntryRejectsCallsBeforeTouchingArgumentsOrState() {
        PrefillEndpoint state = state();
        List<Runnable> unguardedCalls =
                List.of(
                        () -> state.addQueuedLocked(null),
                        () -> state.commitStatusLocked(null, null));

        for (Runnable call : unguardedCalls) {
            IllegalStateException failure = assertThrows(IllegalStateException.class, call::run);
            assertEquals("Prefill request changes require an Endpoint lock", failure.getMessage());
        }
        assertEquals(
                0L,
                org.flexlb.balance.endpoint.EndpointTestSupport.versions(state).ownershipVersion());
    }

    private static PrefillEndpoint state() {
        return org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                new ReentrantLock(), OrderedPrefillRequestQueue.fifo(16), () -> 0L);
    }
}
