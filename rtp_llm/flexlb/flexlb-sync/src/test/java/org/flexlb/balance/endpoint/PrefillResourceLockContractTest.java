package org.flexlb.balance.endpoint;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Checks resource locks and status updates remain internal implementation details.
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
    void requestAndBatchStatusUpdateIsPrivate() throws Exception {
        var method =
                PrefillEndpoint.class.getDeclaredMethod(
                        "updateRequestAndBatchStatusLocked",
                        org.flexlb.dao.master.WorkerStatus.StatusObservation.class);
        org.junit.jupiter.api.Assertions.assertTrue(
                java.lang.reflect.Modifier.isPrivate(method.getModifiers()));
    }
}
