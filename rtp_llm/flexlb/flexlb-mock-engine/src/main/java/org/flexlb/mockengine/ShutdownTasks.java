package org.flexlb.mockengine;

import java.util.List;

/** Independent close operations: one failure cannot hide the remaining owners. */
final class ShutdownTasks {
    static void runAll(List<Runnable> operations) {
        RuntimeException failure = null;
        for (Runnable operation : operations) {
            try { operation.run(); }
            catch (RuntimeException error) {
                if (failure == null) failure = new IllegalStateException("resource shutdown failed");
                failure.addSuppressed(error);
            }
        }
        if (failure != null) throw failure;
    }
}
