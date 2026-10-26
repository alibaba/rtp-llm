package org.flexlb.util;

import java.util.function.Consumer;
import java.util.function.Supplier;

/** Preserve the first failure while callers attempt every independent cleanup. */
public final class Failures {
    private Failures() {}

    public static Throwable append(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (next != null && first != next) {
            try {
                first.addSuppressed(next);
            } catch (Throwable ignored) {
                // Diagnostic failure must not prevent the remaining cleanup operations.
            }
        }
        return first;
    }

    public static RuntimeException propagate(Throwable failure, String message) {
        if (failure instanceof RuntimeException runtime) {
            return runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(message, failure);
    }

    public static void rethrow(Throwable failure, String message) {
        if (failure != null) {
            throw propagate(failure, message);
        }
    }

    public static Throwable run(Throwable first, Runnable action) {
        if (action == null) {
            return first;
        }
        try {
            action.run();
            return first;
        } catch (Throwable failure) {
            return append(first, failure);
        }
    }

    public static Throwable close(AutoCloseable resource) {
        try {
            resource.close();
            return null;
        } catch (Throwable failure) {
            return failure;
        }
    }

    /** Retry one idempotent action, preserving failures even when a later attempt succeeds. */
    public static Throwable recover(Consumer<Throwable> report, Runnable action) {
        Throwable failure = null;
        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                action.run();
                return null;
            } catch (RuntimeException | Error failed) {
                report.accept(failed);
                failure = append(failure, failed);
            }
        }
        return failure;
    }

    /** The caller retains its operation identity; null denotes exhausted action recovery. */
    public static <T> T recover(Consumer<Throwable> report, Supplier<T> action) {
        for (int attempt = 0; attempt < 3; attempt++) {
            try {
                return action.get();
            } catch (RuntimeException | Error failed) {
                report.accept(failed);
            }
        }
        return null;
    }
}
