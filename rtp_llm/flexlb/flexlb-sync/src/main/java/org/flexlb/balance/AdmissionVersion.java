package org.flexlb.balance;

import java.util.function.Predicate;
import java.util.function.Supplier;

/** Per-worker revision used by both Prefill and Decode when registering a placement.
 * The endpoint serializes validation, registration and revision changes under its own lock.
 */
public final class AdmissionVersion {
    public static final long ANY = -1L;
    private volatile long value;

    public long current() { return value; }

    public boolean accepts(long expected) { return expected == ANY || expected == value; }

    /** Retry the whole decision on a conflict. A capacity block is returned to its normal waiter. */
    public static <T> T retryConflicts(Supplier<T> decision, Predicate<T> conflicted) {
        T result = decision.get();
        for (int attempt = 1; attempt < 8 && conflicted.test(result); attempt++) {
            result = decision.get();
        }
        return result;
    }

    /** Caller owns the same lock that protects the workload mutation. */
    public void advance() { value++; }
}
