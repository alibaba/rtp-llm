package org.flexlb.dao;

import org.flexlb.util.PriorityNormalizer;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Immutable scheduling metadata shared by every scheduling mode.
 *
 * <p>The request expiration is an absolute Unix timestamp resolved once from
 * the existing request timestamp and timeout fields. It is never derived from
 * prompt length, priority, or a scheduler retry, so moving between queues or
 * retrying a route cannot extend it.
 */
public final class SchedulingMetadata {

    private final int priority;
    private final long expiresAtMs;

    private SchedulingMetadata(int priority, long expiresAtMs) {
        checkArgument(expiresAtMs > 0, "schedule deadline must be a positive Unix timestamp");
        this.priority = priority;
        this.expiresAtMs = expiresAtMs;
    }

    /** Normalize the caller priority once and bind it to the absolute expiry. */
    public static SchedulingMetadata of(int rawProtoPriority,
                                        String qosHeader,
                                        long expiresAtMs,
                                        int defaultPriority) {
        int priority = PriorityNormalizer.normalize(
                rawProtoPriority, qosHeader, defaultPriority);
        return new SchedulingMetadata(priority, expiresAtMs);
    }

    /** Bind an already-normalized priority to its absolute expiry. */
    public static SchedulingMetadata explicit(int priority, long expiresAtMs) {
        return new SchedulingMetadata(priority, expiresAtMs);
    }

    /** Normalized priority (1-100, higher is more important). */
    public int priority() {
        return priority;
    }

    /** Absolute request expiration time in Unix epoch milliseconds. */
    public long expiresAtMs() {
        return expiresAtMs;
    }

}
