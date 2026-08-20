package org.flexlb.dao;

import org.flexlb.util.PriorityNormalizer;

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
    private final PrioritySource source;
    private final long expiresAtMs;

    private SchedulingMetadata(int priority, PrioritySource source, long expiresAtMs) {
        if (expiresAtMs <= 0) {
            throw new IllegalArgumentException("schedule deadline must be a positive Unix timestamp");
        }
        this.priority = priority;
        this.source = source;
        this.expiresAtMs = expiresAtMs;
    }

    /** Normalize the caller priority once and bind it to the absolute expiry. */
    public static SchedulingMetadata of(int rawProtoPriority,
                                        String qosHeader,
                                        long expiresAtMs,
                                        int defaultPriority) {
        int priority = PriorityNormalizer.normalize(
                rawProtoPriority, qosHeader, defaultPriority);
        boolean explicit = rawProtoPriority != 0
                || (qosHeader != null && !qosHeader.isBlank());
        PrioritySource source = explicit
                ? PrioritySource.EXPLICIT
                : PrioritySource.DEFAULT;
        return new SchedulingMetadata(priority, source, expiresAtMs);
    }

    /** Test/helper factory for an already-normalized explicit priority. */
    public static SchedulingMetadata explicit(int priority, long expiresAtMs) {
        return new SchedulingMetadata(priority, PrioritySource.EXPLICIT, expiresAtMs);
    }

    /** Normalized priority (1-100, higher is more important). */
    public int priority() {
        return priority;
    }

    /** Whether priority was caller-supplied or defaulted. */
    public PrioritySource source() {
        return source;
    }

    /** Absolute request expiration time in Unix epoch milliseconds. */
    public long expiresAtMs() {
        return expiresAtMs;
    }

    /** Remaining request lifetime; may be zero or negative after expiration. */
    public long remainingMs(long nowMs) {
        return expiresAtMs - nowMs;
    }

    public boolean expired(long nowMs) {
        return nowMs >= expiresAtMs;
    }

    /** Priority origin is observability metadata, not an ordering dimension. */
    public enum PrioritySource {
        EXPLICIT,
        DEFAULT
    }
}
