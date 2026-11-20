package org.flexlb.config.scheduler;

import lombok.Getter;
import lombok.Setter;

/**
 * Request inactivity and post-decision Engine visibility are separate deadlines.
 */
@Getter
@Setter
public final class RequestLifecycleConfig {

    private static final long ENGINE_FEEDBACK_GRACE_MS = 10_000L;

    private long dispatchTimeoutMs = 60_000L;
    private long inactivityTimeoutMs = 60_000L;
    private double engineVisibilityMultiplier = 2.0;

    /** Allows for prediction uncertainty and the Engine feedback interval. */
    public long visibilityDeadlineMs(long nowMs, long predictedDurationMs) {
        long delay = (long) Math.ceil(predictedDurationMs * engineVisibilityMultiplier);
        return com.google.common.math.LongMath.saturatedAdd(
                com.google.common.math.LongMath.saturatedAdd(nowMs, delay),
                ENGINE_FEEDBACK_GRACE_MS);
    }
}
