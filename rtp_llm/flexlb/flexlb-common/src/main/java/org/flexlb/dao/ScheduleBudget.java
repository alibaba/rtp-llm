package org.flexlb.dao;

import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.util.PriorityNormalizer;

/**
 * Immutable per-request schedule budget: the single home of the normalized
 * priority, admission SLO, and admission deadline.
 *
 * <p>Created once at request admission by {@link #of(int, String, long, long, int, PrioritySloPolicy)},
 * which runs {@link PriorityNormalizer#normalize} and {@link PrioritySloPolicy#requestSloMs}
 * to derive the coarse deadline ({@code admittedAtMs + requestSloMs}). The
 * coarse deadline is the admission-gate deadline; endpoint-aware deadlines
 * (subtracting predicted prefill time) are computed downstream in
 * {@code buildEnvelope} and are <em>not</em> written back to the budget.
 *
 * <p>{@link #deadlineMs()} always returns the coarse deadline. A rescue
 * re-entry keeps the original budget (design doc 14.3: never extend the SLO).
 *
 * <p>When Auto-TPM is disabled the budget is {@code null} on the
 * {@link BalanceContext}; all accessors that delegate to budget perform a
 * null check and return a legacy-default value.
 *
 * @param priority      normalized priority (1-100, higher = more important)
 * @param source        whether the priority was explicitly carried or defaulted
 *                      (monitoring only — scheduling logic must not read this)
 * @param admittedAtMs  epoch-ms timestamp captured at admission
 * @param requestSloMs  per-request SLO in ms (seqLen bucket × priority multiplier)
 * @param deadlineMs    coarse admission deadline = admittedAtMs + requestSloMs
 */
public final class ScheduleBudget {

    /** First-route discount: attempt ≤ 1 gets 60 % of remaining time. */
    static final double FIRST_ROUTE_ALPHA = 0.6;

    private final int priority;
    private final PrioritySource source;
    private final long admittedAtMs;
    private final long requestSloMs;
    private final long deadlineMs;

    private ScheduleBudget(int priority, PrioritySource source, long admittedAtMs,
                           long requestSloMs, long deadlineMs) {
        this.priority = priority;
        this.source = source;
        this.admittedAtMs = admittedAtMs;
        this.requestSloMs = requestSloMs;
        this.deadlineMs = deadlineMs;
    }

    /**
     * Static factory: normalize the raw proto priority, derive the per-request
     * SLO from the {@link PrioritySloPolicy}, and compute the coarse admission
     * deadline ({@code admittedAtMs + requestSloMs}).
     *
     * @param rawProtoPriority value from the proto field; 0 means unset
     * @param qosHeader        raw {@code x-dashscope-inner-qos-level} header value, may be null/blank
     * @param seqLen           prompt sequence length (token count)
     * @param admittedAtMs     epoch-ms admission timestamp
     * @param defaultPriority  configured default priority (used when neither channel carries a value)
     * @param slo              SLO policy (seqLen buckets × priority multipliers)
     * @return an immutable {@link ScheduleBudget}
     */
    public static ScheduleBudget of(int rawProtoPriority, String qosHeader, long seqLen,
                                    long admittedAtMs, int defaultPriority, PrioritySloPolicy slo) {
        int priority = PriorityNormalizer.normalize(rawProtoPriority, qosHeader, defaultPriority);
        long sloMs = slo.requestSloMs(seqLen, priority);
        long deadlineMs = admittedAtMs + sloMs;
        boolean explicit = rawProtoPriority != 0
                || (qosHeader != null && !qosHeader.isBlank());
        PrioritySource source = explicit ? PrioritySource.EXPLICIT : PrioritySource.DEFAULT;
        return new ScheduleBudget(priority, source, admittedAtMs, sloMs, deadlineMs);
    }

    /**
     * Test helper: create a budget with an explicit priority and deadline,
     * bypassing the normal SLO computation. The {@code requestSloMs} is
     * derived as {@code deadlineMs - admittedAtMs} for consistency.
     */
    public static ScheduleBudget forDeadline(int priority, long admittedAtMs, long deadlineMs) {
        long sloMs = Math.max(0, deadlineMs - admittedAtMs);
        return new ScheduleBudget(priority, PrioritySource.EXPLICIT, admittedAtMs, sloMs, deadlineMs);
    }

    /** Normalized priority (1-100). */
    public int priority() { return priority; }

    /** Whether the priority was explicitly carried or defaulted (monitoring only). */
    public PrioritySource source() { return source; }

    /** Epoch-ms timestamp captured at admission. */
    public long admittedAtMs() { return admittedAtMs; }

    /** Per-request SLO in ms (seqLen bucket × priority multiplier). */
    public long requestSloMs() { return requestSloMs; }

    /** Coarse admission deadline (epoch ms) = admittedAtMs + requestSloMs. */
    public long deadlineMs() { return deadlineMs; }

    /** Remaining time before the deadline (ms); may be negative when expired. */
    public long remainingMs(long nowMs) {
        return deadlineMs - nowMs;
    }

    /** Whether the deadline has already passed. */
    public boolean expired(long nowMs) {
        return deadlineMs <= nowMs;
    }

    /**
     * Per-attempt routing deadline: the first route (attempt ≤ 1) gets a
     * discount of {@link #FIRST_ROUTE_ALPHA} × remaining to encourage early
     * commitment; subsequent attempts use the full deadline.
     */
    public long routeDeadlineMs(int attempt, long nowMs) {
        if (attempt <= 1) {
            return nowMs + Math.round(remainingMs(nowMs) * FIRST_ROUTE_ALPHA);
        }
        return deadlineMs;
    }

    /** Origin of the priority value — monitoring only, never read by scheduling logic. */
    public enum PrioritySource {
        /** Priority was explicitly carried (proto field or header). */
        EXPLICIT,
        /** Priority defaulted because no channel carried a value. */
        DEFAULT
    }
}
