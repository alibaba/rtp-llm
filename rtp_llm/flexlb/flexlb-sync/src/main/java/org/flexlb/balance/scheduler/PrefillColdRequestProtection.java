package org.flexlb.balance.scheduler;

/** Per-pool feedback from accepted, actual cold Prefill queue deadlines. Decision-owner confined. */
final class PrefillColdRequestProtection {
    static final long WINDOW_MS = 60_000L;
    static final long CONFIRM_MS = 120_000L;
    static final long QUIET_MS = 120_000L;
    static final long SOFT_QUIET_MS = 30_000L;
    static final long RISE_MS = 60_000L;
    static final long FALL_MS = 120_000L;

    enum State { NORMAL, CONFIRMING, PROTECTING, RECOVERING }
    record Snapshot(State state, double strength, double arrivalQps, double coldArrivalQps,
                    double earlyEvictionQps, long hardTimeouts) { }

    private State state = State.NORMAL;
    private final long[] hardBuckets = {Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE};
    private long firstHard = Long.MIN_VALUE;
    private long lastHard = Long.MIN_VALUE;
    private long lastSoft = Long.MIN_VALUE;
    private long lastTick = Long.MIN_VALUE;
    private long arrivals;
    private long coldArrivals;
    private long hardTimeouts;
    private double arrivalQps;
    private double coldArrivalQps;
    private double strength;
    private double rate;
    private double credit;

    void arrive() { arrivals++; }
    void coldArrival() { coldArrivals++; }

    void hardTimeout(long nowMs) {
        // Concurrent deadline producers may enqueue adjacent observations out of order.
        if (lastHard != Long.MIN_VALUE && nowMs < lastHard - QUIET_MS) { return; }
        if (lastHard == Long.MIN_VALUE || nowMs - lastHard > QUIET_MS) { firstHard = nowMs; }
        firstHard = Math.min(firstHard, nowMs);
        long bucket = Math.floorDiv(nowMs, WINDOW_MS);
        hardBuckets[Math.floorMod(bucket, hardBuckets.length)] = bucket;
        lastHard = Math.max(lastHard, nowMs);
        hardTimeouts++;
        if (state == State.NORMAL) { state = State.CONFIRMING; }
        if (state == State.RECOVERING) { state = State.PROTECTING; }
    }

    void evicted(long nowMs, int count) {
        if (count > 0) { lastSoft = nowMs; }
    }

    /** One control round. No scheduler stall can bank a catch-up ramp or whole-request credits. */
    int allowance(long nowMs) {
        if (lastTick != Long.MIN_VALUE && nowMs < lastTick) { throw new IllegalArgumentException("time moved backwards"); }
        long interval = lastTick == Long.MIN_VALUE ? 1_000L : Math.max(1L, nowMs - lastTick);
        long step = lastTick == Long.MIN_VALUE ? 0L : Math.min(1_000L, interval);
        double observedQps = arrivals * 1_000.0 / interval;
        double observedColdQps = coldArrivals * 1_000.0 / interval;
        double alpha = 1.0 - Math.exp(-interval / 30_000.0);
        arrivalQps = lastTick == Long.MIN_VALUE ? observedQps : arrivalQps + alpha * (observedQps - arrivalQps);
        coldArrivalQps = lastTick == Long.MIN_VALUE ? observedColdQps : coldArrivalQps + alpha * (observedColdQps - coldArrivalQps);
        arrivals = coldArrivals = 0L;
        lastTick = nowMs;
        long quiet = lastHard == Long.MIN_VALUE ? Long.MAX_VALUE : nowMs - lastHard;
        if (state == State.CONFIRMING) {
            if (quiet > QUIET_MS) {
                state = State.NORMAL;
                firstHard = Long.MIN_VALUE;
            } else if (nowMs - firstHard >= CONFIRM_MS && completeWindows(nowMs)) {
                state = State.PROTECTING;
                step = 0L;
            }
        }
        boolean softQuiet = lastSoft == Long.MIN_VALUE || nowMs - lastSoft >= SOFT_QUIET_MS;
        if (state == State.PROTECTING) {
            if (quiet >= QUIET_MS && softQuiet) { state = State.RECOVERING; }
            else if (quiet <= WINDOW_MS) { strength = Math.min(1.0, strength + (double) step / RISE_MS); }
        }
        if (state == State.RECOVERING) {
            if (!softQuiet) { state = State.PROTECTING; }
            else {
                strength = Math.max(0.0, strength - (double) step / FALL_MS);
                if (strength < 1e-10) {
                    strength = 0.0;
                    state = State.NORMAL;
                    firstHard = lastHard = Long.MIN_VALUE;
                    java.util.Arrays.fill(hardBuckets, Long.MIN_VALUE);
                }
            }
        }
        if (strength == 0.0) { rate = credit = 0.0; return 0; }
        rate = Math.min(strength * coldArrivalQps, rate + Math.max(0.1, 0.01 * arrivalQps));
        credit += rate;
        int count = (int) Math.min(Integer.MAX_VALUE, Math.floor(credit));
        credit -= Math.floor(credit);
        return count;
    }

    private boolean completeWindows(long nowMs) {
        long bucket = Math.floorDiv(nowMs, WINDOW_MS);
        return hardBuckets[Math.floorMod(bucket - 1L, hardBuckets.length)] == bucket - 1L
                && hardBuckets[Math.floorMod(bucket - 2L, hardBuckets.length)] == bucket - 2L;
    }

    Snapshot snapshot() { return new Snapshot(state, strength, arrivalQps, coldArrivalQps, rate, hardTimeouts); }
}
