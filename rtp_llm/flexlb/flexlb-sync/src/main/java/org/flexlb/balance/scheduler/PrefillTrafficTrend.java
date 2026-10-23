package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig;

import java.util.ArrayDeque;

/** Bounded per-pool arrival-rate and queue-drain detector. Times are monotonic milliseconds.
 * The decision owner records new arrivals once; retries and migrations are not arrivals.
 */
public final class PrefillTrafficTrend {
    public enum State { WARMUP, NORMAL, BURST_PROBE, GROWING, RECOVERING, GRACE_EXHAUSTED }
    public record Snapshot(State state, double arrivalQps, double previousQps,
                           double inputTps, double previousInputTps, double queuedWorkMs,
                           boolean grace, long graceRemainingMs) { }
    private record Sample(long at, long arrivals, long tokens, double queue, double discarded) { }
    private final PrefillQueueSheddingConfig.RecoveryConfig config;
    private final ArrayDeque<Sample> samples = new ArrayDeque<>();
    private long arrivals;
    private long tokens;
    private double discardedWorkMs;
    private long graceStarted = -1;
    private boolean episodeUsed;
    private boolean probeUsed;
    private long probeStarted = -1;
    private double graceQpsCeiling;
    private double graceTpsCeiling;
    private long episodeStarted = -1;
    private long quietStarted = -1;
    private long automaticProbeMs;

    public PrefillTrafficTrend(PrefillQueueSheddingConfig.RecoveryConfig config) {
        this.config = config;
    }

    public void arrive(long inputTokens) {
        if (inputTokens < 0) { throw new IllegalArgumentException("negative input tokens"); }
        arrivals++;
        tokens = Math.addExact(tokens, inputTokens);
    }

    /** Own shedding/expiry must not masquerade as natural queue drainage. */
    public void discard(double fleetWorkMs) {
        if (!Double.isFinite(fleetWorkMs) || fleetWorkMs < 0) { throw new IllegalArgumentException("invalid discarded work"); }
        discardedWorkMs += fleetWorkMs;
    }

    /** No grace for a growing/stationary queue, missing history, or hard overload.
     * One finite grace budget per overload episode prevents indefinite refusal suppression.
     */
    public Snapshot observe(long nowMs, double queuedWorkMs, long normalThresholdMs, long hardThresholdMs) {
        return observe(nowMs, queuedWorkMs, normalThresholdMs, hardThresholdMs, 0);
    }

    public Snapshot observe(long nowMs, double queuedWorkMs, long normalThresholdMs, long hardThresholdMs,
                            long longestServiceMs) {
        if (!Double.isFinite(queuedWorkMs) || queuedWorkMs < 0) {
            throw new IllegalArgumentException("invalid queue work");
        }
        if (!samples.isEmpty() && nowMs < samples.getLast().at) {
            throw new IllegalArgumentException("time moved backwards");
        }
        if (config.getWindowMs() == 0) {
            return observeAutomatic(nowMs, queuedWorkMs, normalThresholdMs, hardThresholdMs, longestServiceMs);
        }
        if (queuedWorkMs <= normalThresholdMs) {
            graceStarted = -1;
            episodeUsed = false;
            probeUsed = false;
            probeStarted = -1;
        }
        long window = config.getWindowMs();
        Sample middle = null, older = null;
        for (Sample sample : samples) {
            if (sample.at <= nowMs - window) { middle = sample; }
            if (sample.at <= nowMs - 2 * window) { older = sample; }
        }
        Snapshot result;
        if (middle == null || older == null || middle.at == older.at) {
            result = new Snapshot(State.WARMUP, 0, 0, 0, 0, queuedWorkMs, false, 0);
        } else {
            double recentMs = nowMs - middle.at, previousMs = middle.at - older.at;
            double qps = (arrivals - middle.arrivals) * 1000.0 / recentMs;
            double oldQps = (middle.arrivals - older.arrivals) * 1000.0 / previousMs;
            double tps = (tokens - middle.tokens) * 1000.0 / recentMs;
            double oldTps = (middle.tokens - older.tokens) * 1000.0 / previousMs;
            boolean declining = oldQps > 0 && oldTps > 0
                    && qps <= oldQps * (1 - config.getMinArrivalDropFraction())
                    && tps <= oldTps * (1 - config.getMinArrivalDropFraction());
            boolean draining = middle.queue > normalThresholdMs
                    && queuedWorkMs + discardedWorkMs - middle.discarded
                    <= middle.queue * (1 - config.getMinQueueDrainFraction());
            boolean remainingBelowPeak = episodeUsed && qps <= graceQpsCeiling && tps <= graceTpsCeiling;
            boolean recovery = (declining || remainingBelowPeak) && draining && queuedWorkMs > normalThresholdMs
                    && queuedWorkMs < hardThresholdMs;
            if (recovery && !episodeUsed) {
                graceStarted = nowMs;
                episodeUsed = true;
                graceQpsCeiling = oldQps * (1 - config.getMinArrivalDropFraction());
                graceTpsCeiling = oldTps * (1 - config.getMinArrivalDropFraction());
            }
            long remaining = graceStarted < 0 ? 0 : Math.max(0, (config.getMaxGraceMs() > 0
                    ? config.getMaxGraceMs() : Math.max(0, hardThresholdMs - normalThresholdMs)) - (nowMs - graceStarted));
            boolean grace = recovery && remaining > 0;
            State state = grace ? State.RECOVERING
                    : recovery && episodeUsed ? State.GRACE_EXHAUSTED
                    : qps > oldQps && queuedWorkMs >= middle.queue ? State.GROWING : State.NORMAL;
            result = new Snapshot(state, qps, oldQps, tps, oldTps, queuedWorkMs, grace, grace ? remaining : 0);
        }
        boolean overloaded = queuedWorkMs > normalThresholdMs && queuedWorkMs < hardThresholdMs;
        boolean sudden = result.state() == State.WARMUP || result.arrivalQps() > 2 * result.previousQps()
                && result.inputTps() > 2 * result.previousInputTps();
        if (overloaded && sudden && !probeUsed) {
            probeUsed = true;
            probeStarted = nowMs;
        }
        long probeRemaining = probeStarted < 0 ? 0 : Math.max(0, (config.getBurstProbeMs() > 0
                ? config.getBurstProbeMs() : window * 2) - (nowMs - probeStarted));
        if (overloaded && probeRemaining > 0 && !result.grace()) {
            result = new Snapshot(State.BURST_PROBE, result.arrivalQps(), result.previousQps(),
                    result.inputTps(), result.previousInputTps(), queuedWorkMs, true, probeRemaining);
        }
        if (samples.isEmpty() || nowMs - samples.getLast().at >= Math.max(1, window / 4)) {
            samples.addLast(new Sample(nowMs, arrivals, tokens, queuedWorkMs, discardedWorkMs));
        }
        // Retain one sample at/before the oldest lookup boundary, plus the two windows.
        while (samples.size() > 2) {
            var iterator = samples.iterator(); iterator.next();
            if (iterator.next().at > nowMs - 2 * window) { break; }
            samples.removeFirst();
        }
        return result;
    }

    /** A burst gets an observation interval, then only demonstrable natural drainage gets grace.
     * Budget starts at first overload, never at each retry or recovery sample. Rearm requires quiet history.
     */
    private Snapshot observeAutomatic(long now, double queue, long soft, long hard, long longestServiceMs) {
        long window = Math.clamp(soft / 10, 1, 2_000);
        long budget = Math.max(0, Math.min(soft, hard - soft));
        if (config.getMaxGraceMs() > 0) { budget = Math.min(budget, config.getMaxGraceMs()); }
        if (queue <= soft) {
            if (quietStarted < 0) { quietStarted = now; }
            if (now - quietStarted >= window * 2) { episodeStarted = -1; }
        } else {
            quietStarted = -1;
            if (episodeStarted < 0) {
                episodeStarted = now;
                // A discrete long Prefill can make a healthy queue look flat until its first completion.
                // Freeze the observation budget at episode start; later arrivals cannot extend it.
                automaticProbeMs = Math.min(budget, config.getBurstProbeMs() > 0 ? config.getBurstProbeMs()
                        : Math.max(window * 2, Math.min(soft / 4, Math.max(0, longestServiceMs)) * 2));
            }
        }
        Sample middle = null, older = null;
        for (var sample : samples) {
            if (sample.at <= now - window) { middle = sample; }
            if (sample.at <= now - window * 2) { older = sample; }
        }
        double qps = 0, tps = 0, oldQps = 0, oldTps = 0;
        double drain = 0;
        if (middle != null) {
            double elapsed = now - middle.at;
            qps = (arrivals - middle.arrivals) * 1000.0 / elapsed;
            tps = (tokens - middle.tokens) * 1000.0 / elapsed;
            drain = (middle.queue - queue - (discardedWorkMs - middle.discarded)) / elapsed;
            if (older != null && middle.at > older.at) {
                oldQps = (middle.arrivals - older.arrivals) * 1000.0 / (middle.at - older.at);
                oldTps = (middle.tokens - older.tokens) * 1000.0 / (middle.at - older.at);
            }
        }
        long remaining = episodeStarted < 0 ? 0 : Math.max(0, budget - (now - episodeStarted));
        long probe = automaticProbeMs;
        boolean overloaded = queue > soft && queue < hard;
        boolean probing = overloaded && remaining > 0 && now - episodeStarted < probe;
        // Use net service progress, not QPS alone: larger queries and our own evictions cannot fake recovery.
        boolean recoverable = overloaded && drain > 0 && (queue - soft) / drain <= remaining;
        boolean grace = probing || recoverable && remaining > 0;
        State state = probing ? State.BURST_PROBE : grace ? State.RECOVERING
                : queue <= soft ? State.NORMAL : remaining == 0 ? State.GRACE_EXHAUSTED : State.GROWING;
        if (samples.isEmpty() || now - samples.getLast().at >= Math.max(1, window / 4)) {
            samples.addLast(new Sample(now, arrivals, tokens, queue, discardedWorkMs));
        }
        while (samples.size() > 2) {
            var iterator = samples.iterator(); iterator.next();
            if (iterator.next().at > now - window * 2) { break; }
            samples.removeFirst();
        }
        return new Snapshot(state, qps, oldQps, tps, oldTps, queue, grace, grace ? remaining : 0);
    }
}
