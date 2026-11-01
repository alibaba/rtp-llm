package org.flexlb.balance.scheduler;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Reactive overload policy. Only observed queue age can authorize eviction. */
public final class PrefillQueueShedding {
    private PrefillQueueShedding() { }

    /** Advisory best-cache service estimate, captured before endpoint admission filters. */
    public record Estimate(long inputTokens, long hitTokens, long workMs, int workers) {
        public Estimate {
            if (inputTokens < 0 || hitTokens < 0 || hitTokens > inputTokens || workMs < 0 || workers <= 0) {
                throw new IllegalArgumentException("invalid Prefill queue estimate");
            }
        }
        public double fleetWorkMs() { return (double) workMs / workers; }
        public double hitRate() {
            return inputTokens == 0 ? 0.0 : (double) hitTokens / inputTokens;
        }
    }

    /** waitedMs is actual elapsed time in the placement queue, never a service prediction. */
    public record Candidate<T>(T value, int priority, long sequence, Estimate estimate, long waitedMs) { }

    /** Fixed control cadence prevents a busy scheduling loop from amplifying soft eviction. */
    public static final long SOFT_INTERVAL_MS = 1_000L;

    public static <T> Comparator<Candidate<T>> evictionOrder() {
        return Comparator.<Candidate<T>>comparingInt(Candidate::priority)
                .thenComparingInt(c -> c.estimate().hitTokens() == 0 ? 0 : 1)
                .thenComparingDouble(c -> c.estimate().hitRate())
                .thenComparing(Comparator.comparingLong((Candidate<T> c) -> c.sequence()).reversed());
    }

    /** Per-pool feedback. Credits are rates integrated over elapsed time, not a fresh
     * percentage of the same backlog each round. Own evictions are excluded when
     * detecting natural queue drainage. Never bank unused whole-request credits.
     */
    public static final class Controller {
        private long lastRoundMs = Long.MIN_VALUE;
        private int remainingAfterEviction;
        private double credit;

        public <T> List<T> victims(List<Candidate<T>> candidates, long thresholdMs, long maxWaitMs, long nowMs) {
            return victims(candidates, thresholdMs, maxWaitMs, nowMs, 0.1);
        }

        private <T> List<T> victims(List<Candidate<T>> candidates, long thresholdMs, long maxWaitMs,
                long nowMs, double maxFraction) {
            if (lastRoundMs != Long.MIN_VALUE && nowMs - lastRoundMs < SOFT_INTERVAL_MS) { return List.of(); }
            long elapsedMs = lastRoundMs == Long.MIN_VALUE ? SOFT_INTERVAL_MS
                    : Math.min(SOFT_INTERVAL_MS, nowMs - lastRoundMs);
            boolean draining = lastRoundMs != Long.MIN_VALUE && candidates.size() < remainingAfterEviction;
            lastRoundMs = nowMs;
            remainingAfterEviction = candidates.size();
            var overdue = new ArrayList<Candidate<T>>();
            double rate = 0;
            for (var candidate : candidates) {
                if (candidate.waitedMs() <= thresholdMs) { continue; }
                overdue.add(candidate);
                rate += (double) elapsedMs / Math.max(SOFT_INTERVAL_MS, maxWaitMs - candidate.waitedMs());
            }
            if (overdue.isEmpty() || draining) { credit = 0; return List.of(); }
            credit += rate;
            // At most 10% of overdue requests per round; hard deadlines remain independent.
            int count = Math.min((int) Math.floor(credit), Math.max(1, (int) Math.ceil(overdue.size() * maxFraction)));
            credit -= Math.floor(credit);
            overdue.sort(evictionOrder());
            remainingAfterEviction -= count;
            return overdue.subList(0, count).stream().map(Candidate::value).toList();
        }
    }

    /** Optional recovery only reduces age-based eviction; prediction never activates it. */
    public static <T> List<T> adaptiveVictims(List<Candidate<T>> candidates,
            org.flexlb.config.PrefillQueueSheddingConfig config, PrefillTrafficTrend.Snapshot trend,
            Controller controller, long nowMs) {
        long threshold = config.getExpectedWaitThresholdMs();
        if (config.getRecovery() != null && trend != null && trend.grace()) {
            threshold = trend.state() == PrefillTrafficTrend.State.BURST_PROBE ? config.getMaxWaitMs()
                    : (long) Math.min(config.getMaxWaitMs(), threshold * config.getRecovery().getWaitMultiplier());
        }
        double maxFraction = config.getRecovery() != null && trend != null && trend.grace()
                ? Math.min(0.1, config.getRecovery().getMaxEvictionFraction()) : 0.1;
        return controller.victims(candidates, threshold, config.getMaxWaitMs(), nowMs, maxFraction);
    }
}
