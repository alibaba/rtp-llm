package org.flexlb.balance.scheduler;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Pure overload policy. Work is expressed in milliseconds of one fleet's service capacity. */
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

    public record Candidate<T>(T value, int priority, long sequence, Estimate estimate) { }

    private static <T> Comparator<Candidate<T>> benefitOrder() {
        return Comparator.<Candidate<T>>comparingInt(Candidate::priority)
                .thenComparingInt(c -> c.estimate().hitTokens() == 0 ? 0 : 1)
                .thenComparingDouble(c -> c.estimate().hitRate());
    }

    public static <T> Comparator<Candidate<T>> evictionOrder() {
        return PrefillQueueShedding.<T>benefitOrder()
                .thenComparing(Comparator.comparingLong((Candidate<T> c) -> c.sequence()).reversed());
    }

    /** Priority dominates cache benefit; exact ties retain older requests. Input is not mutated. */
    public static <T> List<T> victims(List<Candidate<T>> candidates, long thresholdMs) {
        double workMs = candidates.stream().mapToDouble(c -> c.estimate().fleetWorkMs()).sum();
        if (workMs <= thresholdMs) { return List.of(); }
        var sorted = new ArrayList<>(candidates);
        sorted.sort(evictionOrder());
        var victims = new ArrayList<T>();
        for (var candidate : sorted) {
            if (workMs <= thresholdMs) { break; }
            victims.add(candidate.value());
            workMs -= candidate.estimate().fleetWorkMs();
        }
        return victims;
    }
    /** Same bounded recovery decision for production and deterministic traffic experiments. */
    public static <T> List<T> adaptiveVictims(List<Candidate<T>> candidates,
            org.flexlb.config.PrefillQueueSheddingConfig config, PrefillTrafficTrend.Snapshot trend) {
        return adaptiveVictims(candidates, config, trend, ignored -> Long.MAX_VALUE);
    }

    /**
     * Estimates select excess work, but cannot empty the service feed in one step.
     * Automatic recovery already observes each overload episode. Hard overload must
     * not give every new cold arrival a fresh grace period ahead of older hot work.
     * Original request deadlines remain authoritative, including for the feed reserve.
     */
    public static <T> List<T> adaptiveVictims(List<Candidate<T>> candidates,
            org.flexlb.config.PrefillQueueSheddingConfig config, PrefillTrafficTrend.Snapshot trend,
            java.util.function.ToLongFunction<T> ageMs) {
        var selected = selectAdaptiveVictims(candidates, config, trend);
        if (config.getRecovery() == null || config.getRecovery().getWindowMs() != 0 || selected.isEmpty()) {
            return selected;
        }
        double work = candidates.stream().mapToDouble(c -> c.estimate().fleetWorkMs()).sum();
        long observationMs = work >= config.getMaxWaitMs() ? 0L : config.getExpectedWaitThresholdMs();
        int workers = candidates.stream().mapToInt(c -> c.estimate().workers()).max().orElse(0);
        int reserve = Math.min(candidates.size(), workers);
        // Retain a quarter for transient prediction error while promptly removing excess cold work.
        int limit = Math.min(candidates.size() - reserve, (int) ((3L * candidates.size() + 3L) / 4));
        if (observationMs == 0L) {
            return List.copyOf(selected.subList(0, Math.min(limit, selected.size())));
        }
        var byValue = new java.util.HashMap<T, Candidate<T>>();
        candidates.forEach(candidate -> byValue.put(candidate.value(), candidate));
        Comparator<Candidate<T>> benefit = benefitOrder();
        Candidate<T> protectedTier = null;
        for (T value : selected) {
            if (ageMs.applyAsLong(value) < observationMs) {
                var candidate = byValue.get(value);
                if (protectedTier == null || benefit.compare(candidate, protectedTier) < 0) {
                    protectedTier = candidate;
                }
            }
        }
        var result = new ArrayList<T>();
        for (T value : selected) {
            if (result.size() >= limit) { break; }
            // Same-tier older work can still be shed. A young cold request must never
            // redirect its rejection onto a higher-priority or better-cache request.
            if (protectedTier != null && benefit.compare(byValue.get(value), protectedTier) > 0) { break; }
            if (ageMs.applyAsLong(value) >= observationMs) { result.add(value); }
        }
        return result;
    }

    private static <T> List<T> selectAdaptiveVictims(List<Candidate<T>> candidates,
            org.flexlb.config.PrefillQueueSheddingConfig config, PrefillTrafficTrend.Snapshot trend) {
        if (config.getRecovery() == null || trend == null || !trend.grace()) {
            return victims(candidates, config.getExpectedWaitThresholdMs());
        }
        if (trend.state() == PrefillTrafficTrend.State.BURST_PROBE || config.getRecovery().getWindowMs() == 0) {
            return victims(candidates, config.getMaxWaitMs());
        }
        long threshold = (long) Math.min(config.getMaxWaitMs(),
                config.getExpectedWaitThresholdMs() * config.getRecovery().getWaitMultiplier());
        var selected = victims(candidates, threshold);
        int count = Math.min(selected.size(), Math.max(1,
                (int) Math.ceil(candidates.size() * config.getRecovery().getMaxEvictionFraction())));
        return List.copyOf(selected.subList(0, count));
    }

}
