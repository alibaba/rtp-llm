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

    public static <T> Comparator<Candidate<T>> evictionOrder() {
        return Comparator.<Candidate<T>>comparingInt(Candidate::priority)
                .thenComparingInt(c -> c.estimate().hitTokens() == 0 ? 0 : 1)
                .thenComparingDouble(c -> c.estimate().hitRate())
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

    /** Automatic soft shedding gives a young request its soft wait budget. Hard overload is immediate.
     * Filter the original victims rather than evicting higher-priority/cache-beneficial work instead.
     * Age is the original request age, so retries, migration and repeated bursts cannot renew it.
     */
    public static <T> List<T> adaptiveVictims(List<Candidate<T>> candidates,
            org.flexlb.config.PrefillQueueSheddingConfig config, PrefillTrafficTrend.Snapshot trend,
            java.util.function.ToLongFunction<T> ageMs) {
        var selected = selectAdaptiveVictims(candidates, config, trend);
        if (config.getRecovery() == null || config.getRecovery().getWindowMs() != 0
                || candidates.stream().mapToDouble(c -> c.estimate().fleetWorkMs()).sum() >= config.getMaxWaitMs()) {
            return selected;
        }
        return selected.stream().filter(value -> ageMs.applyAsLong(value) >= config.getExpectedWaitThresholdMs()).toList();
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
