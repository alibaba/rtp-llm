package org.flexlb.balance.scheduler;

import org.flexlb.config.PrefillQueueSheddingConfig.DeadlineBudgetConfig;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/** Cold-only queue relief using calibrated work, actual capacity pressure and request deadlines.
 * The fluid fleet estimate is advisory: cache affinity and Decode can still limit completion.
 */
public final class PrefillDeadlineBudget {
    public record Fleet(int workers, int saturatedWorkers, double outstandingTokens, boolean known) {
        public boolean saturated(DeadlineBudgetConfig config) {
            return known && workers > 0 && Double.isFinite(outstandingTokens) && outstandingTokens >= 0
                    && saturatedWorkers * 100.0 >= workers * config.getSaturatedWorkerPercent();
        }
    }

    public record Candidate<T>(T value, int priority, long sequence, long computeTokens,
                               boolean cold, long remainingMs) { }

    private long saturatedSinceMs = Long.MIN_VALUE;
    private long lastRoundMs = Long.MIN_VALUE;
    private int previousWorkers;

    /** Called at the scheduler scan cadence, even when the queue is empty. */
    public <T> List<T> victims(List<Candidate<T>> candidates, Fleet fleet, DeadlineBudgetConfig config,
                              boolean priorityOrdering, long nowMs) {
        if (fleet == null || !fleet.saturated(config) || candidates.isEmpty()) {
            saturatedSinceMs = Long.MIN_VALUE;
            return List.of();
        }
        // Expansion/shrinkage invalidates the previous pressure episode immediately.
        if (saturatedSinceMs == Long.MIN_VALUE || previousWorkers != fleet.workers()) {
            saturatedSinceMs = nowMs;
            previousWorkers = fleet.workers();
        }
        if (nowMs - saturatedSinceMs < config.getSaturationHoldMs()
                || lastRoundMs != Long.MIN_VALUE && nowMs - lastRoundMs < PrefillQueueShedding.SOFT_INTERVAL_MS) {
            return List.of();
        }
        lastRoundMs = nowMs;
        double tokensPerMs = fleet.workers() * (config.getTokensPerSecond() / 1000.0) * config.getCapacityFactor();
        if (!Double.isFinite(tokensPerMs) || tokensPerMs <= 0) { return List.of(); }
        var ordered = new ArrayList<>(candidates);
        Comparator<Candidate<T>> order = Comparator.comparingLong(Candidate::sequence);
        if (priorityOrdering) { order = Comparator.<Candidate<T>>comparingInt(Candidate::priority).reversed().thenComparing(order); }
        ordered.sort(order);
        // FIFO ignores request priority. Within the active ordering tier, remove
        // the most work first to reject fewer requests for the same relief.
        Comparator<Candidate<T>> victimOrder = Comparator.comparingLong((Candidate<T> c) -> c.computeTokens()).reversed()
                .thenComparing(Comparator.comparingLong((Candidate<T> c) -> c.sequence()).reversed());
        if (priorityOrdering) {
            victimOrder = Comparator.<Candidate<T>>comparingInt(Candidate::priority).thenComparing(victimOrder);
        }
        var cold = new PriorityQueue<>(victimOrder);
        double queuedTokens = 0;
        var selected = new ArrayList<T>();
        for (var candidate : ordered) {
            // Work before this request determines its dispatch eligibility, not its own service time.
            double ahead = queuedTokens;
            queuedTokens += candidate.computeTokens();
            if (candidate.cold() && candidate.computeTokens() > 0) { cold.add(candidate); }
            double budget = Math.max(0L, candidate.remainingMs() - config.getDispatchSlackMs()) * tokensPerMs;
            while (fleet.outstandingTokens() + ahead > budget && !cold.isEmpty()) {
                if (priorityOrdering && cold.peek().priority() > candidate.priority()) { break; }
                var victim = cold.remove();
                selected.add(victim.value());
                queuedTokens -= victim.computeTokens();
                if (victim == candidate) { break; }
                ahead -= victim.computeTokens();
            }
        }
        return selected;
    }
}
