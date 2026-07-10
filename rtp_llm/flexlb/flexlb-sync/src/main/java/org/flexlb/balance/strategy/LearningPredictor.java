package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Prefill-time predictor with hardcoded linear regression and online learning.
 *
 * <p>Formula: {@code y = w0 + w1 * sum(computeTokens) + w2 * sum(hitCacheTokens)}
 * where {@code computeTokens = inputTokens - hitCacheTokens} (floor at 0).
 *
 * <p>The three weights ({@code w0}, {@code w1}, {@code w2}) are volatile doubles
 * that can be updated at runtime via {@link #setParameter(String, double)}.
 * The {@link #learn(List, long, long)} callback is a stub — the actual online
 * learning algorithm will be implemented in a future iteration.
 */
public class LearningPredictor implements PrefillTimePredictor {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private volatile double w0 = 50.0;
    private volatile double w1 = 0.5;
    private volatile double w2 = 0.3;

    public LearningPredictor() {
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        long inputTokens = Math.max(0L, totalTokens);
        long boundedHit = Math.max(0L, Math.min(hitTokens, inputTokens));
        long computeTokens = inputTokens - boundedHit;
        return (long) (w0 + w1 * computeTokens + w2 * boundedHit);
    }

    @Override
    public long predictBatchMs(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0;
        }
        double sumCompute = 0;
        double sumHit = 0;
        for (BatchItem item : items) {
            long seq = Math.max(0L, item.seqLen());
            long hit = Math.max(0L, Math.min(item.hitCache(), seq));
            sumCompute += (seq - hit);
            sumHit += hit;
        }
        return (long) (w0 + w1 * sumCompute + w2 * sumHit);
    }

    @Override
    public void learn(List<BatchItem> items, long predictedMs, long actualMs) {
        logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}",
                items != null ? items.size() : 0, predictedMs, actualMs);
    }

    // ---- parameter management ----

    public double getParameter(String name) {
        return switch (name) {
            case "w0" -> w0;
            case "w1" -> w1;
            case "w2" -> w2;
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        };
    }

    public void setParameter(String name, double value) {
        switch (name) {
            case "w0" -> w0 = value;
            case "w1" -> w1 = value;
            case "w2" -> w2 = value;
            default -> throw new IllegalArgumentException("Unknown parameter: " + name);
        }
    }

    public Set<String> parameterNames() {
        return Set.of("w0", "w1", "w2");
    }

    public Map<String, Double> getParameters() {
        return Map.of("w0", w0, "w1", w1, "w2", w2);
    }

    public boolean hasParameters() {
        return true;
    }

    public String formulaString() {
        return "w0 + w1*sum(computeTokens) + w2*sum(hitCacheTokens)"
                + " [w0=" + w0 + ", w1=" + w1 + ", w2=" + w2 + "]";
    }
}
