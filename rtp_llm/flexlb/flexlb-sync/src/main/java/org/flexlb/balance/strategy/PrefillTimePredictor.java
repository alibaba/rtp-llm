package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.List;

/**
 * α₀ + α₁·Σcᵢ + α₂·Σcᵢ² + α₃·Σ(cᵢ·pᵢ) + α₄·Σpᵢ + α₅·bs
 *
 * where cᵢ = inputLen - hitCacheTokens (compute tokens), pᵢ = hitCacheTokens, bs = batch size.
 */
public class PrefillTimePredictor {

    private final double a0, a1, a2, a3, a4, a5;

    public PrefillTimePredictor(double a0, double a1, double a2, double a3, double a4, double a5) {
        this.a0 = a0;
        this.a1 = a1;
        this.a2 = a2;
        this.a3 = a3;
        this.a4 = a4;
        this.a5 = a5;
    }

    /** Estimate prefill time for a single request from raw token counts. */
    public long estimateMs(long totalTokens, long hitTokens) {
        long c = Math.max(0, totalTokens - hitTokens);
        return (long) (a0 + a1 * c + a2 * c * c + a3 * c * hitTokens + a4 * hitTokens + a5);
    }

    /** Estimate prefill time for a batch of {@link BatchItem}s. */
    public long predictBatchMs(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0;
        }
        int bs = items.size();
        long sumC = 0;
        double sumQuadratic = 0;
        long sumP = 0;

        for (BatchItem item : items) {
            long c = item.computeTokens();
            long p = item.hitCache();
            sumC += c;
            sumQuadratic += a2 * c * c + a3 * c * p;
            sumP += p;
        }

        return (long) (a0 + a1 * sumC + sumQuadratic + a4 * sumP + a5 * bs);
    }
}
