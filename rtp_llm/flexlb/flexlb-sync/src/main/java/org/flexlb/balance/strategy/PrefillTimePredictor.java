package org.flexlb.balance.strategy;

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

    public long estimateMs(long totalTokens, long hitTokens) {
        return predictBatchMs(List.of(new BatchRequest(0, totalTokens, hitTokens)));
    }

    public long predictBatchMs(List<BatchRequest> requests) {
        if (requests.isEmpty()) {
            return 0;
        }
        int bs = requests.size();
        long sumC = 0;
        double sumQuadratic = 0;
        long sumP = 0;

        for (BatchRequest r : requests) {
            long c = r.computeTokens();
            long p = r.hitCache();
            sumC += c;
            sumQuadratic += a2 * c * c + a3 * c * p;
            sumP += p;
        }

        return (long) (a0 + a1 * sumC + sumQuadratic + a4 * sumP + a5 * bs);
    }
}
