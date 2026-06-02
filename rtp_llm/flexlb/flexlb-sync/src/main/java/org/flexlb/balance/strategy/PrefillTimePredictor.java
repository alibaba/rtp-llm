package org.flexlb.balance.strategy;

import java.util.List;

/**
 * α₀ + α₁·Σcᵢ + max_i(α₂·cᵢ² + α₃·cᵢ·pᵢ) + α₄·max(pᵢ) + α₅·bs
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
        return predictBatchMs(List.of(new RequestProfile(totalTokens, hitTokens)));
    }

    public long predictBatchMs(List<RequestProfile> requests) {
        if (requests.isEmpty()) {
            return 0;
        }
        int bs = requests.size();
        long sumC = 0;
        double maxQuadratic = 0;
        long maxP = 0;

        for (RequestProfile r : requests) {
            long c = r.computeTokens();
            long p = r.hitCacheTokens();
            sumC += c;
            maxQuadratic = Math.max(maxQuadratic, a2 * c * c + a3 * c * p);
            maxP = Math.max(maxP, p);
        }

        return (long) (a0 + a1 * sumC + maxQuadratic + a4 * maxP + a5 * bs);
    }
}
