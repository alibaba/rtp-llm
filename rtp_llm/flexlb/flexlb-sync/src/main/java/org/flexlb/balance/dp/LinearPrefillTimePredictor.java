package org.flexlb.balance.dp;

import org.flexlb.dao.master.TaskInfo;
import org.springframework.stereotype.Component;

/**
 * Default {@link PrefillTimePredictor} that delegates to the linear estimator
 * baked into {@link TaskInfo#estimatePrefillTimeMs}. Strictly monotonic in
 * {@code totalComputeTokens} (coefficient 1.0 > 0).
 */
@Component
public class LinearPrefillTimePredictor implements PrefillTimePredictor {

    @Override
    public long estimateMs(long totalComputeTokens, long totalHitCacheTokens) {
        return TaskInfo.estimatePrefillTimeMs(totalComputeTokens, totalHitCacheTokens);
    }
}
