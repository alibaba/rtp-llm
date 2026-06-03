package org.flexlb.balance.strategy;

import java.util.List;

/**
 * Predicts prefill execution time for a batch of requests.
 */
public interface PrefillTimePredictor {

    long estimateMs(long totalTokens, long hitTokens);

    long predictBatchMs(List<RequestProfile> requests);
}
