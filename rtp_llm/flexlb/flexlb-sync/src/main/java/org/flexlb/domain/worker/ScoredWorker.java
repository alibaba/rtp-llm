package org.flexlb.domain.worker;

import org.flexlb.dao.master.WorkerStatus;

public record ScoredWorker(WorkerStatus worker,
                           long ttft,
                           long hitCacheTokens,
                           long lastSelectedTime,
                           long localMatchTokens,
                           long p2pFetchTokens,
                           long p2pTotalMatchTokens) {

    public ScoredWorker(WorkerStatus worker, long ttft, long hitCacheTokens, long lastSelectedTime) {
        this(worker, ttft, hitCacheTokens, lastSelectedTime, 0, 0, 0);
    }
}
