package org.flexlb.domain.worker;

import org.flexlb.dao.master.WorkerStatus;

public record ScoredWorker(WorkerStatus worker,
                           long schedulingScore,
                           long estimatedTtft,
                           long hitCacheTokens,
                           long lastSelectedTime) {}
