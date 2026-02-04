package org.flexlb.domain.worker;

import org.flexlb.dao.master.WorkerStatus;

public record ScoredWorker(WorkerStatus worker, long ttft, long hitCacheTokens, long lastSelectedTime) {}
