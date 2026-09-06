package org.flexlb.cache.domain;

import java.util.List;

public record BlockHashCalculationResult(List<Long> blockCacheKeys, long queueWaitTimeUs, long executionTimeUs) {
}
