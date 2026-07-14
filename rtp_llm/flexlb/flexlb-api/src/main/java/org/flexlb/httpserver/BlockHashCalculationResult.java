package org.flexlb.httpserver;

import java.util.List;

public record BlockHashCalculationResult(
        List<Long> blockCacheKeys,
        long queueWaitTimeUs,
        long executionTimeUs) {
}
