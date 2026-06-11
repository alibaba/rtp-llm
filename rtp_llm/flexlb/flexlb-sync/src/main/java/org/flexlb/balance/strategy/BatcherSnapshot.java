package org.flexlb.balance.strategy;

import java.util.List;

public record BatcherSnapshot(
        int queueSize,
        List<BatchRequest> requests,
        long earliestEnqueueTimeMs,
        long headDeadlineMs) {

    public static final BatcherSnapshot EMPTY = new BatcherSnapshot(0, List.of(), Long.MAX_VALUE, Long.MAX_VALUE);

    public long totalInputTokens() {
        long total = 0;
        for (BatchRequest r : requests) {
            total += r.seqLen();
        }
        return total;
    }

    public long totalHitCacheTokens() {
        long total = 0;
        for (BatchRequest r : requests) {
            total += r.hitCache();
        }
        return total;
    }
}
