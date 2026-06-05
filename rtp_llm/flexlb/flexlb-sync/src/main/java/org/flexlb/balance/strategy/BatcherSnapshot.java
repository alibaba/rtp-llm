package org.flexlb.balance.strategy;

import java.util.List;

public record BatcherSnapshot(
        int queueSize,
        List<RequestProfile> requests,
        long earliestEnqueueTimeMs,
        long headDeadlineMs) {

    public static final BatcherSnapshot EMPTY = new BatcherSnapshot(0, List.of(), Long.MAX_VALUE, Long.MAX_VALUE);

    public long totalInputTokens() {
        long total = 0;
        for (RequestProfile r : requests) {
            total += r.inputLen();
        }
        return total;
    }

    public long totalHitCacheTokens() {
        long total = 0;
        for (RequestProfile r : requests) {
            total += r.hitCacheTokens();
        }
        return total;
    }
}
