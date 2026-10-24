package org.flexlb.balance.queue;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * 在队列锁内取出的队列内容快照，创建后不可变。
 */
public record QueuedRequestsSnapshot(
        String endpointId,
        long queueVersion,
        List<ScheduledRequest> items) {
    public QueuedRequestsSnapshot {
        if (queueVersion < 0L) {
            throw new IllegalArgumentException(
                    "queueVersion must be non-negative");
        }
        items = List.copyOf(items);
    }
}
