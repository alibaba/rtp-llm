package org.flexlb.balance.scheduler.priority;

import java.util.List;

/**
 * Strongly-consistent point-in-time view of one prefill batcher queue: the
 * membership and version are captured in the same {@code queueLock} critical
 * section (see {@code PrefillQueueManager.snapshot()}; the sort may run on
 * the thread-confined copy outside the lock), so version and
 * items always belong to the same queue state. {@link #queueVersion} is
 * re-validated by
 * the atomic replace/remove/offer operations so a plan built on this snapshot
 * can never be applied against a mutated queue.
 *
 * @param endpointId    prefill endpoint key ("ip:httpPort")
 * @param queueVersion  queue version at capture time
 * @param queueCapacity configured hard queue limit (flexlbBatchQueueMaxSize;
 *                      0 = unbounded)
 * @param items         queued requests in queue order (head first)
 */
public record PrefillQueueSnapshot(
        String endpointId,
        long queueVersion,
        int queueCapacity,
        List<QueuedRequestSnapshot> items) {
}
