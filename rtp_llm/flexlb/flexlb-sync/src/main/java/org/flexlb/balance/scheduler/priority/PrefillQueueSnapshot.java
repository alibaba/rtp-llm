package org.flexlb.balance.scheduler.priority;

import java.util.List;

/**
 * Strongly-consistent point-in-time view of one prefill batcher queue: it is
 * built entirely inside the batcher's {@code queueLock} critical section
 * (see {@code PrefillQueueManager.snapshot()}), so the mutation generation
 * and items always belong to the same queue state. Victim replacement later
 * validates only the selected victims under the same queue lock.
 *
 * @param endpointId    prefill endpoint key ("ip:httpPort")
 * @param queueVersion  queue mutation generation at capture time
 * @param queueCapacity active dispatcher hard queue limit; 0 = unbounded
 * @param items         queued requests in queue order (head first)
 */
public record PrefillQueueSnapshot(
        String endpointId,
        long queueVersion,
        int queueCapacity,
        List<QueuedRequestSnapshot> items) {
}
