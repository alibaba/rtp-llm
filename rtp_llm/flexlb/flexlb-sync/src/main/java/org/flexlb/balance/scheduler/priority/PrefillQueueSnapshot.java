package org.flexlb.balance.scheduler.priority;

import java.util.List;

/**
 * Strongly-consistent point-in-time view of one prefill batcher queue. The
 * mutation generation and the queue membership are captured atomically
 * inside the batcher's {@code queueLock} critical section (see
 * {@code PrefillQueueManager.snapshot()}); the O(n log n) ordering of that
 * frozen membership copy runs outside the lock because item ordering keys
 * are frozen once the item is constructed — "version unchanged ⇒ queue
 * content unchanged" holds, and the output order equals the in-lock
 * full-sort order bit-for-bit. Victim replacement later validates only the
 * selected victims under the same queue lock.
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
