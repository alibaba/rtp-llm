package org.flexlb.balance.scheduler;

import java.util.Collections;
import java.util.List;

/**
 * Immutable snapshot of the batcher queue at a specific version.
 *
 * <p>Used by the versioned CAS API in {@link BatcherContext} to enable
 * atomic check-then-act operations (e.g. {@code tryRemove}, {@code tryOffer})
 * from external callers such as {@code PriorityAdmissionScheduler}.
 *
 * <p>The {@link #version()} field must be passed back to the CAS methods
 * as the expected version for optimistic concurrency control.
 */
public final class QueueSnapshot {

    private final long version;
    private final int queueSize;
    private final List<ItemSummary> items;

    public QueueSnapshot(long version, int queueSize, List<ItemSummary> items) {
        this.version = version;
        this.queueSize = queueSize;
        this.items = Collections.unmodifiableList(items);
    }

    /** Monotonic version at the time of snapshot. */
    public long version() {
        return version;
    }

    public int queueSize() {
        return queueSize;
    }

    /** Immutable list of item summaries, ordered by sort key. */
    public List<ItemSummary> items() {
        return items;
    }

    /**
     * Summary of a single queued item.
     */
    public static final class ItemSummary {
        private final long requestId;
        private final int priority;
        private final long deadlineMs;
        private final long seqLen;

        public ItemSummary(long requestId, int priority, long deadlineMs, long seqLen) {
            this.requestId = requestId;
            this.priority = priority;
            this.deadlineMs = deadlineMs;
            this.seqLen = seqLen;
        }

        public long requestId() { return requestId; }
        public int priority() { return priority; }
        public long deadlineMs() { return deadlineMs; }
        public long seqLen() { return seqLen; }
    }
}
