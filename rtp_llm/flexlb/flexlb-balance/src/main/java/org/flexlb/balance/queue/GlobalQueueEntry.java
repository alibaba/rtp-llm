package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.util.PriorityNormalizer;

import java.util.Comparator;

/**
 * One request identity retained by the model-wide ordered queue.
 */
public final class GlobalQueueEntry {
    static final Comparator<GlobalQueueEntry> SEQUENCE_ORDER =
            Comparator.comparingLong(entry -> entry.sequence);
    static final Comparator<GlobalQueueEntry> PRIORITY_ORDER =
            Comparator.comparingInt(GlobalQueueEntry::priority)
                    .reversed()
                    .thenComparing(SEQUENCE_ORDER);

    private final RequestContext context;
    private final String routingGroup;

    long sequence;
    volatile boolean removed = true;

    GlobalQueueEntry previous;
    GlobalQueueEntry next;

    public GlobalQueueEntry(RequestContext context, String routingGroup) {
        this(context, routingGroup, 0L);
    }

    public GlobalQueueEntry(RequestContext context, String routingGroup, long sequence) {
        this.context = context;
        this.routingGroup = routingGroup;
        this.sequence = sequence;
    }

    public RequestContext context() {
        return context;
    }

    public String routingGroup() {
        return routingGroup;
    }

    public long sequence() {
        return sequence;
    }

    public boolean isRemoved() {
        return removed;
    }

    public int priority() {
        int priority = context.getPriority();
        // Legacy internal callers can carry 0/invalid input; only the global queue defaults it.
        return PriorityNormalizer.isValid(priority)
                ? priority
                : PriorityNormalizer.DEFAULT_PRIORITY;
    }

    /**
     * Called only by the owning queue while holding its caller's synchronization boundary.
     */
    void markQueued(long sequence) {
        if (!removed || sequence <= 0) throw new IllegalStateException("invalid queue membership");
        this.sequence = sequence;
        removed = false;
    }

    void markRemoved() {
        removed = true;
    }
}
