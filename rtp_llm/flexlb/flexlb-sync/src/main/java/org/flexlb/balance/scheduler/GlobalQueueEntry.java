package org.flexlb.balance.scheduler;

import org.flexlb.util.PriorityNormalizer;

import java.util.Comparator;

/** One request identity retained by the model-wide ordered queue. */
final class GlobalQueueEntry {
    static final Comparator<GlobalQueueEntry> SEQUENCE_ORDER = Comparator.comparingLong(entry -> entry.sequence);
    static final Comparator<GlobalQueueEntry> PRIORITY_ORDER =
            Comparator.comparingInt(GlobalQueueEntry::priority).reversed().thenComparing(SEQUENCE_ORDER);

    final RequestContext context;
    final String routingGroup;
    long sequence;
    volatile boolean removed = true;
    GlobalQueueEntry previous;
    GlobalQueueEntry next;

    GlobalQueueEntry(RequestContext context, String routingGroup) {
        this.context = context;
        this.routingGroup = routingGroup;
    }

    int priority() {
        int priority = context.getPriority();
        // Legacy internal callers can carry 0/invalid input; only the global queue defaults it.
        return PriorityNormalizer.isValid(priority) ? priority : PriorityNormalizer.DEFAULT_PRIORITY;
    }
}
