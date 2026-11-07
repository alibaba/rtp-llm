package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteExecution;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Request identities in frozen queue order; all operations run in the PrefillEndpoint state lock.
 */
public interface PrefillRequestQueue extends Iterable<RouteExecution> {
    boolean enabled();

    long version();

    boolean add(RouteExecution route);

    boolean remove(RouteExecution route);

    boolean contains(RouteExecution route);

    RouteExecution peek();

    boolean isEmpty();

    int size();

    int size(int priority);

    void clear();

    Snapshot capture();

    /**
     * A prefix after a blocked head, or empty to preserve strict queue order. No mutation.
     */
    default List<RouteExecution> candidatesAfterBlockedHead(
            List<RouteExecution> ordered, boolean priorityOrdering, long nowMs) {
        return List.of();
    }

    /**
     * Called in the Endpoint lock only after a successful dispatch commit.
     */
    default void recordDispatch(RouteExecution skippedHead) {}

    /**
     * Fields available to both live queue ordering and frozen prediction inputs.
     */
    interface QueueItem {
        String requestId();

        int priority();

        long enqueueSeq();

        long enqueuedAtMs();

        long expiresAtMs();

        long seqLen();

        long hitCache();
    }

    /**
     * The comparator must depend only on frozen QueueItem fields and remain stable.
     */
    record Snapshot(List<RouteExecution> items, Comparator<QueueItem> order) {
        public Snapshot {
            items = List.copyOf(items);
            Objects.requireNonNull(order, "order");
        }
    }

    @FunctionalInterface
    interface Factory {
        PrefillRequestQueue create(Comparator<QueueItem> order);
    }
}
