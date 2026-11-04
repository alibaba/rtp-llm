package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/** Exact identities in frozen queue order; all operations use the owning PrefillEndpoint lock. */
public interface PrefillRequestQueue extends Iterable<RequestRoute> {
    boolean enabled();

    long version();

    boolean add(RequestRoute route);

    boolean remove(RequestRoute route);

    boolean contains(RequestRoute route);

    RequestRoute peek();

    boolean isEmpty();

    int size();

    int size(int priority);

    void clear();

    Snapshot capture();

    /** A prefix after a blocked head, or empty to preserve strict queue order. No mutation. */
    default List<RequestRoute> candidatesAfterBlockedHead(
            List<RequestRoute> ordered, boolean priorityOrdering, long nowMs) {
        return List.of();
    }

    /** Called under the endpoint lock only after a successful dispatch commit. */
    default void recordDispatch(RequestRoute skippedHead) {}

    /** Fields available to both live queue ordering and frozen prediction inputs. */
    interface QueueItem {
        long requestId();

        int priority();

        long enqueueSeq();

        long enqueuedAtMs();

        long expiresAtMs();

        long seqLen();

        long hitCache();
    }

    /** The comparator must depend only on frozen QueueItem fields and remain stable. */
    record Snapshot(List<RequestRoute> items, Comparator<QueueItem> order) {
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
