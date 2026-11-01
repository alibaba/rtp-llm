package org.flexlb.balance.endpoint;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.RequestRoute;

import java.util.Comparator;
import java.util.List;

/** Exact identities in frozen queue order; all operations use the owning PrefillState lock. */
public interface WorkerQueue extends Iterable<RequestRoute> {
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

    /** Whether a ready request may pass a capacity-blocked head; the state still checks ownership. */
    default boolean mayBypass(RequestRoute head, RequestRoute candidate) {
        return false;
    }

    /** Membership and input values remain stable after the owner releases its lock. */
    interface Snapshot {
        List<GroupPlanner.Item> projectedItems();
    }

    @FunctionalInterface
    interface Factory {
        WorkerQueue create(Comparator<RequestRoute> order);
    }
}
