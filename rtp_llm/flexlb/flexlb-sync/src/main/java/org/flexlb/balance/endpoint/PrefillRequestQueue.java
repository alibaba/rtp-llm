package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;

import java.util.Comparator;
import java.util.List;

/** Exact identities in frozen queue order; all operations use the owning PrefillState lock. */
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

    /** Membership and input values remain stable after the owner releases its lock. */
    interface Snapshot {
        List<RequestRoute> items();
    }

    @FunctionalInterface
    interface Factory {
        PrefillRequestQueue create(Comparator<RequestRoute> order);
    }
}
