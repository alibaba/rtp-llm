package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteResult;

import java.util.List;

/**
 * Request identities in acceptance order; all operations run in the PrefillEndpoint state lock.
 * Implementations only change queue storage and must not invoke request lifecycle callbacks.
 */
public interface PrefillRequestQueue extends Iterable<RouteResult> {
    boolean enabled();

    long version();

    /**
     * Append an exact member without changing existing order. False or an exception leaves membership and its revision unchanged.
     * Implementations must finish validation before publishing the member; notifications belong
     * to the caller after this operation returns.
     */
    boolean add(RouteResult route);

    boolean remove(RouteResult route);

    boolean contains(RouteResult route);

    RouteResult peek();

    boolean isEmpty();

    int size();

    int size(int priority);

    void clear();

    Snapshot capture();

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
     * Captures accepted order; later queue mutations do not change these members.
     */
    record Snapshot(List<RouteResult> items) {
        public Snapshot {
            items = List.copyOf(items);
        }
    }

    /**
     * Creates queue storage owned and synchronized by one Prefill endpoint.
     */
    @FunctionalInterface
    interface Factory {
        PrefillRequestQueue create();
    }
}
