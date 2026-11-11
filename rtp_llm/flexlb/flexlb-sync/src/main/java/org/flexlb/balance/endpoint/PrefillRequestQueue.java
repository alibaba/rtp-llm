package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.util.PriorityNormalizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkState;

/**
 * Stores one Prefill Endpoint's requests in FIFO order; removing a request preserves the remaining order.
 * Operations run under the Endpoint state lock. Disabled queues have no storage and reject additions.
 */
public final class PrefillRequestQueue implements Iterable<RouteResult> {
    private static final PrefillRequestQueue DISABLED = new PrefillRequestQueue();
    private static final Snapshot EMPTY = new Snapshot(List.of());

    private final LinkedHashSet<Entry> queue;
    // Identity lookup is part of this index, not a second ownership ledger.
    private final IdentityHashMap<RouteResult, Entry> identities;
    private final int[] priorityCounts;

    private volatile long version;

    private Snapshot capture;

    private PrefillRequestQueue() {
        queue = null;
        identities = null;
        priorityCounts = null;
    }

    private PrefillRequestQueue(int initialCapacity) {
        priorityCounts = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        identities = new IdentityHashMap<>(initialCapacity);
        queue = new LinkedHashSet<>(initialCapacity);
    }

    public static PrefillRequestQueue disabled() {
        return DISABLED;
    }

    public static PrefillRequestQueue fifo(int initialCapacity) {
        return new PrefillRequestQueue(initialCapacity);
    }

    /** Membership revision; mutations hold the owning ledger lock. */
    public long version() {
        return version;
    }

    public boolean enabled() {
        return queue != null;
    }

    public boolean add(RouteResult item) {
        checkState(queue != null, "DIRECT Prefill generation has no request queue");
        Objects.requireNonNull(item, "item");
        if (identities.containsKey(item)) {
            return false;
        }
        int priority = Objects.checkIndex(item.priority(), priorityCounts.length);
        Entry entry = new Entry(item);
        queue.add(entry);
        identities.put(item, entry);
        priorityCounts[priority]++;
        capture = null;
        version++;
        return true;
    }

    public RouteResult peek() {
        return isEmpty() ? null : queue.getFirst().request;
    }

    public boolean isEmpty() {
        return queue == null || queue.isEmpty();
    }

    public boolean remove(RouteResult item) {
        Entry entry = identities == null ? null : identities.get(item);
        if (entry == null) {
            return false;
        }
        queue.remove(entry);
        identities.remove(item);
        priorityCounts[item.priority()]--;
        capture = null;
        version++;
        return true;
    }

    public boolean contains(RouteResult item) {
        return identities != null && identities.containsKey(item);
    }

    public int size() {
        return queue == null ? 0 : queue.size();
    }

    public int size(int priority) {
        return priorityCounts == null ? 0 : priorityCounts[priority];
    }

    public void clear() {
        if (isEmpty()) {
            return;
        }
        Arrays.fill(priorityCounts, 0);
        queue.clear();
        identities.clear();
        capture = null;
        version++;
    }

    /** Caller holds the owning ledger lock; reuse membership until it changes. */
    public Snapshot capture() {
        if (queue == null) {
            return EMPTY;
        }
        if (capture == null) {
            capture = new Snapshot(queue.stream().map(entry -> entry.request).toList());
        }
        return capture;
    }

    @Override
    public Iterator<RouteResult> iterator() {
        if (queue == null) {
            return Collections.emptyIterator();
        }
        Iterator<Entry> ordered = queue.iterator();
        return new Iterator<>() {
            @Override
            public boolean hasNext() {
                return ordered.hasNext();
            }

            @Override
            public RouteResult next() {
                return ordered.next().request;
            }
        };
    }

    /**
     * Keeps an exact route identity in the insertion-ordered queue.
     */
    private static final class Entry {
        private final RouteResult request;

        private Entry(RouteResult request) {
            this.request = request;
        }
    }

    /**
     * Fields available to both live queue ordering and frozen prediction inputs.
     */
    public interface QueueItem {
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
    public record Snapshot(List<RouteResult> items) {
        public Snapshot {
            items = List.copyOf(items);
        }
    }
}
