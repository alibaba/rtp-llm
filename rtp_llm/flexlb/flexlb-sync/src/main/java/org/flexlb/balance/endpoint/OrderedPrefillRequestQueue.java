package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteExecution;
import org.flexlb.util.PriorityNormalizer;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.TreeSet;

import static com.google.common.base.Preconditions.checkState;

/**
 * Ordered request queue for one Prefill generation.
 *
 * <p>The scheduler type selects queue storage at construction time.
 * DIRECT has no backing queue and cannot accept ACTIVE queue work; QUEUE owns
 * one ordered queue shared by its runtime and canonical Prefill ledger.</p>
 */
public final class OrderedPrefillRequestQueue implements PrefillRequestQueue {
    private static final OrderedPrefillRequestQueue DISABLED = new OrderedPrefillRequestQueue();

    private final TreeSet<Entry> queue;
    // Identity lookup is part of this index, not a second ownership ledger.
    private final IdentityHashMap<RouteExecution, Entry> identities;
    private final int[] priorityCounts;
    private long nextSequence;
    private RouteExecution bypassedHead;
    private volatile long version;
    private Snapshot capture;
    private final Comparator<QueueItem> ordering;
    private static final Snapshot EMPTY = new Snapshot(List.of(), (left, right) -> 0);

    public static OrderedPrefillRequestQueue disabled() {
        return DISABLED;
    }

    public static OrderedPrefillRequestQueue ordered(
            int initialCapacity, Comparator<QueueItem> ordering) {
        return new OrderedPrefillRequestQueue(initialCapacity, ordering);
    }

    private OrderedPrefillRequestQueue() {
        queue = null;
        identities = null;
        priorityCounts = null;
        ordering = null;
    }

    private static final class Entry {
        private final RouteExecution request;
        private final long sequence;

        private Entry(RouteExecution request, long sequence) {
            this.request = request;
            this.sequence = sequence;
        }
    }

    private OrderedPrefillRequestQueue(int initialCapacity, Comparator<QueueItem> ordering) {
        this.ordering = Objects.requireNonNull(ordering, "ordering");
        priorityCounts = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        identities = new IdentityHashMap<>(initialCapacity);
        queue =
                new TreeSet<>(
                        (left, right) -> {
                            int compared = ordering.compare(left.request, right.request);
                            // Preserve distinct identities even with an equal scheduling key.
                            return compared != 0
                                    ? compared
                                    : Long.compare(left.sequence, right.sequence);
                        });
    }

    /** Membership revision; mutations hold the owning ledger lock. */
    public long version() {
        return version;
    }

    public boolean enabled() {
        return queue != null;
    }

    public boolean add(RouteExecution item) {
        checkState(queue != null, "DIRECT Prefill generation has no request queue");
        Objects.requireNonNull(item, "item");
        if (identities.containsKey(item)) {
            return false;
        }
        Entry entry = new Entry(item, nextSequence++);
        queue.add(entry);
        identities.put(item, entry);
        priorityCounts[item.priority()]++;
        if (bypassedHead != null && bypassedHead != peek()) bypassedHead = null;
        capture = null;
        version++;
        return true;
    }

    public boolean remove(RouteExecution item) {
        Entry entry = identities == null ? null : identities.get(item);
        if (entry == null) {
            return false;
        }
        queue.remove(entry);
        identities.remove(item);
        priorityCounts[item.priority()]--;
        if (bypassedHead != null && bypassedHead != peek()) bypassedHead = null;
        capture = null;
        version++;
        return true;
    }

    public boolean contains(RouteExecution item) {
        return identities != null && identities.containsKey(item);
    }

    public RouteExecution peek() {
        return isEmpty() ? null : queue.first().request;
    }

    public boolean isEmpty() {
        return queue == null || queue.isEmpty();
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
        bypassedHead = null;
        capture = null;
        version++;
    }

    /** Caller holds the owning ledger lock; reuse membership until it changes. */
    public Snapshot capture() {
        if (queue == null) {
            return EMPTY;
        }
        if (capture == null) {
            capture = new Snapshot(queue.stream().map(entry -> entry.request).toList(), ordering);
        }
        return capture;
    }

    @Override
    public List<RouteExecution> candidatesAfterBlockedHead(
            List<RouteExecution> ordered, boolean priorityOrdering, long nowMs) {
        if (!priorityOrdering || ordered.isEmpty()) return List.of();
        var head = ordered.getFirst();
        if (head != peek() || head == bypassedHead || head.requestExpired(nowMs)) return List.of();
        int end = 1;
        while (end < ordered.size()) {
            var candidate = ordered.get(end);
            if (head.priority() != candidate.priority()
                    || candidate.requestExpired(nowMs)
                    || head.decodeEndpoint() == null
                    || candidate.decodeEndpoint() == null
                    || head.decodeEndpoint() == candidate.decodeEndpoint()) break;
            end++;
        }
        return ordered.subList(1, end);
    }

    @Override
    public void recordDispatch(RouteExecution skippedHead) {
        bypassedHead = skippedHead;
    }

    @Override
    public Iterator<RouteExecution> iterator() {
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
            public RouteExecution next() {
                return ordered.next().request;
            }
        };
    }
}
