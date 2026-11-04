package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;
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
 * Active request identities for one Prefill generation.
 *
 * <p>The scheduler type selects queue storage at construction time.
 * DIRECT has no backing queue and cannot accept ACTIVE queue work; QUEUE owns
 * one ordered index shared by its runtime and canonical Prefill ledger.</p>
 */
public final class PrefillActiveIndex implements PrefillRequestQueue {
    private static final PrefillActiveIndex DISABLED = new PrefillActiveIndex();

    private final TreeSet<Entry> queue;
    // Identity lookup is part of this index, not a second ownership ledger.
    private final IdentityHashMap<RequestRoute, Entry> identities;
    private final int[] priorityCounts;
    private long nextSequence;
    private RequestRoute bypassedHead;
    private volatile long version;
    private Snapshot capture;
    private final Comparator<QueueItem> ordering;
    private static final Snapshot EMPTY = new Snapshot(List.of(), (left, right) -> 0);

    public static PrefillActiveIndex disabled() {
        return DISABLED;
    }

    public static PrefillActiveIndex ordered(int initialCapacity, Comparator<QueueItem> ordering) {
        return new PrefillActiveIndex(initialCapacity, ordering);
    }

    private PrefillActiveIndex() {
        queue = null;
        identities = null;
        priorityCounts = null;
        ordering = null;
    }

    private static final class Entry {
        private final RequestRoute request;
        private final long sequence;

        private Entry(RequestRoute request, long sequence) {
            this.request = request;
            this.sequence = sequence;
        }
    }

    private PrefillActiveIndex(int initialCapacity, Comparator<QueueItem> ordering) {
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

    public boolean add(RequestRoute item) {
        checkState(queue != null, "DIRECT Prefill generation has no active request index");
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

    public boolean remove(RequestRoute item) {
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

    public boolean contains(RequestRoute item) {
        return identities != null && identities.containsKey(item);
    }

    public RequestRoute peek() {
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
    public List<RequestRoute> candidatesAfterBlockedHead(
            List<RequestRoute> ordered, boolean priorityOrdering, long nowMs) {
        if (!priorityOrdering || ordered.isEmpty()) return List.of();
        var head = ordered.getFirst();
        if (head != peek() || head == bypassedHead || head.requestExpired(nowMs)) return List.of();
        int end = 1;
        while (end < ordered.size()) {
            var candidate = ordered.get(end);
            if (head.priority() != candidate.priority()
                    || candidate.requestExpired(nowMs)
                    || head.decodeEp() == null
                    || candidate.decodeEp() == null
                    || head.decodeEp() == candidate.decodeEp()) break;
            end++;
        }
        return ordered.subList(1, end);
    }

    @Override
    public void recordDispatch(RequestRoute skippedHead) {
        bypassedHead = skippedHead;
    }

    @Override
    public Iterator<RequestRoute> iterator() {
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
            public RequestRoute next() {
                return ordered.next().request;
            }
        };
    }
}
