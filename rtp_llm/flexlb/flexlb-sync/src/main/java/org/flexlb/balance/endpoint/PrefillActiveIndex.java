package org.flexlb.balance.endpoint;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.util.PriorityNormalizer;

import java.util.Arrays;
import java.util.Collection;
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
public final class PrefillActiveIndex implements Iterable<RequestRoute> {
    private static final PrefillActiveIndex DISABLED = new PrefillActiveIndex();

    private final TreeSet<Entry> queue;
    // Identity lookup is part of this index, not a second ownership ledger.
    private final IdentityHashMap<RequestRoute, Entry> identities;
    private final int[] priorityCounts;
    private long nextSequence;
    private volatile long version;
    private Capture capture;

    public static PrefillActiveIndex disabled() {
        return DISABLED;
    }

    public static PrefillActiveIndex ordered(int initialCapacity, Comparator<RequestRoute> ordering) {
        return new PrefillActiveIndex(initialCapacity, ordering);
    }

    private PrefillActiveIndex() {
        queue = null;
        identities = null;
        priorityCounts = null;
    }

    /** Immutable membership; prediction objects are materialized outside the owner lock. */
    public static final class Capture {
        private static final Capture EMPTY = new Capture(List.of());
        private final List<Entry> entries;
        private volatile List<GroupPlanner.Item> projectedItems;

        private Capture(Collection<Entry> entries) {
            this.entries = List.copyOf(entries);
        }

        public boolean isEmpty() {
            return entries.isEmpty();
        }

        public List<GroupPlanner.Item> projectedItems() {
            List<GroupPlanner.Item> result = projectedItems;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (projectedItems == null) {
                    projectedItems = List.of(entries.stream().map(Entry::projectedItem)
                            .toArray(GroupPlanner.Item[]::new));
                }
                return projectedItems;
            }
        }
    }

    /** One identity in the active index; survives membership snapshot rebuilds. */
    private static final class Entry {
        private final RequestRoute request;
        private final long sequence;
        private volatile GroupPlanner.Item projectedItem;

        private Entry(RequestRoute request, long sequence) {
            this.request = request;
            this.sequence = sequence;
        }

        private GroupPlanner.Item projectedItem() {
            GroupPlanner.Item result = projectedItem;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (projectedItem == null) {
                    projectedItem = new GroupPlanner.Item(
                            request.requestId(), request.priority(), request.enqueueSeq(),
                            request.enqueuedAtMs(), request.expiresAtMs(), request.seqLen(),
                            request.hitCache());
                }
                return projectedItem;
            }
        }
    }

    private PrefillActiveIndex(int initialCapacity, Comparator<RequestRoute> ordering) {
        Objects.requireNonNull(ordering, "ordering");
        priorityCounts = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        identities = new IdentityHashMap<>(initialCapacity);
        queue = new TreeSet<>((left, right) -> {
            int compared = ordering.compare(left.request, right.request);
            // Preserve distinct identities even with an equal scheduling key.
            return compared != 0 ? compared : Long.compare(left.sequence, right.sequence);
        });
    }

    /** Membership revision; mutations hold the owning ledger lock. */
    public long version() { return version; }

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

    public int size(int priority) { return priorityCounts == null ? 0 : priorityCounts[priority]; }

    public void clear() {
        if (isEmpty()) { return; }
        Arrays.fill(priorityCounts, 0);
        queue.clear();
        identities.clear();
        capture = null;
        version++;
    }

    /** Caller holds the owning ledger lock; reuse membership until it changes. */
    public Capture capture() {
        if (queue == null) { return Capture.EMPTY; }
        if (capture == null) {
            capture = queue.isEmpty() ? Capture.EMPTY : new Capture(queue);
        }
        return capture;
    }

    @Override
    public Iterator<RequestRoute> iterator() {
        if (queue == null) { return Collections.emptyIterator(); }
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
