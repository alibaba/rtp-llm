package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.planner.GroupPlanner;

import java.util.Collections;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.TreeSet;

/**
 * Active request identities for one Prefill generation.
 *
 * <p>The scheduler type selects one implementation at construction time.
 * DIRECT has no backing queue and cannot accept ACTIVE queue work; QUEUE owns
 * one ordered index shared by its runtime and canonical Prefill ledger.</p>
 */
public sealed interface PrefillActiveIndex extends Iterable<ScheduledRequest>
        permits PrefillActiveIndex.Disabled, PrefillActiveIndex.Ordered {

    static PrefillActiveIndex disabled() {
        return Disabled.INSTANCE;
    }

    static PrefillActiveIndex ordered(
            int initialCapacity,
            Comparator<ScheduledRequest> ordering) {
        return new Ordered(initialCapacity, ordering);
    }

    boolean add(ScheduledRequest item);

    boolean remove(ScheduledRequest item);

    boolean contains(ScheduledRequest item);

    ScheduledRequest peek();

    boolean isEmpty();

    int size();

    void clear();

    /** Caller holds the owning PrefillState lock. Reused until membership changes. */
    Capture capture();

    /** Immutable membership; prediction objects are materialized outside the owner lock. */
    final class Capture {
        private static final Capture EMPTY = new Capture(List.of());
        private final List<Entry> entries;
        private final List<ScheduledRequest> items;
        private volatile List<GroupPlanner.Item> projectedItems;

        private Capture(Collection<Entry> entries) {
            this.entries = List.copyOf(entries);
            List<ScheduledRequest> requests = new ArrayList<>(entries.size());
            for (Entry entry : entries) {
                requests.add(entry.request);
            }
            this.items = List.copyOf(requests);
        }

        public List<ScheduledRequest> items() {
            return items;
        }

        public List<GroupPlanner.Item> projectedItems() {
            List<GroupPlanner.Item> result = projectedItems;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (projectedItems == null) {
                    List<GroupPlanner.Item> resultItems = new ArrayList<>(entries.size());
                    for (Entry entry : entries) {
                        resultItems.add(entry.projectedItem());
                    }
                    projectedItems = List.copyOf(resultItems);
                }
                return projectedItems;
            }
        }
    }

    /** One identity in the active index; survives membership snapshot rebuilds. */
    final class Entry {
        private final ScheduledRequest request;
        private final long sequence;
        private volatile GroupPlanner.Item projectedItem;

        private Entry(ScheduledRequest request, long sequence) {
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

    final class Disabled implements PrefillActiveIndex {
        private static final Disabled INSTANCE = new Disabled();

        private Disabled() {
        }

        @Override
        public boolean add(ScheduledRequest item) {
            throw new IllegalStateException(
                    "DIRECT Prefill generation has no active request index");
        }

        @Override
        public boolean remove(ScheduledRequest item) {
            return false;
        }

        @Override
        public boolean contains(ScheduledRequest item) {
            return false;
        }

        @Override
        public ScheduledRequest peek() {
            return null;
        }

        @Override
        public boolean isEmpty() {
            return true;
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public void clear() {
        }

        @Override
        public Capture capture() {
            return Capture.EMPTY;
        }

        @Override
        public Iterator<ScheduledRequest> iterator() {
            return Collections.emptyIterator();
        }
    }

    final class Ordered implements PrefillActiveIndex {
        private final TreeSet<Entry> queue;
        // Identity lookup is part of this index, not a second ownership ledger.
        private final IdentityHashMap<ScheduledRequest, Entry> identities;
        private long nextSequence;
        private Capture capture;

        private Ordered(int initialCapacity, Comparator<ScheduledRequest> ordering) {
            Objects.requireNonNull(ordering, "ordering");
            identities = new IdentityHashMap<>(initialCapacity);
            queue = new TreeSet<>((left, right) -> {
                int compared = ordering.compare(left.request, right.request);
                // Preserve distinct identities even with an equal scheduling key.
                return compared != 0 ? compared : Long.compare(left.sequence, right.sequence);
            });
        }

        @Override
        public boolean add(ScheduledRequest item) {
            Objects.requireNonNull(item, "item");
            if (identities.containsKey(item)) {
                return false;
            }
            Entry entry = new Entry(item, nextSequence++);
            queue.add(entry);
            identities.put(item, entry);
            capture = null;
            return true;
        }

        @Override
        public boolean remove(ScheduledRequest item) {
            Entry entry = identities.get(item);
            if (entry == null) {
                return false;
            }
            queue.remove(entry);
            identities.remove(item);
            capture = null;
            return true;
        }

        @Override
        public boolean contains(ScheduledRequest item) {
            return identities.containsKey(item);
        }

        @Override
        public ScheduledRequest peek() {
            return queue.isEmpty() ? null : queue.first().request;
        }

        @Override
        public boolean isEmpty() {
            return queue.isEmpty();
        }

        @Override
        public int size() {
            return queue.size();
        }

        @Override
        public void clear() {
            queue.clear();
            identities.clear();
            capture = null;
        }

        @Override
        public Capture capture() {
            if (capture == null) {
                capture = queue.isEmpty() ? Capture.EMPTY : new Capture(queue);
            }
            return capture;
        }

        @Override
        public Iterator<ScheduledRequest> iterator() {
            Iterator<Entry> ordered = queue.iterator();
            return new Iterator<>() {
                @Override
                public boolean hasNext() {
                    return ordered.hasNext();
                }

                @Override
                public ScheduledRequest next() {
                    return ordered.next().request;
                }
            };
        }
    }
}
