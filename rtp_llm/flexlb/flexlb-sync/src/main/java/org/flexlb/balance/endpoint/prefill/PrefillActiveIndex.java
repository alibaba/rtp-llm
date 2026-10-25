package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.util.PriorityNormalizer;

import java.util.Arrays;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * Active request identities for one Prefill generation.
 *
 * <p>One ordered index is shared by the worker queue runtime and the canonical
 * Prefill ledger.</p>
 */
public sealed interface PrefillActiveIndex extends Iterable<ScheduledRequest> permits PrefillActiveIndex.Ordered {

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

    int size(int priority);

    void clear();

    /** Caller holds the owning PrefillState lock. Reused until membership changes. */
    WaitingQueueView capture();

    /** One identity in the active index; survives membership snapshot rebuilds. */
    final class Entry {
        public final ScheduledRequest request;
        private final long sequence;
        private volatile GroupPlanner.Item plannerItem;

        private Entry(ScheduledRequest request, long sequence) {
            this.request = request;
            this.sequence = sequence;
        }

        public GroupPlanner.Item plannerItem() {
            GroupPlanner.Item result = plannerItem;
            if (result != null) {
                return result;
            }
            synchronized (this) {
                if (plannerItem == null) {
                    plannerItem = new GroupPlanner.Item(
                            request.requestId(), request.priority(), request.enqueueSeq(),
                            request.enqueuedAtMs(), request.expiresAtMs(), request.seqLen(),
                            request.hitCache());
                }
                return plannerItem;
            }
        }
    }

    final class Ordered implements PrefillActiveIndex {
        private final TreeSet<Entry> queue;
        // Identity lookup is part of this index, not a second ownership ledger.
        private final IdentityHashMap<ScheduledRequest, Entry> identities;
        private final int[] priorityCounts = new int[PriorityNormalizer.MAX_PRIORITY + 1];
        private long nextSequence;
        private WaitingQueueView capture;

        private Ordered(int initialCapacity, Comparator<ScheduledRequest> ordering) {
            identities = new IdentityHashMap<>(initialCapacity);
            queue = new TreeSet<>((left, right) -> {
                int compared = ordering.compare(left.request, right.request);
                // Preserve distinct identities even with an equal scheduling key.
                return compared != 0 ? compared : Long.compare(left.sequence, right.sequence);
            });
        }

        @Override
        public boolean add(ScheduledRequest item) {
            if (identities.containsKey(item)) {
                return false;
            }
            Entry entry = new Entry(item, nextSequence++);
            queue.add(entry);
            identities.put(item, entry);
            priorityCounts[item.priority()]++;
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
            priorityCounts[item.priority()]--;
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
        public int size(int priority) { return priorityCounts[priority]; }

        @Override
        public void clear() {
            Arrays.fill(priorityCounts, 0);
            queue.clear();
            identities.clear();
            capture = null;
        }

        @Override
        public WaitingQueueView capture() {
            if (capture == null) {
                capture = queue.isEmpty() ? WaitingQueueView.EMPTY : new WaitingQueueView(queue);
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
