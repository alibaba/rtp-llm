package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.Objects;
import java.util.PriorityQueue;

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
        public Iterator<ScheduledRequest> iterator() {
            return Collections.emptyIterator();
        }
    }

    final class Ordered implements PrefillActiveIndex {
        private final PriorityQueue<ScheduledRequest> queue;

        private Ordered(
                int initialCapacity,
                Comparator<ScheduledRequest> ordering) {
            queue = new PriorityQueue<>(
                    initialCapacity,
                    Objects.requireNonNull(ordering, "ordering"));
        }

        @Override
        public boolean add(ScheduledRequest item) {
            return queue.add(item);
        }

        @Override
        public boolean remove(ScheduledRequest item) {
            return queue.remove(item);
        }

        @Override
        public boolean contains(ScheduledRequest item) {
            return queue.contains(item);
        }

        @Override
        public ScheduledRequest peek() {
            return queue.peek();
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
        }

        @Override
        public Iterator<ScheduledRequest> iterator() {
            return queue.iterator();
        }
    }
}
