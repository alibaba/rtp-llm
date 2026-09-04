package org.flexlb.balance.scheduler;

import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.function.Predicate;

/**
 * FIFO or fixed-range PRIORITY index for the model queue.
 *
 * <p>The coordinator lock is the sole synchronization boundary. Entries are
 * intrusive nodes, so completion and cancellation unlink an arbitrary request
 * in O(1) without leaving a tombstone behind a blocked head.</p>
 */
final class OrderedRequestQueue {

    private static final int PRIORITY_LEVELS =
            PriorityNormalizer.MAX_PRIORITY + 1;

    private final boolean priorityOrdering;
    private final Bucket fifo = new Bucket();
    private final Bucket[] priorityBuckets = new Bucket[PRIORITY_LEVELS];
    private final BitSet nonEmptyPriorities = new BitSet(PRIORITY_LEVELS);
    private int size;

    OrderedRequestQueue(boolean priorityOrdering) {
        this.priorityOrdering = priorityOrdering;
    }

    void add(GlobalQueueEntry entry) {
        if (priorityOrdering) {
            Bucket bucket =
                    priorityBuckets[entry.priority];
            if (bucket == null) {
                bucket = new Bucket();
                priorityBuckets[entry.priority] = bucket;
            }
            bucket.add(entry);
            nonEmptyPriorities.set(entry.priority);
        } else {
            fifo.add(entry);
        }
        size++;
    }

    int size() {
        return size;
    }

    GlobalQueueEntry peekHead() {
        if (!priorityOrdering) {
            return fifo.head;
        }
        int priority = nonEmptyPriorities.previousSetBit(
                PRIORITY_LEVELS - 1);
        Bucket bucket = priority < 0
                ? null : priorityBuckets[priority];
        return bucket == null ? null : bucket.head;
    }

    List<GlobalQueueEntry> snapshotPrefix(
            int limit,
            Predicate<GlobalQueueEntry> eligible,
            GlobalQueueEntry frontier) {
        if (limit <= 0) {
            return List.of();
        }
        List<GlobalQueueEntry> result = new ArrayList<>(
                Math.min(limit, size));
        if (priorityOrdering) {
            for (int priority = nonEmptyPriorities.previousSetBit(
                            PRIORITY_LEVELS - 1);
                    priority >= 0 && result.size() < limit;
                    priority = nonEmptyPriorities.previousSetBit(priority - 1)) {
                Bucket bucket = priorityBuckets[priority];
                if (bucket == null) {
                    continue;
                }
                if (!appendEligible(bucket, result, limit, eligible, frontier)) {
                    return result;
                }
            }
        } else {
            appendEligible(fifo, result, limit, eligible, frontier);
        }
        return result;
    }

    boolean hasHigherPriorityEntry(
            GlobalQueueEntry entry,
            Predicate<GlobalQueueEntry> eligible) {
        if (!priorityOrdering) {
            return false;
        }
        for (int priority = nonEmptyPriorities.previousSetBit(
                        PRIORITY_LEVELS - 1);
                priority > entry.priority;
                priority = nonEmptyPriorities.previousSetBit(priority - 1)) {
            Bucket bucket = priorityBuckets[priority];
            if (bucket != null) {
                for (GlobalQueueEntry candidate = bucket.head;
                        candidate != null;
                        candidate = candidate.next) {
                    if (eligible.test(candidate)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    boolean remove(GlobalQueueEntry entry) {
        return markRemoved(entry);
    }

    boolean markCompleted(GlobalQueueEntry entry) {
        return markRemoved(entry);
    }

    void pruneCompletedHeads() {
        if (priorityOrdering) {
            while (!nonEmptyPriorities.isEmpty()) {
                int priority = nonEmptyPriorities.previousSetBit(
                        PRIORITY_LEVELS - 1);
                Bucket bucket = priorityBuckets[priority];
                pruneBucket(bucket);
                if (bucket == null || bucket.isEmpty()) {
                    nonEmptyPriorities.clear(priority);
                } else {
                    break;
                }
            }
        } else {
            pruneBucket(fifo);
        }
    }

    List<GlobalQueueEntry> drain() {
        List<GlobalQueueEntry> entries = new ArrayList<>();
        if (priorityOrdering) {
            for (Bucket bucket : priorityBuckets) {
                if (bucket != null) {
                    drainBucket(bucket, entries);
                }
            }
            nonEmptyPriorities.clear();
        } else {
            drainBucket(fifo, entries);
        }
        if (size != 0) {
            throw new IllegalStateException(
                    "ordered queue drain left active entries");
        }
        return entries;
    }

    private boolean markRemoved(GlobalQueueEntry entry) {
        if (entry.removed) {
            return false;
        }
        if (size <= 0) {
            throw new IllegalStateException("ordered queue size underflow");
        }
        entry.removed = true;
        unlink(entry);
        size--;
        return true;
    }

    private void pruneBucket(Bucket bucket) {
        while (bucket != null && !bucket.isEmpty()) {
            GlobalQueueEntry head = bucket.head;
            if (!head.future.isDone()) {
                return;
            }
            markRemoved(head);
        }
    }

    private void unlink(GlobalQueueEntry entry) {
        if (!entry.linked) {
            throw new IllegalStateException(
                    "ordered queue entry is not linked");
        }
        Bucket bucket = priorityOrdering
                ? priorityBuckets[entry.priority] : fifo;
        if (bucket == null) {
            throw new IllegalStateException(
                    "ordered queue bucket is missing");
        }
        bucket.remove(entry);
        if (priorityOrdering && bucket.isEmpty()) {
            nonEmptyPriorities.clear(entry.priority);
        }
    }

    private void drainBucket(
            Bucket bucket,
            List<GlobalQueueEntry> entries) {
        while (!bucket.isEmpty()) {
            GlobalQueueEntry entry = bucket.head;
            markRemoved(entry);
            entries.add(entry);
        }
    }

    private static boolean appendEligible(
            Bucket source,
            List<GlobalQueueEntry> result,
            int limit,
            Predicate<GlobalQueueEntry> eligible,
            GlobalQueueEntry frontier) {
        for (GlobalQueueEntry entry = source.head;
                entry != null;
                entry = entry.next) {
            if (entry == frontier) {
                return false;
            }
            if (eligible.test(entry)) {
                result.add(entry);
                if (result.size() == limit) {
                    break;
                }
            }
        }
        return true;
    }

    private static final class Bucket {
        private GlobalQueueEntry head;
        private GlobalQueueEntry tail;

        void add(GlobalQueueEntry entry) {
            if (entry.linked || entry.previous != null || entry.next != null) {
                throw new IllegalStateException(
                        "ordered queue entry is already linked");
            }
            entry.previous = tail;
            if (tail == null) {
                head = entry;
            } else {
                tail.next = entry;
            }
            tail = entry;
            entry.linked = true;
        }

        void remove(GlobalQueueEntry entry) {
            GlobalQueueEntry previous = entry.previous;
            GlobalQueueEntry next = entry.next;
            if (previous == null) {
                if (head != entry) {
                    throw new IllegalStateException(
                            "ordered queue head linkage is inconsistent");
                }
                head = next;
            } else {
                previous.next = next;
            }
            if (next == null) {
                if (tail != entry) {
                    throw new IllegalStateException(
                            "ordered queue tail linkage is inconsistent");
                }
                tail = previous;
            } else {
                next.previous = previous;
            }
            entry.previous = null;
            entry.next = null;
            entry.linked = false;
        }

        boolean isEmpty() {
            return head == null;
        }
    }
}
