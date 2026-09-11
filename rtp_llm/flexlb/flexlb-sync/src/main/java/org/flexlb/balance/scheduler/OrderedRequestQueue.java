package org.flexlb.balance.scheduler;

import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.NavigableSet;
import java.util.List;
import java.util.TreeSet;
import java.util.function.Predicate;

/**
 * FIFO or fixed-range PRIORITY index for the model queue.
 *
 * <p>The coordinator lock is the sole synchronization boundary. Entries are
 * intrusive nodes, so completion and cancellation unlink an arbitrary request
 * in O(1) without leaving a tombstone behind a blocked head. Ready retries use
 * a separate ordered index and share the scan budget with forward progress.</p>
 */
final class OrderedRequestQueue {

    private static final int PRIORITY_LEVELS =
            PriorityNormalizer.MAX_PRIORITY + 1;

    private static final Comparator<GlobalQueueEntry> SEQUENCE_ORDER =
            Comparator.comparingLong(entry -> entry.sequence);
    private static final Comparator<GlobalQueueEntry> PRIORITY_ORDER =
            Comparator.<GlobalQueueEntry>comparingInt(entry -> entry.priority).reversed()
                    .thenComparing(SEQUENCE_ORDER);

    private final boolean priorityOrdering;
    private final Bucket fifo = new Bucket();
    private final Bucket[] priorityBuckets = new Bucket[PRIORITY_LEVELS];
    private final BitSet nonEmptyPriorities = new BitSet(PRIORITY_LEVELS);
    private final BitSet pendingPriorities = new BitSet(PRIORITY_LEVELS);
    private int size;
    private long nextSequence;

    OrderedRequestQueue(boolean priorityOrdering) {
        this.priorityOrdering = priorityOrdering;
    }

    void add(GlobalQueueEntry entry) {
        entry.sequence = ++nextSequence;
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
        rewindScanTo(entry);
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

    /** Continue an ordered scan, counting every examined entry against the budget. */
    List<GlobalQueueEntry> scanForPlanningCandidates(
            int candidateLimit, int scanBudget, Predicate<GlobalQueueEntry> eligible) {
        if (candidateLimit <= 0 || scanBudget <= 0) {
            return List.of();
        }
        List<GlobalQueueEntry> result = new ArrayList<>(Math.min(candidateLimit, size));
        for (int checked = 0; checked < scanBudget && result.size() < candidateLimit; checked++) {
            int priority = priorityOrdering
                    ? pendingPriorities.previousSetBit(PRIORITY_LEVELS - 1) : -1;
            Bucket bucket = priorityOrdering
                    ? (priority < 0 ? null : priorityBuckets[priority]) : fifo;
            if (bucket == null) {
                break;
            }
            GlobalQueueEntry entry = bucket.pollNextRequest();
            if (entry == null) {
                break;
            }
            if (priorityOrdering && !bucket.hasPendingRequests()) {
                pendingPriorities.clear(priority);
            }
            if (!result.contains(entry) && eligible.test(entry)) {
                result.add(entry);
            }
        }
        result.sort(priorityOrdering ? PRIORITY_ORDER : SEQUENCE_ORDER);
        return result;
    }

    boolean hasUnscannedRequests() {
        return priorityOrdering ? !pendingPriorities.isEmpty()
                : fifo.hasPendingRequests();
    }

    /** Start scanning at this request, or keep an existing earlier scan position. */
    private void rewindScanTo(GlobalQueueEntry entry) {
        if (!entry.linked || entry.removed) {
            return;
        }
        Bucket bucket = priorityOrdering ? priorityBuckets[entry.priority] : fifo;
        if (bucket.nextToScan == null || entry.sequence < bucket.nextToScan.sequence) {
            bucket.nextToScan = entry;
        }
        if (priorityOrdering) {
            pendingPriorities.set(entry.priority);
        }
    }

    /** Make an awakened request directly available to the next bounded scan. */
    void markRequestReadyForRetry(GlobalQueueEntry entry) {
        if (!entry.linked || entry.removed) {
            return;
        }
        Bucket bucket = priorityOrdering ? priorityBuckets[entry.priority] : fifo;
        bucket.readyRetries.add(entry);
        if (priorityOrdering) {
            pendingPriorities.set(entry.priority);
        }
    }

    boolean hasEarlierRequestsToScan(GlobalQueueEntry entry) {
        int priority = priorityOrdering
                ? pendingPriorities.previousSetBit(PRIORITY_LEVELS - 1) : -1;
        Bucket bucket = priorityOrdering
                ? (priority < 0 ? null : priorityBuckets[priority]) : fifo;
        GlobalQueueEntry next = bucket == null ? null : bucket.peekNextRequest();
        return next != null
                && (priorityOrdering ? PRIORITY_ORDER : SEQUENCE_ORDER).compare(next, entry) < 0;
    }

    boolean remove(GlobalQueueEntry entry) {
        return markRemoved(entry);
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
        if (priorityOrdering && !bucket.hasPendingRequests()) {
            pendingPriorities.clear(entry.priority);
        }
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

    private static final class Bucket {
        private GlobalQueueEntry head;
        private GlobalQueueEntry tail;
        private GlobalQueueEntry nextToScan;
        private final NavigableSet<GlobalQueueEntry> readyRetries =
                new TreeSet<>(SEQUENCE_ORDER);

        boolean hasPendingRequests() {
            return nextToScan != null || !readyRetries.isEmpty();
        }

        GlobalQueueEntry peekNextRequest() {
            GlobalQueueEntry retry = readyRetries.isEmpty() ? null : readyRetries.first();
            if (retry == null) {
                return nextToScan;
            }
            return nextToScan == null || retry.sequence < nextToScan.sequence ? retry : nextToScan;
        }

        GlobalQueueEntry pollNextRequest() {
            GlobalQueueEntry entry = peekNextRequest();
            if (entry != null) {
                if (entry == nextToScan) {
                    nextToScan = entry.next;
                }
                readyRetries.remove(entry);
            }
            return entry;
        }

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
            readyRetries.remove(entry);
            GlobalQueueEntry previous = entry.previous;
            GlobalQueueEntry next = entry.next;
            if (nextToScan == entry) {
                nextToScan = next;
            }
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
