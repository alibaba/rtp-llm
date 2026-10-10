package org.flexlb.balance.scheduler;

import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import java.util.NavigableSet;
import java.util.TreeSet;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkState;

/**
 * FIFO or fixed-range PRIORITY index for the model queue.
 *
 * <p>The QueuedRequestScheduler lock is the sole synchronization boundary. Entries are
 * intrusive nodes, so completion and cancellation unlink an arbitrary request
 * in O(1) without leaving a removed entry behind a blocked head. Ready retries use
 * a separate ordered index and share the scan budget with forward progress.</p>
 */
final class OrderedRequestQueue {

    private static final int PRIORITY_LEVELS =
            PriorityNormalizer.MAX_PRIORITY + 1;

    private final boolean priorityOrdering;
    private final Comparator<GlobalQueueEntry> order;
    private final NavigableSet<GlobalQueueEntry> readyRetries;
    private final Bucket[] buckets;
    private final BitSet pendingPriorities = new BitSet(PRIORITY_LEVELS);
    private final int[] priorityCounts = new int[PRIORITY_LEVELS];
    private int size;
    private long nextSequence;

    OrderedRequestQueue(boolean priorityOrdering) {
        this.priorityOrdering = priorityOrdering;
        this.order = priorityOrdering ? GlobalQueueEntry.PRIORITY_ORDER : GlobalQueueEntry.SEQUENCE_ORDER;
        this.readyRetries = new TreeSet<>(order);
        this.buckets = new Bucket[priorityOrdering ? PRIORITY_LEVELS : 1];
    }

    void add(GlobalQueueEntry entry) {
        entry.sequence = ++nextSequence;
        insert(entry, false);
    }

    /** FIFO uses bucket zero; PRIORITY uses the request's priority as its bucket. */
    private int bucketIndex(GlobalQueueEntry entry) {
        return priorityOrdering ? entry.priority() : 0;
    }

    private void insert(GlobalQueueEntry entry, boolean restoring) {
        int index = bucketIndex(entry);
        Bucket bucket = buckets[index];
        if (bucket == null) { buckets[index] = bucket = new Bucket(); }
        bucket.insert(entry, restoring);
        size++;
        priorityCounts[entry.priority()]++;
        if (bucket.nextToScan == null || entry.sequence < bucket.nextToScan.sequence) {
            bucket.nextToScan = entry;
        }
        pendingPriorities.set(index);
    }

    int[] priorityCounts() {
        return priorityCounts.clone();
    }

    int size() {
        return size;
    }

    /** Rare route withdrawal: restore the original position without issuing a new sequence. */
    void restore(GlobalQueueEntry entry) {
        checkState(entry.removed && entry.sequence > 0L, "only a previously removed entry can be restored");
        insert(entry, true);
    }

    /**
     * Continue an ordered scan, counting every examined entry against the budget.
     * Eligibility may remove entries, but must not requeue an already selected entry.
     */
    List<GlobalQueueEntry> scanForPlanningCandidates(
            int candidateLimit, int scanBudget, Predicate<GlobalQueueEntry> eligible) {
        if (candidateLimit <= 0 || scanBudget <= 0) {
            return List.of();
        }
        List<GlobalQueueEntry> result = new ArrayList<>(Math.min(candidateLimit, size));
        for (int checked = 0; checked < scanBudget && result.size() < candidateLimit; checked++) {
            int index = pendingPriorities.previousSetBit(buckets.length - 1);
            GlobalQueueEntry forward = index < 0 ? null : buckets[index].nextToScan;
            GlobalQueueEntry retry = readyRetries.isEmpty() ? null : readyRetries.first();
            GlobalQueueEntry entry = retry == null || forward != null && order.compare(forward, retry) <= 0
                    ? forward : retry;
            if (entry == null) { break; }
            if (entry == forward) {
                buckets[index].nextToScan = entry.next;
                if (entry.next == null) { pendingPriorities.clear(index); }
            }
            readyRetries.remove(entry);
            if (eligible.test(entry)) {
                result.add(entry);
            }
        }
        result.sort(order);
        return result;
    }

    boolean hasUnscannedRequests() {
        return !pendingPriorities.isEmpty() || !readyRetries.isEmpty();
    }

    /** Make an awakened request directly available to the next bounded scan. */
    void markRequestReadyForRetry(GlobalQueueEntry entry) {
        if (entry.removed) { return; }
        readyRetries.add(entry);
    }

    boolean remove(GlobalQueueEntry entry) {
        if (entry.removed) { return false; }
        checkState(size > 0, "ordered queue size underflow");
        int index = bucketIndex(entry);
        Bucket bucket = buckets[index];
        checkState(bucket != null, "ordered queue bucket is missing");
        readyRetries.remove(entry);
        bucket.remove(entry);
        if (bucket.nextToScan == null) { pendingPriorities.clear(index); }
        size--;
        priorityCounts[entry.priority()]--;
        return true;
    }

    List<GlobalQueueEntry> drain() {
        List<GlobalQueueEntry> entries = new ArrayList<>(size);
        for (Bucket bucket : buckets) {
            if (bucket == null) { continue; }
            while (bucket.head != null) {
                GlobalQueueEntry entry = bucket.head;
                remove(entry);
                entries.add(entry);
            }
        }
        checkState(size == 0, "ordered queue drain left active entries");
        return entries;
    }

    private static final class Bucket {
        private GlobalQueueEntry head;
        private GlobalQueueEntry tail;
        private GlobalQueueEntry nextToScan;
        void insert(GlobalQueueEntry entry, boolean restoring) {
            checkState(entry.removed && entry.previous == null && entry.next == null,
                    "ordered queue entry is already linked");
            GlobalQueueEntry before = restoring ? head : null;
            while (before != null && before.sequence < entry.sequence) { before = before.next; }
            GlobalQueueEntry previous = before == null ? tail : before.previous;
            entry.previous = previous;
            entry.next = before;
            if (previous == null) { head = entry; } else { previous.next = entry; }
            if (before == null) { tail = entry; } else { before.previous = entry; }
            entry.removed = false;
        }

        void remove(GlobalQueueEntry entry) {
            GlobalQueueEntry previous = entry.previous;
            GlobalQueueEntry next = entry.next;
            if (nextToScan == entry) {
                nextToScan = next;
            }
            if (previous == null) {
                checkState(head == entry, "ordered queue head linkage is inconsistent");
                head = next;
            } else {
                previous.next = next;
            }
            if (next == null) {
                checkState(tail == entry, "ordered queue tail linkage is inconsistent");
                tail = previous;
            } else {
                next.previous = previous;
            }
            entry.previous = null;
            entry.next = null;
            entry.removed = true;
        }

    }
}
