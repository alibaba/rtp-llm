package org.flexlb.balance.scheduler;

import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import java.util.NavigableSet;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Predicate;

/**
 * FIFO scan order for the model queue, with an optional fixed-range PRIORITY
 * index kept only to pick a preemption victim when the caller's capacity
 * check reports the queue full.
 *
 * <p>{@link #scanForPlanningCandidates} always returns pending requests in
 * enqueue order, regardless of priority: a request's priority never changes
 * when it gets planned while the queue has room. Priority only matters to
 * {@link #findQueuedRequestToPreempt}, which the caller invokes solely when
 * admitting a new request would exceed the configured queue capacity.</p>
 *
 * <p>The coordinator lock is the sole synchronization boundary. Entries are
 * intrusive nodes, so completion and cancellation unlink an arbitrary request
 * in O(1) without leaving a removed entry behind a blocked head. Ready retries use
 * a separate ordered index and share the scan budget with forward progress.</p>
 */
final class OrderedRequestQueue {

    private static final int PRIORITY_LEVELS = PriorityNormalizer.MAX_PRIORITY + 1;

    private static final Comparator<GlobalQueueEntry> SEQUENCE_ORDER =
            Comparator.comparingLong(entry -> entry.sequence);

    private final boolean priorityOrdering;
    private final Bucket fifo = new Bucket();
    private final Bucket[] priorityBuckets = new Bucket[PRIORITY_LEVELS];
    private final BitSet pendingPriorities = new BitSet(PRIORITY_LEVELS);
    private final int[] priorityCounts = new int[PRIORITY_LEVELS];
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
        } else {
            fifo.add(entry);
        }
        size++;
        priorityCounts[entry.priority]++;
        rewindScanTo(entry);
    }

    int[] priorityCounts() {
        return priorityCounts.clone();
    }

    int size() {
        return size;
    }

    long oldestEnteredAtNanos() {
        long oldest = Long.MAX_VALUE;
        if (priorityOrdering) {
            for (Bucket bucket : priorityBuckets) {
                if (bucket != null) {
                    oldest = Math.min(oldest, bucket.oldestEnteredAtNanos());
                }
            }
        } else {
            oldest = fifo.oldestEnteredAtNanos();
        }
        return oldest == Long.MAX_VALUE ? 0L : oldest;
    }

    /**
     * Find a lower-priority request to preempt without removing it.
     * Check lower priority levels first; within a level, choose the newest request
     * that is not being planned or already completed. Caller holds the queue lock.
     */
    GlobalQueueEntry findQueuedRequestToPreempt(int incomingPriority, Set<GlobalQueueEntry> requestsBeingPlanned) {
        for (int priority = PriorityNormalizer.MIN_PRIORITY; priority < incomingPriority; priority++) {
            Bucket bucket = priorityBuckets[priority];
            if (bucket == null) {
                continue;
            }
            for (GlobalQueueEntry entry = bucket.tail; entry != null; entry = entry.previous) {
                if (!requestsBeingPlanned.contains(entry) && !entry.future.isDone()) {
                    return entry;
                }
            }
        }
        return null;
    }

    /** Rare route withdrawal: restore the original position without issuing a new sequence. */
    void restore(GlobalQueueEntry entry) {
        if (!entry.removed || entry.linked || entry.sequence <= 0L) {
            throw new IllegalStateException("only a previously removed entry can be restored");
        }
        Bucket bucket = priorityOrdering ? priorityBuckets[entry.priority] : fifo;
        bucket.restore(entry);
        entry.removed = false;
        size++;
        priorityCounts[entry.priority]++;
        rewindScanTo(entry);
    }

    /** Continue an ordered scan, counting every examined entry against the budget. */
    List<GlobalQueueEntry> scanForPlanningCandidates(
            int candidateLimit, int scanBudget, Predicate<GlobalQueueEntry> eligible) {
        if (candidateLimit <= 0 || scanBudget <= 0) {
            return List.of();
        }
        List<GlobalQueueEntry> result = new ArrayList<>(Math.min(candidateLimit, size));
        for (int checked = 0; checked < scanBudget && result.size() < candidateLimit; checked++) {
            GlobalQueueEntry entry = pollOldestPendingRequest();
            if (entry == null) {
                break;
            }
            if (!result.contains(entry) && eligible.test(entry)) {
                result.add(entry);
            }
        }
        return result;
    }

    /**
     * Remove and return the pending request with the smallest enqueue sequence across
     * every priority level, so planning always proceeds in enqueue order regardless of
     * priority. Priority buckets stay in place purely so {@link #findQueuedRequestToPreempt}
     * can still pick a victim by priority when the queue is full.
     */
    private GlobalQueueEntry pollOldestPendingRequest() {
        if (!priorityOrdering) {
            return fifo.pollNextRequest();
        }
        int oldestPriority = -1;
        GlobalQueueEntry oldest = null;
        for (int priority = 0; priority < PRIORITY_LEVELS; priority++) {
            if (!pendingPriorities.get(priority)) {
                continue;
            }
            GlobalQueueEntry candidate = priorityBuckets[priority].peekNextRequest();
            if (candidate == null) {
                continue;
            }
            if (oldest == null || candidate.sequence < oldest.sequence) {
                oldest = candidate;
                oldestPriority = priority;
            }
        }
        if (oldest == null) {
            return null;
        }
        Bucket bucket = priorityBuckets[oldestPriority];
        GlobalQueueEntry entry = bucket.pollNextRequest();
        if (!bucket.hasPendingRequests()) {
            pendingPriorities.clear(oldestPriority);
        }
        return entry;
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
        priorityCounts[entry.priority]--;
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

        long oldestEnteredAtNanos() {
            long oldest = Long.MAX_VALUE;
            for (GlobalQueueEntry entry = head; entry != null; entry = entry.next) {
                oldest = Math.min(oldest, entry.enteredQueueAtNanos);
            }
            return oldest;
        }

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

        void restore(GlobalQueueEntry entry) {
            GlobalQueueEntry before = head;
            while (before != null && before.sequence < entry.sequence) { before = before.next; }
            if (before == null) {
                add(entry);
                return;
            }
            entry.next = before;
            entry.previous = before.previous;
            if (before.previous == null) { head = entry; } else { before.previous.next = entry; }
            before.previous = entry;
            entry.linked = true;
        }

        boolean isEmpty() {
            return head == null;
        }
    }
}
