package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static com.google.common.base.Preconditions.checkState;

/**
 * FIFO or fixed-range PRIORITY index for the model queue.
 *
 * <p>The QueuedRequestScheduler lock is the sole synchronization boundary. Private
 * nodes let completion and cancellation unlink an arbitrary request
 * in O(1) without leaving a removed entry behind a blocked head. Each scan keeps
 * its cursor until every request has been examined.</p>
 */
public final class OrderedRequestQueue implements SchedulingQueue {
    private static final int PRIORITY_LEVELS = PriorityNormalizer.MAX_PRIORITY + 1;

    private final boolean priorityOrdering;

    private final Comparator<RequestContext> order;
    private final Bucket[] buckets;
    private final BitSet pendingPriorities = new BitSet(PRIORITY_LEVELS);
    private final int[] priorityCounts = new int[PRIORITY_LEVELS];
    private final Map<RequestContext, Node> members = new IdentityHashMap<>();

    private long nextSequence;

    public OrderedRequestQueue(boolean priorityOrdering) {
        this.priorityOrdering = priorityOrdering;
        this.order =
                priorityOrdering
                        ? Comparator.comparingInt(SchedulingQueue::priority)
                                .reversed()
                                .thenComparingLong(RequestContext::getPlacementSequence)
                        : Comparator.comparingLong(RequestContext::getPlacementSequence);
        this.buckets = new Bucket[priorityOrdering ? PRIORITY_LEVELS : 1];
    }

    public void add(RequestContext request) {
        checkState(!members.containsKey(request), "request is already queued");
        request.setPlacementSequence(++nextSequence);
        insert(request, false);
    }

    public int[] priorityCounts() {
        return priorityCounts.clone();
    }

    public int size() {
        return members.size();
    }

    /**
     * Rare route withdrawal: restore the original position without issuing a new sequence.
     */
    public void restore(RequestContext request) {
        checkState(
                request.getPlacementSequence() > 0L,
                "only a previously queued request can be restored");
        insert(request, true);
    }

    /**
     * Continue an ordered scan, counting every examined entry against the budget.
     * Eligibility may remove entries, but must not requeue an already selected entry.
     */
    public List<RequestContext> scanForPlanningCandidates(
            int candidateLimit, int scanBudget, Predicate<RequestContext> eligible) {
        if (candidateLimit <= 0 || scanBudget <= 0) {
            return List.of();
        }
        List<RequestContext> result = new ArrayList<>(Math.min(candidateLimit, members.size()));
        for (int checked = 0; checked < scanBudget && result.size() < candidateLimit; checked++) {
            int index = pendingPriorities.previousSetBit(buckets.length - 1);
            if (index < 0) break;
            Node entry = buckets[index].nextToScan;
            buckets[index].nextToScan = entry.next;
            if (entry.next == null) pendingPriorities.clear(index);
            if (eligible.test(entry.request)) {
                result.add(entry.request);
            }
        }
        result.sort(order);
        return result;
    }

    public void beginScan() {
        for (int index = 0; index < buckets.length; index++) {
            var bucket = buckets[index];
            if (bucket == null) continue;
            bucket.nextToScan = bucket.head;
            pendingPriorities.set(index, bucket.head != null);
        }
    }

    public void retry(RequestContext request) {
        var entry = members.get(request);
        if (entry == null) return;
        int index = bucketIndex(request);
        var bucket = buckets[index];
        if (bucket.nextToScan == null
                || request.getPlacementSequence()
                        < bucket.nextToScan.request.getPlacementSequence()) {
            bucket.nextToScan = entry;
        }
        pendingPriorities.set(index);
    }

    public boolean hasUnscannedRequests() {
        return !pendingPriorities.isEmpty();
    }

    public List<RequestContext> drain() {
        List<RequestContext> entries = new ArrayList<>(members.size());
        for (Bucket bucket : buckets) {
            if (bucket == null) {
                continue;
            }
            while (bucket.head != null) {
                Node entry = bucket.head;
                remove(entry.request);
                entries.add(entry.request);
            }
        }
        checkState(members.isEmpty(), "ordered queue drain left active entries");
        return entries;
    }

    public boolean remove(RequestContext request) {
        Node entry = members.remove(request);
        if (entry == null) return false;
        int index = bucketIndex(request);
        Bucket bucket = buckets[index];
        bucket.remove(entry);
        if (bucket.nextToScan == null) pendingPriorities.clear(index);
        priorityCounts[SchedulingQueue.priority(request)]--;
        return true;
    }

    public boolean contains(RequestContext request) {
        return members.containsKey(request);
    }

    @Override
    public Comparator<RequestContext> order() {
        return order;
    }

    private void insert(RequestContext request, boolean restoring) {
        checkState(!members.containsKey(request), "request is already queued");
        Node entry = new Node(request);
        int index = bucketIndex(request);
        Bucket bucket = buckets[index];
        if (bucket == null) buckets[index] = bucket = new Bucket();
        bucket.insert(entry, restoring);
        members.put(request, entry);
        priorityCounts[SchedulingQueue.priority(request)]++;
        if (bucket.nextToScan == null
                || request.getPlacementSequence()
                        < bucket.nextToScan.request.getPlacementSequence()) {
            bucket.nextToScan = entry;
        }
        pendingPriorities.set(index);
    }

    /**
     * FIFO uses bucket zero; PRIORITY uses the request's priority as its bucket.
     */
    private int bucketIndex(RequestContext request) {
        return priorityOrdering ? SchedulingQueue.priority(request) : 0;
    }

    /**
     * Links one queued request to its neighbors so removal preserves FIFO order.
     */
    private static final class Node {
        private final RequestContext request;
        private Node previous;
        private Node next;

        private Node(RequestContext request) {
            this.request = request;
        }
    }

    /**
     * Retains FIFO links and the next request to inspect at one priority.
     */
    private static final class Bucket {
        private Node head;
        private Node tail;
        private Node nextToScan;

        private void insert(Node entry, boolean restoring) {
            Node before = restoring ? head : null;
            while (before != null
                    && before.request.getPlacementSequence()
                            < entry.request.getPlacementSequence()) {
                before = before.next;
            }
            Node previous = before == null ? tail : before.previous;
            entry.previous = previous;
            entry.next = before;
            if (previous == null) {
                head = entry;
            } else {
                previous.next = entry;
            }
            if (before == null) {
                tail = entry;
            } else {
                before.previous = entry;
            }
        }

        private void remove(Node entry) {
            Node previous = entry.previous;
            Node next = entry.next;
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
        }
    }
}
