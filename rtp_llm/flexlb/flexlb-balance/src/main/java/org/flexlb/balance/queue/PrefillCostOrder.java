package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.function.Predicate;

/**
 * Ranks a queue snapshot by priority and exact compute/input ratios.
 * Unknown estimates use stable ordering; estimates remain usable while a refresh is in flight.
 */
final class PrefillCostOrder {
    private final boolean priority;

    private final Comparator<RequestContext> stableOrder;
    private final Map<RequestContext, Entry> entries = new IdentityHashMap<>();
    private final java.util.LinkedHashSet<SchedulingQueue.EstimateRequest> refresh =
            new java.util.LinkedHashSet<>();

    PrefillCostOrder(boolean priority, Comparator<RequestContext> stableOrder) {
        this.priority = priority;
        this.stableOrder = stableOrder;
    }

    void add(RequestContext request) {
        var entry = new Entry();
        entries.put(request, entry);
        enqueue(request, entry);
    }

    List<RequestContext> select(
            List<RequestContext> snapshot, int limit, Predicate<RequestContext> eligible) {
        var ready = new ArrayList<RequestContext>();
        boolean unknown = false;
        long now = System.nanoTime();
        for (var request : snapshot) {
            if (!eligible.test(request)) continue;
            var entry = entries.get(request);
            if (entry == null) continue;
            ready.add(request);
            unknown |= entry.compute < 0 || input(request) <= 0;
            if (entry.ticket == null
                    && now - entry.estimatedAtNanos
                            >= java.util.concurrent.TimeUnit.SECONDS.toNanos(1))
                enqueue(request, entry);
        }
        if (ready.size() <= 1) return ready;
        Comparator<RequestContext> order = unknown ? stableOrder : this::compare;
        // As in GLM, keep the worst of the retained requests at the heap head.
        var selected =
                new PriorityQueue<RequestContext>(Math.min(limit, ready.size()), order.reversed());
        for (var request : ready) {
            if (selected.size() < limit) {
                selected.add(request);
            } else if (order.compare(request, selected.element()) < 0) {
                selected.remove();
                selected.add(request);
            }
        }
        return selected.stream().sorted(order).toList();
    }

    List<SchedulingQueue.EstimateRequest> estimates(int limit) {
        var result = new ArrayList<SchedulingQueue.EstimateRequest>();
        while (result.size() < limit && !refresh.isEmpty()) {
            var iterator = refresh.iterator();
            var ticket = iterator.next();
            iterator.remove();
            var entry = entries.get(ticket.request());
            if (entry != null && entry.ticket == ticket) result.add(ticket);
        }
        return result;
    }

    void remove(RequestContext request) {
        var entry = entries.remove(request);
        if (entry != null && entry.ticket != null) refresh.remove(entry.ticket);
    }

    private int compare(RequestContext a, RequestContext b) {
        int compared =
                priority
                        ? Integer.compare(SchedulingQueue.priority(b), SchedulingQueue.priority(a))
                        : 0;
        if (compared != 0) return compared;
        long left = entries.get(a).compute;
        long right = entries.get(b).compute;
        compared = ratio(left, input(a), right, input(b));
        if (compared == 0) compared = Long.compare(left, right);
        if (compared == 0) compared = Long.compare(input(a), input(b));
        return compared == 0 ? stableOrder.compare(a, b) : compared;
    }

    private static int ratio(long left, long leftInput, long right, long rightInput) {
        try {
            return Long.compare(
                    Math.multiplyExact(left, rightInput), Math.multiplyExact(right, leftInput));
        } catch (ArithmeticException overflow) {
            return BigInteger.valueOf(left)
                    .multiply(BigInteger.valueOf(rightInput))
                    .compareTo(BigInteger.valueOf(right).multiply(BigInteger.valueOf(leftInput)));
        }
    }

    private static long input(RequestContext request) {
        return request.getRequest().getSeqLen();
    }

    private void enqueue(RequestContext request, Entry entry) {
        entry.ticket =
                new SchedulingQueue.EstimateRequest(
                        request,
                        estimate -> {
                            if (entries.get(request) != entry || entry.ticket == null) return;
                            entry.compute = estimate.orElse(-1L);
                            entry.ticket = null;
                            entry.estimatedAtNanos = System.nanoTime();
                        });
        refresh.add(entry.ticket);
    }

    /**
     * Owns the cached work estimate and outstanding refresh for one queue membership.
     */
    private static final class Entry {
        private long compute = -1L;
        private SchedulingQueue.EstimateRequest ticket;
        private long estimatedAtNanos;
    }
}
