package org.flexlb.balance.queue;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.worker.endpoint.CapacityKey;

import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.NavigableSet;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Pauses requests sharing a full role/group and permits one capacity retry at a time.
 * Successful admission passes the retry to the next waiter; a capacity miss waits for a
 * broadcast or periodic check. The scheduler lock protects every operation.
 */
public final class CapacityWaitQueue {
    private final Comparator<RequestContext> order;
    private final Map<CapacityKey, Domain> domains = new HashMap<>();
    private final Map<RequestContext, Domain> membership = new IdentityHashMap<>();
    private final NavigableSet<Domain> ready;

    public CapacityWaitQueue(Comparator<RequestContext> order) {
        this.order = order;
        ready =
                new TreeSet<>(
                        (left, right) ->
                                order.compare(left.entries.first(), right.entries.first()));
    }

    public boolean isWaiting(RequestContext request) {
        var domain = membership.get(request);
        return domain != null && domain.retry != request;
    }

    public void park(RequestContext request, CapacityKey blocker) {
        var key = blocker.waitingDomain();
        var domain = membership.get(request);
        if (domain == null || !domain.key.equals(key)) {
            remove(request, false);
            domain = domains.computeIfAbsent(key, Domain::new);
            membership.put(request, domain);
        }
        unindex(domain);
        if (domain.retry == request) domain.retry = null;
        domain.entries.add(request);
        index(domain);
    }

    public void remove(RequestContext request) {
        remove(request, true);
    }

    public void capacityChanged() {
        for (var domain : domains.values()) {
            domain.available = true;
            index(domain);
        }
    }

    public boolean hasReadyRequests() {
        return !ready.isEmpty();
    }

    public void resumeReady(
            int limit, Predicate<RequestContext> canRun, Consumer<RequestContext> resume) {
        var iterator = ready.iterator();
        for (int count = 0; count < limit && iterator.hasNext(); ) {
            var domain = iterator.next();
            var request = domain.entries.first();
            if (!canRun.test(request)) continue;
            iterator.remove();
            domain.entries.pollFirst();
            domain.retry = request;
            domain.available = false;
            resume.accept(request);
            count++;
        }
    }

    public void clear() {
        domains.clear();
        ready.clear();
        membership.clear();
    }

    private void remove(RequestContext request, boolean tryNext) {
        var domain = membership.remove(request);
        if (domain == null) return;
        unindex(domain);
        if (domain.retry == request) {
            domain.retry = null;
            domain.available |= tryNext;
        } else domain.entries.remove(request);
        index(domain);
    }

    private void unindex(Domain domain) {
        if (!domain.entries.isEmpty()) ready.remove(domain);
    }

    private void index(Domain domain) {
        if (domain.retry != null) return;
        if (domain.entries.isEmpty()) domains.remove(domain.key, domain);
        else if (domain.available) ready.add(domain);
    }

    /**
     * Keeps ordered waiters and the single capacity retry for one role/group.
     */
    private final class Domain {
        private final CapacityKey key;
        private final NavigableSet<RequestContext> entries = new TreeSet<>(order);
        private boolean available;
        private RequestContext retry;

        private Domain(CapacityKey key) {
            this.key = key;
        }
    }
}
