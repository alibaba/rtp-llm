package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.PrefillCostOrder.Candidate;

import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.TreeSet;
import java.util.function.Consumer;

/**
 * Capacity-event waiting for the model queue. All methods run under the coordinator lock.
 *
 * <p>A capacity edge opens one retry per domain. Successful admission (or cancellation
 * before admission) preserves the opportunity for the next waiter; a failed attempt
 * consumes it. No endpoint probing or request-specific successor state is needed.
 * Ready domains are globally ordered, and each scan transfers a bounded number of
 * requests. A wakeup grants a fresh route decision, never an endpoint reservation.</p>
 */
final class PlacementWaitQueue {
    private record Waiter(Domain domain, Candidate<GlobalQueueEntry> rank) { }

    private final PlacementAvailability availability;
    private final Comparator<Waiter> waiterOrder;
    private final Map<PlacementKey, Domain> domains = new HashMap<>();
    private final Map<GlobalQueueEntry, Waiter> waiting = new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, Domain> retrying = new IdentityHashMap<>();
    private final NavigableSet<Domain> ready;

    PlacementWaitQueue(boolean priorityOrdering, PlacementAvailability availability) {
        this.availability = Objects.requireNonNull(availability, "availability");
        Comparator<GlobalQueueEntry> fifo = Comparator.comparingLong(entry -> entry.sequence);
        Comparator<GlobalQueueEntry> order = priorityOrdering
                ? Comparator.<GlobalQueueEntry>comparingInt(entry -> entry.priority).reversed().thenComparing(fifo)
                : fifo;
        waiterOrder = Comparator.comparing(Waiter::rank, PrefillCostOrder.comparator(priorityOrdering));
        ready = new TreeSet<>((left, right) ->
                order.compare(left.entries.first().rank.value(), right.entries.first().rank.value()));
    }

    boolean isWaiting(GlobalQueueEntry entry) {
        return waiting.containsKey(entry);
    }

    /** The only park boundary: do not sleep through an edge published during planning. */
    boolean park(GlobalQueueEntry entry, PlacementKey blocker, long observedSequence) {
        return park(entry, blocker, observedSequence, null);
    }

    boolean park(GlobalQueueEntry entry, PlacementKey blocker, long observedSequence, Candidate<GlobalQueueEntry> rank) {
        Objects.requireNonNull(blocker, "blocker");
        if (availability.lastChangedSequence(waitKey(blocker)) > observedSequence) {
            return false;
        }
        finishRetry(entry, false);
        removeWaiting(entry);
        PlacementKey key = waitKey(blocker);
        Domain domain = domains.computeIfAbsent(key, Domain::new);
        unindex(domain);
        addWaiter(domain, rank != null && domain.isPrefillWorker() ? rank
                : new Candidate<>(entry, entry.priority, entry.sequence, 0L, null));
        return true;
    }

    /** An asynchronous lookup can finish after the request parks. Replace its immutable tree key. */
    void updatePrefillCost(Candidate<GlobalQueueEntry> rank) {
        Waiter previous = waiting.get(rank.value());
        if (previous == null || !previous.domain.isPrefillWorker() || previous.rank.equals(rank)) {
            return;
        }
        Domain domain = previous.domain;
        unindex(domain);
        domain.entries.remove(previous);
        addWaiter(domain, rank);
    }

    private void addWaiter(Domain domain, Candidate<GlobalQueueEntry> rank) {
        var waiter = new Waiter(domain, rank);
        domain.entries.add(waiter);
        waiting.put(rank.value(), waiter);
        index(domain);
    }

    /** Constant number of domain lookups, independent of backlog size. */
    void capacityChanged(PlacementKey key) {
        if (key.endpoint() != null) {
            release(waitKey(key));
        }
        if (key.group() != null) {
            release(new PlacementKey(key.role(), key.group()));
        }
        release(PlacementKey.anyGroup(key.role()));
    }

    boolean hasReadyRequests() {
        return !ready.isEmpty();
    }

    void resumeReady(int limit, Consumer<GlobalQueueEntry> resume) {
        for (int count = 0; count < limit && !ready.isEmpty(); count++) {
            Domain domain = ready.pollFirst();
            GlobalQueueEntry entry = domain.entries.pollFirst().rank.value();
            domain.active = true;
            domain.available = false;
            waiting.remove(entry);
            retrying.put(entry, domain);
            // Even completed entries need an owner-side cleanup opportunity.
            resume.accept(entry);
        }
    }

    /** A departing request did not consume the remaining retry opportunity. */
    void remove(GlobalQueueEntry entry) {
        finishRetry(entry, true);
        removeWaiting(entry);
    }

    void clear() {
        domains.clear();
        ready.clear();
        waiting.clear();
        retrying.clear();
    }

    private void finishRetry(GlobalQueueEntry entry, boolean progress) {
        Domain domain = retrying.remove(entry);
        if (domain == null) {
            return;
        }
        domain.active = false;
        // An edge arriving while the retry was active remains available.
        domain.available |= progress;
        index(domain);
    }

    private void removeWaiting(GlobalQueueEntry entry) {
        Waiter waiter = waiting.remove(entry);
        if (waiter != null) {
            Domain domain = waiter.domain;
            unindex(domain);
            domain.entries.remove(waiter);
            index(domain);
        }
    }

    private void release(PlacementKey key) {
        Domain domain = domains.get(key);
        if (domain != null) {
            domain.available = true;
            index(domain);
        }
    }

    private void unindex(Domain domain) {
        if (!domain.entries.isEmpty()) {
            ready.remove(domain);
        }
    }

    private void index(Domain domain) {
        if (!domain.active) {
            if (domain.entries.isEmpty()) {
                domains.remove(domain.key, domain);
            } else if (domain.available) {
                ready.add(domain);
            }
        }
    }

    private static PlacementKey waitKey(PlacementKey key) {
        // Exact waiters follow role/address even when a replacement changes groups.
        return key.endpoint() == null ? key : PlacementKey.exact(key.role(), null, key.endpoint());
    }

    private final class Domain {
        private final PlacementKey key;
        private final NavigableSet<Waiter> entries = new TreeSet<>(waiterOrder);
        private boolean available;
        private boolean active;

        private boolean isPrefillWorker() {
            return key.endpoint() != null && key.role().supportsPrefill();
        }

        private Domain(PlacementKey key) {
            this.key = key;
        }
    }
}
