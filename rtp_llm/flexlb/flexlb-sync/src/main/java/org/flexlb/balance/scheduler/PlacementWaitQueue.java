package org.flexlb.balance.scheduler;

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
    private final PlacementAvailability availability;
    private final Comparator<GlobalQueueEntry> order;
    private final Map<PlacementKey, Domain> domains = new HashMap<>();
    private final Map<GlobalQueueEntry, Domain> waiting = new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, Domain> retrying = new IdentityHashMap<>();
    private final NavigableSet<Domain> ready;

    PlacementWaitQueue(boolean priorityOrdering, PlacementAvailability availability) {
        this.availability = Objects.requireNonNull(availability, "availability");
        Comparator<GlobalQueueEntry> fifo = Comparator.comparingLong(entry -> entry.sequence);
        order = priorityOrdering
                ? Comparator.<GlobalQueueEntry>comparingInt(entry -> entry.priority).reversed().thenComparing(fifo)
                : fifo;
        ready = new TreeSet<>((left, right) -> order.compare(left.entries.first(), right.entries.first()));
    }

    boolean isWaiting(GlobalQueueEntry entry) {
        return waiting.containsKey(entry);
    }

    /** The only park boundary: do not sleep through an edge published during planning. */
    boolean park(GlobalQueueEntry entry, PlacementKey blocker, long observedSequence) {
        Objects.requireNonNull(blocker, "blocker");
        if (availability.lastChangedSequence(waitKey(blocker)) > observedSequence) {
            return false;
        }
        finishRetry(entry, false);
        removeWaiting(entry);
        PlacementKey key = waitKey(blocker);
        Domain domain = domains.computeIfAbsent(key, Domain::new);
        unindex(domain);
        domain.entries.add(entry);
        waiting.put(entry, domain);
        index(domain);
        return true;
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

    boolean hasEarlierReadyRequest(GlobalQueueEntry entry) {
        return !ready.isEmpty() && order.compare(ready.first().entries.first(), entry) < 0;
    }

    boolean hasReadyRequests() {
        return !ready.isEmpty();
    }

    void resumeReady(int limit, Consumer<GlobalQueueEntry> resume) {
        for (int count = 0; count < limit && !ready.isEmpty(); count++) {
            Domain domain = ready.pollFirst();
            GlobalQueueEntry entry = domain.entries.pollFirst();
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
        Domain domain = waiting.remove(entry);
        if (domain != null) {
            unindex(domain);
            domain.entries.remove(entry);
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
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private boolean available;
        private boolean active;

        private Domain(PlacementKey key) {
            this.key = key;
        }
    }
}
