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
 * Capacity-event waiting for the model queue. All methods run under the QueuedRequestScheduler lock.
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
    private final Map<GlobalQueueEntry, Domain> membership = new IdentityHashMap<>();
    private final NavigableSet<Domain> ready;

    PlacementWaitQueue(boolean priorityOrdering, PlacementAvailability availability) {
        this.availability = Objects.requireNonNull(availability, "availability");
        order = priorityOrdering ? GlobalQueueEntry.PRIORITY_ORDER : GlobalQueueEntry.SEQUENCE_ORDER;
        ready = new TreeSet<>((left, right) -> order.compare(left.entries.first(), right.entries.first()));
    }

    boolean isWaiting(GlobalQueueEntry entry) {
        Domain domain = membership.get(entry);
        return domain != null && domain.retry != entry;
    }

    /** The only park boundary: do not sleep through an edge published during planning. */
    boolean park(GlobalQueueEntry entry, PlacementKey blocker, long observedSequence) {
        Objects.requireNonNull(blocker, "blocker");
        if (availability.lastChangedSequence(blocker.capacityDomain()) > observedSequence) {
            return false;
        }
        remove(entry, false);
        PlacementKey key = blocker.capacityDomain();
        Domain domain = domains.computeIfAbsent(key, Domain::new);
        unindex(domain);
        domain.entries.add(entry);
        membership.put(entry, domain);
        index(domain);
        return true;
    }

    /** Constant number of domain lookups, independent of backlog size. */
    void capacityChanged(PlacementKey key) {
        if (key.endpoint() != null) {
            release(key.capacityDomain());
        }
        if (key.group() != null) {
            release(new PlacementKey(key.role(), key.group(), null));
        }
        release(PlacementKey.anyGroup(key.role()));
    }

    boolean hasReadyRequests() {
        return !ready.isEmpty();
    }

    void resumeReady(int limit, Consumer<GlobalQueueEntry> resume) {
        for (int count = 0; count < limit && !ready.isEmpty(); count++) {
            Domain domain = ready.pollFirst();
            GlobalQueueEntry entry = domain.entries.pollFirst();
            domain.retry = entry;
            domain.available = false;
            // Even completed entries need an owner-side cleanup opportunity.
            resume.accept(entry);
        }
    }

    /** A departing request did not consume the remaining retry opportunity. */
    void remove(GlobalQueueEntry entry) {
        remove(entry, true);
    }

    void clear() {
        domains.clear();
        ready.clear();
        membership.clear();
    }

    private void remove(GlobalQueueEntry entry, boolean progress) {
        Domain domain = membership.remove(entry);
        if (domain == null) { return; }
        unindex(domain);
        if (domain.retry == entry) {
            domain.retry = null;
            // An edge arriving during this retry remains available.
            domain.available |= progress;
        } else {
            domain.entries.remove(entry);
        }
        index(domain);
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
        if (domain.retry == null) {
            if (domain.entries.isEmpty()) {
                domains.remove(domain.key, domain);
            } else if (domain.available) {
                ready.add(domain);
            }
        }
    }

    private final class Domain {
        private final PlacementKey key;
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private boolean available;
        private GlobalQueueEntry retry;

        private Domain(PlacementKey key) {
            this.key = key;
        }
    }
}
