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
 * Queues requests waiting for worker capacity. All methods run under the QueuedRequestScheduler lock.
 *
 * <p>A capacity edge opens one retry per domain. Successful admission (or cancellation
 * before admission) preserves the opportunity for the next waiter; a failed attempt
 * consumes it. No endpoint probing or request-specific successor state is needed.
 * Ready domains are globally ordered, and each scan transfers a bounded number of
 * requests. A wakeup grants a fresh route decision, never an endpoint reservation.</p>
 */
final class CapacityWaitQueue {
    private final CapacityNotifier capacityNotifier;
    private final Comparator<GlobalQueueEntry> order;
    private final Map<CapacityKey, Domain> domains = new HashMap<>();
    private final Map<GlobalQueueEntry, Domain> membership = new IdentityHashMap<>();
    private final NavigableSet<Domain> ready;

    CapacityWaitQueue(Comparator<GlobalQueueEntry> order, CapacityNotifier capacityNotifier) {
        this.capacityNotifier = Objects.requireNonNull(capacityNotifier, "capacityNotifier");
        this.order = Objects.requireNonNull(order, "order");
        ready =
                new TreeSet<>(
                        (left, right) ->
                                order.compare(left.entries.first(), right.entries.first()));
    }

    boolean isWaiting(GlobalQueueEntry entry) {
        Domain domain = membership.get(entry);
        return domain != null && domain.retry != entry;
    }

    boolean contains(GlobalQueueEntry entry) {
        return membership.containsKey(entry);
    }

    /**
     * Register a waiter without missing a capacity change published during selection.
     */
    boolean park(GlobalQueueEntry entry, CapacityKey blocker, long observedSequence) {
        Objects.requireNonNull(blocker, "blocker");
        CapacityKey key = blocker.waitingDomain();
        if (capacityNotifier.lastChangedSequence(key) > observedSequence) {
            return false;
        }
        remove(entry, false);
        Domain domain = domains.computeIfAbsent(key, Domain::new);
        unindex(domain);
        domain.entries.add(entry);
        membership.put(entry, domain);
        index(domain);
        return true;
    }

    /** A departing request did not consume the remaining retry opportunity. */
    void remove(GlobalQueueEntry entry) {
        remove(entry, true);
    }

    /**
     * Wake the affected group and role with a constant number of lookups.
     */
    void capacityChanged(CapacityKey key) {
        if (key.group() != null) {
            release(new CapacityKey(key.role(), key.group(), null));
        }
        release(CapacityKey.anyGroup(key.role()));
    }

    boolean hasReadyRequests() {
        return !ready.isEmpty();
    }

    int size() {
        return membership.size();
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

    void clear() {
        domains.clear();
        ready.clear();
        membership.clear();
    }

    private void remove(GlobalQueueEntry entry, boolean progress) {
        Domain domain = membership.remove(entry);
        if (domain == null) {
            return;
        }
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

    private void release(CapacityKey key) {
        Domain domain = domains.get(key);
        if (domain != null) {
            domain.available = true;
            index(domain);
        }
    }

    /**
     * Orders waiters sharing a capacity domain and retains its current retry opportunity.
     */
    private final class Domain {
        private final CapacityKey key;
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private boolean available;
        private GlobalQueueEntry retry;

        private Domain(CapacityKey key) {
            this.key = key;
        }
    }
}
