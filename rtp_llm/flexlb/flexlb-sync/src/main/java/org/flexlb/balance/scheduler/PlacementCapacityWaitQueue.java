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
 * Queues requests waiting for placement capacity. All methods run under the QueuedRequestScheduler lock.
 *
 * <p>A capacity edge opens one retry per domain, cache-affinity cohort, and Prefill admission class.
 * Successful admission (or cancellation before admission) preserves the opportunity
 * for the next waiter; a failed attempt
 * consumes it. A group-level cache or topology change instead invalidates the
 * parked affinity plans: requests sharing the previous leader set may now have
 * different eligible workers. Recheck the oldest cache version first, in queue order
 * within that version.
 * Ready domains are globally ordered, and each scan transfers a bounded number of
 * requests. A wakeup grants a fresh route decision, never an endpoint reservation.</p>
 */
final class PlacementCapacityWaitQueue {
    private final PlacementAvailability availability;
    private final Comparator<GlobalQueueEntry> order;
    // Cold token admission and hot Engine-queue admission are different predicates.
    // A failed cold retry is not evidence that a hot waiter cannot proceed.
    private record WaitKey(PlacementKey capacity, boolean cold) { }
    private final Map<WaitKey, Domain> domains = new HashMap<>();
    /** Reverse index of cache-admission cohorts, never a scan of individual requests. */
    private final Map<PlacementKey, java.util.Set<Domain>> affinityDomains = new HashMap<>();
    private final Map<GlobalQueueEntry, Domain> membership = new IdentityHashMap<>();
    private final NavigableSet<Domain> ready;
    PlacementCapacityWaitQueue(boolean priorityOrdering, PlacementAvailability availability) {
        this(priorityOrdering, availability, 0L);
    }

    private final NavigableSet<GlobalQueueEntry> rescueWaiters;
    private long nextProbeNanos;
    private final long probeIntervalNanos;

    PlacementCapacityWaitQueue(boolean priorityOrdering, PlacementAvailability availability, long probeIntervalNanos) {
        this.probeIntervalNanos = probeIntervalNanos;
        var rescueOrder = Comparator.comparingInt((GlobalQueueEntry e) -> priorityOrdering ? -e.priority() : 0)
                .thenComparingInt(e -> e.rescueAttempts)
                .thenComparingDouble(e -> e.rescueHitFraction)
                .thenComparing(GlobalQueueEntry.SEQUENCE_ORDER);
        rescueWaiters = probeIntervalNanos > 0 ? new TreeSet<>(rescueOrder) : null;
        this.availability = Objects.requireNonNull(availability, "availability");
        order = priorityOrdering ? GlobalQueueEntry.PRIORITY_ORDER : GlobalQueueEntry.SEQUENCE_ORDER;
        ready = new TreeSet<>((left, right) -> order.compare(left.next(), right.next()));
    }

    boolean isWaiting(GlobalQueueEntry entry) {
        Domain domain = membership.get(entry);
        return domain != null && domain.retry != entry;
    }

    /** The only park boundary: do not sleep through an edge published during planning. */
    boolean park(GlobalQueueEntry entry, PlacementKey blocker, long observedSequence) {
        return park(entry, blocker, observedSequence, Double.NaN);
    }

    boolean park(GlobalQueueEntry entry, PlacementKey blocker, long observedSequence, double hitFraction) {
        Objects.requireNonNull(blocker, "blocker");
        if (availability.lastChangedSequence(blocker.capacityDomain()) > observedSequence) {
            return false;
        }
        remove(entry, false);
        WaitKey key = new WaitKey(blocker.capacityDomain(),
                blocker.role().supportsPrefill() && entry.knownColdRequest);
        Domain domain = domains.computeIfAbsent(key, this::newDomain);
        unindex(domain);
        domain.add(entry);
        if (rescueWaiters != null && !blocker.cacheWorkers().isEmpty() && Double.isFinite(hitFraction)) {
            entry.rescueHitFraction = hitFraction;
            if (rescueWaiters.isEmpty()) { nextProbeNanos = System.nanoTime() + probeIntervalNanos; }
            rescueWaiters.add(entry);
        }
        membership.put(entry, domain);
        index(domain);
        return true;
    }

    long nanosUntilProbe() {
        return rescueWaiters == null || rescueWaiters.isEmpty() ? Long.MAX_VALUE : nextProbeNanos - System.nanoTime();
    }

    GlobalQueueEntry pollProbe() {
        if (nanosUntilProbe() > 0) { return null; }
        var entry = rescueWaiters.first();
        remove(entry, false);
        entry.rescueAttempts++;
        nextProbeNanos = System.nanoTime() + probeIntervalNanos;
        return entry;
    }

    /** Wake only the cohorts whose cache candidates include this worker. */
    void capacityChanged(PlacementKey key) {
        if (key.endpoint() != null) {
            release(key.capacityDomain());
        }
        if (key.group() != null) {
            release(new PlacementKey(key.role(), key.group(), null));
        }
        release(PlacementKey.anyGroup(key.role()));
        if (key.endpoint() != null) {
            releaseAffinities(key.capacityDomain(), false);
        } else {
            releaseAffinities(key, false);
            if (key.group() != null) { releaseAffinities(PlacementKey.anyGroup(key.role()), true); }
        }
    }

    private void releaseAffinities(PlacementKey key, boolean wildcardWaitersOnly) {
        var affected = affinityDomains.get(key);
        if (affected != null) {
            for (Domain domain : affected) {
                if (wildcardWaitersOnly && domain.key.capacity().group() != null) { continue; }
                unindex(domain);
                if (key.endpoint() == null) {
                    // A cache/topology update can make only a peer's prefix eligible.
                    // Invalidate each parked plan once, issuing one retry per domain at a time.
                    domain.cacheVersion++;
                } else {
                    domain.available = true;
                }
                index(domain);
            }
        }
    }

    private java.util.Set<PlacementKey> affinityEdges(WaitKey key) {
        var capacity = key.capacity;
        var edges = new java.util.HashSet<PlacementKey>();
        if (!capacity.cacheWorkers().isEmpty()) {
            for (String worker : capacity.cacheWorkers()) {
                edges.add(PlacementKey.exact(capacity.role(), null, worker));
            }
            edges.add(new PlacementKey(capacity.role(), capacity.group(), null));
            edges.add(PlacementKey.anyGroup(capacity.role()));
        }
        return edges;
    }

    private Domain newDomain(WaitKey key) {
        Domain domain = new Domain(key);
        for (PlacementKey edge : affinityEdges(key)) {
            affinityDomains.computeIfAbsent(edge, ignored -> new java.util.HashSet<>()).add(domain);
        }
        return domain;
    }

    private void removeDomain(Domain domain) {
        domains.remove(domain.key, domain);
        for (PlacementKey edge : affinityEdges(domain.key)) {
            var listeners = affinityDomains.get(edge);
            listeners.remove(domain);
            if (listeners.isEmpty()) { affinityDomains.remove(edge); }
        }
    }

    boolean hasReadyRequests() {
        return !ready.isEmpty();
    }

    void resumeReady(int limit, Consumer<GlobalQueueEntry> resume) {
        for (int count = 0; count < limit && !ready.isEmpty(); count++) {
            Domain domain = ready.pollFirst();
            GlobalQueueEntry entry = domain.next();
            domain.take(entry);
            if (rescueWaiters != null) { rescueWaiters.remove(entry); }
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
        if (rescueWaiters != null) { rescueWaiters.clear(); }
        domains.clear();
        affinityDomains.clear();
        ready.clear();
        membership.clear();
    }

    private void remove(GlobalQueueEntry entry, boolean progress) {
        if (rescueWaiters != null) { rescueWaiters.remove(entry); }
        Domain domain = membership.remove(entry);
        if (domain == null) { return; }
        unindex(domain);
        if (domain.retry == entry) {
            domain.retry = null;
            // An edge arriving during this retry remains available.
            domain.available |= progress;
        } else {
            domain.take(entry);
        }
        index(domain);
    }

    private void release(PlacementKey key) {
        release(new WaitKey(key, false));
        if (key.role().supportsPrefill()) { release(new WaitKey(key, true)); }
    }

    private void release(WaitKey key) {
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
                removeDomain(domain);
            } else if (domain.available || domain.hasStaleCachePlan()) {
                ready.add(domain);
            }
        }
    }

    private final class Domain {
        private final WaitKey key;
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private boolean available;
        private GlobalQueueEntry retry;

        private long cacheVersion;
        private final Map<GlobalQueueEntry, Long> checkedVersion;
        private final NavigableSet<GlobalQueueEntry> cacheOrder;

        private Domain(WaitKey key) {
            this.key = key;
            checkedVersion = key.capacity().cacheWorkers().isEmpty() ? null : new IdentityHashMap<>();
            cacheOrder = checkedVersion == null ? null : new TreeSet<>(
                    Comparator.<GlobalQueueEntry>comparingLong(checkedVersion::get).thenComparing(order));
        }

        private void add(GlobalQueueEntry entry) {
            entries.add(entry);
            if (cacheOrder != null) {
                checkedVersion.put(entry, cacheVersion);
                cacheOrder.add(entry);
            }
        }

        private void take(GlobalQueueEntry entry) {
            entries.remove(entry);
            if (cacheOrder != null) {
                cacheOrder.remove(entry);
                checkedVersion.remove(entry);
            }
        }

        private boolean hasStaleCachePlan() {
            return cacheOrder != null && !cacheOrder.isEmpty()
                    && checkedVersion.get(cacheOrder.first()) < cacheVersion;
        }

        private GlobalQueueEntry next() {
            return hasStaleCachePlan() ? cacheOrder.first() : entries.first();
        }
    }
}
