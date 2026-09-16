package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;

import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/** Ordered waiters for exact endpoints and overlapping selector domains. */
final class BlockedRequestIndex {

    record Conflict(WorkerEndpoint endpoint, PlacementKey blocker) {
    }

    private final Map<WorkerEndpoint, EndpointWaiters> byEndpoint =
            new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, EndpointWaiters> membership =
            new IdentityHashMap<>();
    private final Map<String, Set<WorkerEndpoint>> endpointsByAddress =
            new java.util.HashMap<>();
    private final Comparator<GlobalQueueEntry> order;
    private final NavigableSet<GlobalQueueEntry> selectorWaiters;
    private final Map<String, NavigableSet<GlobalQueueEntry>> selectorsByGroup = new HashMap<>();

    BlockedRequestIndex(boolean priorityOrdering) {
        Comparator<GlobalQueueEntry> fifo = Comparator.comparingLong(entry -> entry.sequence);
        order = priorityOrdering
                ? Comparator.<GlobalQueueEntry>comparingInt(entry -> entry.priority).reversed()
                        .thenComparing(fifo)
                : fifo;
        selectorWaiters = new TreeSet<>(order);
    }

    boolean isBlocked(GlobalQueueEntry entry) {
        return entry.blockedEndpoint != null || isSelectorBlocked(entry);
    }

    boolean isSelectorBlocked(GlobalQueueEntry entry) {
        // Unbound requests can select any group. Explicit groups may bypass
        // only other explicit groups; wildcard blockers remain fleet-wide.
        return entry.routingGroup == null ? precedes(selectorWaiters, entry)
                : precedes(selectorsByGroup.get(null), entry)
                        || precedes(selectorsByGroup.get(entry.routingGroup), entry);
    }

    private boolean precedes(NavigableSet<GlobalQueueEntry> waiters, GlobalQueueEntry entry) {
        return waiters != null && !waiters.isEmpty()
                && order.compare(waiters.first(), entry) <= 0;
    }

    Conflict conflict(
            GlobalQueueEntry entry,
            QueueRouteAdmission admission) {
        Conflict conflict = conflict(entry, admission.selectedPrefillEndpoint());
        return conflict != null
                ? conflict
                : conflict(entry, admission.selectedDecodeEndpoint());
    }

    void parkExact(
            GlobalQueueEntry entry,
            PlacementKey blocker,
            WorkerEndpoint endpoint) {
        PlacementKey exactBlocker = Objects.requireNonNull(blocker, "blocker");
        WorkerEndpoint exactEndpoint = Objects.requireNonNull(
                endpoint, "endpoint");
        if (!Objects.equals(exactBlocker.endpoint(), exactEndpoint.ipPort())) {
            throw new IllegalArgumentException(
                    "exact blocker does not identify its endpoint");
        }

        EndpointWaiters current = membership.get(entry);
        if (current != null && current.endpoint == exactEndpoint) {
            current.blocker = exactBlocker;
            if (current.claimant == entry) {
                current.claimant = null;
            }
            entry.blockedKey = exactBlocker;
            entry.blockedEndpoint = exactEndpoint;
            return;
        }
        detachEntry(entry, true);

        EndpointWaiters target = byEndpoint.computeIfAbsent(
                exactEndpoint,
                ignored -> register(new EndpointWaiters(
                        exactEndpoint, exactBlocker)));
        target.blocker = exactBlocker;
        target.entries.add(entry);
        membership.put(entry, target);
        entry.blockedKey = exactBlocker;
        entry.blockedEndpoint = exactEndpoint;
    }

    void parkSelector(GlobalQueueEntry entry, PlacementKey blocker) {
        clearEntry(entry);
        entry.blockedKey = Objects.requireNonNull(blocker, "blocker");
        selectorWaiters.add(entry);
        selectorsByGroup.computeIfAbsent(entry.routingGroup,
                ignored -> new TreeSet<>(order)).add(entry);
    }

    void clearEntry(GlobalQueueEntry entry) {
        detachEntry(entry, true);
        clearSelector(entry);
        entry.blockedKey = null;
        entry.blockedEndpoint = null;
    }

    private void clearSelector(GlobalQueueEntry entry) {
        if (!selectorWaiters.remove(entry)) {
            return;
        }
        NavigableSet<GlobalQueueEntry> group = selectorsByGroup.get(entry.routingGroup);
        group.remove(entry);
        if (group.isEmpty()) {
            selectorsByGroup.remove(entry.routingGroup);
        }
    }

    /**
     * Retire an activated claimant after publication. A commit to the same
     * endpoint consumed the capacity edge, so its successor remains parked.
     * A commit on another endpoint leaves the original opportunity unused and
     * therefore advances the next waiter.
     */
    void routeCommitted(
            GlobalQueueEntry entry,
            QueueRouteAdmission admission) {
        EndpointWaiters waiters = membership.get(entry);
        if (waiters == null) {
            clearEntry(entry);
            return;
        }
        boolean consumedClaim = waiters.claimant == entry
                && selects(admission, waiters.endpoint);
        detachEntry(entry, !consumedClaim);
        clearSelector(entry);
        entry.blockedKey = null;
        entry.blockedEndpoint = null;
    }

    /**
     * One physical capacity edge releases one ordered claimant. Once that
     * claimant finishes or changes route, the next waiter is tried. This
     * bounds a one-slot release to one success plus at most one confirming
     * miss instead of replanning every request parked on the endpoint.
     */
    void capacityChanged(PlacementKey event) {
        releaseSelectors(event);
        forEachExactEndpoint(event, this::activateNext);
    }

    /** A generation change invalidates every route pinned to that address. */
    void topologyChanged(PlacementKey event) {
        releaseSelectors(event);
        forEachExactEndpoint(event, this::releaseAll);
    }

    private void releaseSelectors(PlacementKey event) {
        if (event == null) {
            return;
        }
        releaseSelectors(selectorsByGroup.get(null), event);
        if (event.group() != null) {
            releaseSelectors(selectorsByGroup.get(event.group()), event);
        }
    }

    private void releaseSelectors(NavigableSet<GlobalQueueEntry> entries, PlacementKey event) {
        if (entries == null) {
            return;
        }
        for (GlobalQueueEntry entry : List.copyOf(entries)) {
            if (isRelevant(entry.blockedKey, event)) {
                clearEntry(entry);
            }
        }
    }

    void clear() {
        for (GlobalQueueEntry entry : selectorWaiters) {
            entry.blockedKey = null;
        }
        selectorWaiters.clear();
        selectorsByGroup.clear();
        for (EndpointWaiters waiters : byEndpoint.values()) {
            for (GlobalQueueEntry entry : waiters.entries) {
                entry.blockedKey = null;
                entry.blockedEndpoint = null;
            }
        }
        byEndpoint.clear();
        membership.clear();
        endpointsByAddress.clear();
    }

    private Conflict conflict(
            GlobalQueueEntry entry,
            WorkerEndpoint endpoint) {
        EndpointWaiters waiters = byEndpoint.get(endpoint);
        if (waiters == null || waiters.entries.isEmpty()
                || waiters.claimant == entry
                || order.compare(entry, waiters.entries.first()) < 0) {
            return null;
        }
        return new Conflict(endpoint, waiters.blocker);
    }

    private static boolean selects(
            QueueRouteAdmission admission,
            WorkerEndpoint endpoint) {
        return admission.selectedPrefillEndpoint() == endpoint
                || admission.selectedDecodeEndpoint() == endpoint;
    }

    private EndpointWaiters register(EndpointWaiters waiters) {
        endpointsByAddress.computeIfAbsent(
                        waiters.endpoint.ipPort(),
                        ignored -> new LinkedHashSet<>())
                .add(waiters.endpoint);
        return waiters;
    }

    private void detachEntry(GlobalQueueEntry entry, boolean advanceClaimant) {
        EndpointWaiters waiters = membership.remove(entry);
        if (waiters == null) {
            return;
        }
        boolean wasClaimant = waiters.claimant == entry;
        if (wasClaimant) {
            waiters.claimant = null;
        }
        waiters.entries.remove(entry);
        if (waiters.entries.isEmpty()) {
            unregister(waiters);
        } else if (wasClaimant && advanceClaimant) {
            activateNext(waiters.endpoint);
        }
    }

    private void activateNext(WorkerEndpoint endpoint) {
        EndpointWaiters waiters = byEndpoint.get(endpoint);
        if (waiters == null || waiters.claimant != null) {
            return;
        }
        while (!waiters.entries.isEmpty()) {
            GlobalQueueEntry next = waiters.entries.iterator().next();
            if (next.removed || next.future.isDone()) {
                membership.remove(next);
                waiters.entries.remove(next);
                continue;
            }
            waiters.claimant = next;
            next.blockedKey = null;
            next.blockedEndpoint = null;
            return;
        }
        unregister(waiters);
    }

    private void releaseAll(WorkerEndpoint endpoint) {
        EndpointWaiters waiters = byEndpoint.get(endpoint);
        if (waiters == null) {
            return;
        }
        for (GlobalQueueEntry entry : waiters.entries) {
            membership.remove(entry);
            entry.blockedKey = null;
            entry.blockedEndpoint = null;
        }
        waiters.entries.clear();
        waiters.claimant = null;
        unregister(waiters);
    }

    private void unregister(EndpointWaiters waiters) {
        byEndpoint.remove(waiters.endpoint);
        Set<WorkerEndpoint> endpoints =
                endpointsByAddress.get(waiters.endpoint.ipPort());
        if (endpoints == null) {
            return;
        }
        endpoints.remove(waiters.endpoint);
        if (endpoints.isEmpty()) {
            endpointsByAddress.remove(waiters.endpoint.ipPort());
        }
    }

    private void forEachExactEndpoint(
            PlacementKey event,
            java.util.function.Consumer<WorkerEndpoint> action) {
        if (event == null || event.endpoint() == null) {
            return;
        }
        Set<WorkerEndpoint> endpoints = endpointsByAddress.get(event.endpoint());
        if (endpoints == null || endpoints.isEmpty()) {
            return;
        }
        for (WorkerEndpoint endpoint : Set.copyOf(endpoints)) {
            action.accept(endpoint);
        }
    }

    private static boolean isRelevant(
            PlacementKey blocker,
            PlacementKey event) {
        if (blocker == null || event == null
                || blocker.role() != event.role()) {
            return false;
        }
        if (blocker.endpoint() != null) {
            return Objects.equals(blocker.endpoint(), event.endpoint());
        }
        return Objects.equals(blocker.group(), event.group())
                || blocker.group() == null;
    }

    private final class EndpointWaiters {
        private final WorkerEndpoint endpoint;
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private PlacementKey blocker;
        private GlobalQueueEntry claimant;

        private EndpointWaiters(
                WorkerEndpoint endpoint,
                PlacementKey blocker) {
            this.endpoint = endpoint;
            this.blocker = blocker;
        }
    }
}
