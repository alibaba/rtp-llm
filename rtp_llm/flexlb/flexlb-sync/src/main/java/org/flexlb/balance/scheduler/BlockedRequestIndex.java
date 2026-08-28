package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;

import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Exact endpoint waiters plus the single non-bypassable selector frontier. */
final class BlockedRequestIndex {

    record Conflict(WorkerEndpoint endpoint, PlacementKey blocker) {
    }

    private final Map<WorkerEndpoint, EndpointWaiters> byEndpoint =
            new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, EndpointWaiters> membership =
            new IdentityHashMap<>();
    private final Map<String, Set<WorkerEndpoint>> endpointsByAddress =
            new java.util.HashMap<>();
    private GlobalQueueEntry frontier;

    boolean isExactBlocked(GlobalQueueEntry entry) {
        return entry.blockedEndpoint != null;
    }

    GlobalQueueEntry frontier() {
        return frontier;
    }

    void clearStaleFrontier() {
        if (frontier != null
                && (frontier.removed || frontier.future.isDone())) {
            clearEntry(frontier);
        }
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

    void parkFrontier(GlobalQueueEntry entry, PlacementKey blocker) {
        clearEntry(entry);
        if (frontier != null && frontier != entry) {
            throw new IllegalStateException(
                    "selector frontier already belongs to another request");
        }
        entry.blockedKey = Objects.requireNonNull(blocker, "blocker");
        frontier = entry;
    }

    void clearEntry(GlobalQueueEntry entry) {
        detachEntry(entry, true);
        if (frontier == entry) {
            frontier = null;
        }
        entry.blockedKey = null;
        entry.blockedEndpoint = null;
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
        if (frontier == entry) {
            frontier = null;
        }
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
        if (frontier != null && isRelevant(frontier.blockedKey, event)) {
            clearEntry(frontier);
        }
        forEachExactEndpoint(event, this::activateNext);
    }

    /** A generation change invalidates every route pinned to that address. */
    void topologyChanged(PlacementKey event) {
        if (frontier != null && isRelevant(frontier.blockedKey, event)) {
            clearEntry(frontier);
        }
        forEachExactEndpoint(event, this::releaseAll);
    }

    void clear() {
        if (frontier != null) {
            frontier.blockedKey = null;
            frontier = null;
        }
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
                || waiters.claimant == entry) {
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

    private static final class EndpointWaiters {
        private final WorkerEndpoint endpoint;
        private final LinkedHashSet<GlobalQueueEntry> entries =
                new LinkedHashSet<>();
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
