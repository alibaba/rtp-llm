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

/**
 * Ordered waiters for exact endpoints and overlapping selector domains.
 *
 * <p>Each endpoint allows one waiting request to retry at a time:
 * a capacity notification wakes the first waiter; removing that request wakes
 * its successor; parking it on the same endpoint stops retries until the next
 * capacity notification. Waking a request does not reserve capacity.</p>
 */
final class BlockedRequestIndex {

    /** Exact endpoint and capacity key used when a request must wait. */
    record WaitTarget(WorkerEndpoint endpoint, PlacementKey key) {
    }

    private final Map<WorkerEndpoint, EndpointWaiters> waitersByEndpoint =
            new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, EndpointWaiters> waitersByRequest =
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

    WaitTarget findBlockingEndpoint(
            GlobalQueueEntry entry,
            RouteAdmission admission) {
        WaitTarget conflict = findBlockingEndpoint(entry, admission.prefillEndpoint());
        return conflict != null
                ? conflict
                : findBlockingEndpoint(entry, admission.decodeEndpoint());
    }

    /** Return the active retry's source; ordinary and still-waiting requests have none. */
    WaitTarget retrySource(GlobalQueueEntry entry) {
        EndpointWaiters waiters = waitersByRequest.get(entry);
        if (waiters == null || waiters.activeRetry != entry) {
            return null;
        }
        return new WaitTarget(waiters.endpoint, waiters.blocker);
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

        EndpointWaiters current = waitersByRequest.get(entry);
        if (current != null && current.endpoint == exactEndpoint) {
            current.blocker = exactBlocker;
            if (current.activeRetry == entry) {
                // Still full: put this request back to sleep without waking its successor.
                current.activeRetry = null;
            }
            entry.blockedKey = exactBlocker;
            entry.blockedEndpoint = exactEndpoint;
            return;
        }
        removeEndpointWaiterAndWakeNext(entry);

        EndpointWaiters target = waitersByEndpoint.computeIfAbsent(
                exactEndpoint,
                ignored -> register(new EndpointWaiters(
                        exactEndpoint, exactBlocker)));
        target.blocker = exactBlocker;
        target.entries.add(entry);
        waitersByRequest.put(entry, target);
        entry.blockedKey = exactBlocker;
        entry.blockedEndpoint = exactEndpoint;
    }

    void parkSelector(GlobalQueueEntry entry, PlacementKey blocker) {
        removeAndWakeNext(entry);
        entry.blockedKey = Objects.requireNonNull(blocker, "blocker");
        selectorWaiters.add(entry);
        selectorsByGroup.computeIfAbsent(entry.routingGroup,
                ignored -> new TreeSet<>(order)).add(entry);
    }

    /** Remove a waiter and wake its successor only if it was the active retry. */
    void removeAndWakeNext(GlobalQueueEntry entry) {
        removeEndpointWaiterAndWakeNext(entry);
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
     * Start one ordered retry chain. Multiple releases may share one notification;
     * successful admission advances the chain until a successor confirms it is full.
     * Repeated notifications do not wake a second request while a retry is active.
     */
    void capacityChanged(PlacementKey event) {
        releaseSelectors(event);
        forEachExactEndpoint(event, this::wakeNextWaiter);
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
                removeAndWakeNext(entry);
            }
        }
    }

    void clear() {
        for (GlobalQueueEntry entry : selectorWaiters) {
            entry.blockedKey = null;
        }
        selectorWaiters.clear();
        selectorsByGroup.clear();
        for (EndpointWaiters waiters : waitersByEndpoint.values()) {
            for (GlobalQueueEntry entry : waiters.entries) {
                entry.blockedKey = null;
                entry.blockedEndpoint = null;
            }
        }
        waitersByEndpoint.clear();
        waitersByRequest.clear();
        endpointsByAddress.clear();
    }

    private WaitTarget findBlockingEndpoint(
            GlobalQueueEntry entry,
            WorkerEndpoint endpoint) {
        EndpointWaiters waiters = waitersByEndpoint.get(endpoint);
        if (waiters == null || waiters.entries.isEmpty()
                || waiters.activeRetry == entry
                || order.compare(entry, waiters.entries.first()) < 0) {
            return null;
        }
        return new WaitTarget(endpoint, waiters.blocker);
    }

    private EndpointWaiters register(EndpointWaiters waiters) {
        endpointsByAddress.computeIfAbsent(
                        waiters.endpoint.ipPort(),
                        ignored -> new LinkedHashSet<>())
                .add(waiters.endpoint);
        return waiters;
    }

    private void removeEndpointWaiterAndWakeNext(GlobalQueueEntry entry) {
        EndpointWaiters waiters = waitersByRequest.remove(entry);
        if (waiters == null) {
            return;
        }
        boolean wasActiveRetry = waiters.activeRetry == entry;
        if (wasActiveRetry) {
            waiters.activeRetry = null;
        }
        waiters.entries.remove(entry);
        if (waiters.entries.isEmpty()) {
            unregister(waiters);
        } else if (wasActiveRetry) {
            wakeNextWaiter(waiters.endpoint);
        }
    }

    private void wakeNextWaiter(WorkerEndpoint endpoint) {
        EndpointWaiters waiters = waitersByEndpoint.get(endpoint);
        if (waiters == null || waiters.activeRetry != null) {
            return;
        }
        while (!waiters.entries.isEmpty()) {
            GlobalQueueEntry next = waiters.entries.iterator().next();
            if (next.removed || next.future.isDone()) {
                waitersByRequest.remove(next);
                waiters.entries.remove(next);
                continue;
            }
            waiters.activeRetry = next;
            next.blockedKey = null;
            next.blockedEndpoint = null;
            return;
        }
        unregister(waiters);
    }

    private void releaseAll(WorkerEndpoint endpoint) {
        EndpointWaiters waiters = waitersByEndpoint.get(endpoint);
        if (waiters == null) {
            return;
        }
        for (GlobalQueueEntry entry : waiters.entries) {
            waitersByRequest.remove(entry);
            entry.blockedKey = null;
            entry.blockedEndpoint = null;
        }
        waiters.entries.clear();
        waiters.activeRetry = null;
        unregister(waiters);
    }

    private void unregister(EndpointWaiters waiters) {
        waitersByEndpoint.remove(waiters.endpoint);
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
        private GlobalQueueEntry activeRetry;

        private EndpointWaiters(
                WorkerEndpoint endpoint,
                PlacementKey blocker) {
            this.endpoint = endpoint;
            this.blocker = blocker;
        }
    }
}
