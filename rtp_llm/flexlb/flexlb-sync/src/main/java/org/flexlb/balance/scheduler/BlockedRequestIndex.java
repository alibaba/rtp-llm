package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;

import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.NavigableSet;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Consumer;

/**
 * Ordered endpoint waiters and request-local selector retry chains.
 *
 * <p>Each endpoint allows one waiting request to retry at a time:
 * a capacity notification wakes the first waiter; removing that request wakes
 * its successor unless that exact successor must still wait. Parking on the
 * same endpoint stops retries until the next capacity notification. Waking a
 * request does not reserve capacity.</p>
 */
final class BlockedRequestIndex {

    /** Exact endpoint and capacity key used when a request must wait. */
    record WaitTarget(WorkerEndpoint endpoint, PlacementKey key) {
    }

    /** Snapshot of the ordered successor whose capacity may be checked outside the queue lock. */
    record RetryContinuation(GlobalQueueEntry active, WaitTarget source, GlobalQueueEntry next) {
    }

    private final Map<WorkerEndpoint, EndpointWaiters> waitersByEndpoint =
            new IdentityHashMap<>();
    private final Map<GlobalQueueEntry, EndpointWaiters> waitersByRequest =
            new IdentityHashMap<>();
    private final Map<String, Set<WorkerEndpoint>> endpointsByAddress =
            new java.util.HashMap<>();
    private final boolean priorityOrdering;
    private final Comparator<GlobalQueueEntry> order;
    private final Consumer<GlobalQueueEntry> onWake;
    private final Map<GlobalQueueEntry, SelectorWait> selectorWaiters = new IdentityHashMap<>();
    private final Map<PlacementKey, SelectorDomain> selectorDomains = new HashMap<>();

    BlockedRequestIndex(boolean priorityOrdering) {
        this(priorityOrdering, ignored -> { });
    }

    BlockedRequestIndex(boolean priorityOrdering, Consumer<GlobalQueueEntry> onWake) {
        this.priorityOrdering = priorityOrdering;
        this.onWake = Objects.requireNonNull(onWake, "onWake");
        Comparator<GlobalQueueEntry> fifo = Comparator.comparingLong(entry -> entry.sequence);
        order = priorityOrdering
                ? Comparator.<GlobalQueueEntry>comparingInt(entry -> entry.priority).reversed()
                        .thenComparing(fifo)
                : fifo;
    }

    boolean isBlocked(GlobalQueueEntry entry) {
        return entry.blockedEndpoint != null || isSelectorBlocked(entry);
    }

    boolean isSelectorBlocked(GlobalQueueEntry entry) {
        SelectorWait waiter = selectorWaiters.get(entry);
        return waiter != null && !waiter.batch().ready;
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

    RetryContinuation retryContinuation(GlobalQueueEntry entry) {
        EndpointWaiters waiters = waitersByRequest.get(entry);
        if (waiters == null || waiters.activeRetry != entry) {
            return null;
        }
        GlobalQueueEntry next = firstLiveSuccessor(waiters, entry);
        return next == null ? null : new RetryContinuation(
                entry, new WaitTarget(waiters.endpoint, waiters.blocker), next);
    }

    void pauseContinuation(RetryContinuation candidate) {
        Objects.requireNonNull(candidate, "candidate");
        EndpointWaiters waiters = waitersByRequest.get(candidate.active());
        if (waiters != null && waiters.activeRetry == candidate.active()
                && waiters.endpoint == candidate.source().endpoint()
                && Objects.equals(waiters.blocker, candidate.source().key())
                && firstLiveSuccessor(waiters, candidate.active()) == candidate.next()) {
            waiters.pausedSuccessor = candidate.next();
        }
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

        clearSelector(entry);
        EndpointWaiters current = waitersByRequest.get(entry);
        if (current != null && current.endpoint == exactEndpoint) {
            current.blocker = exactBlocker;
            if (current.activeRetry == entry) {
                // Still full: put this request back to sleep without waking its successor.
                current.activeRetry = null;
                current.pausedSuccessor = null;
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
        SelectorDomain domain = selectorDomains.computeIfAbsent(blocker, ignored -> new SelectorDomain());
        selectorWaiters.put(entry, new SelectorWait(domain, domain.add(entry)));
    }

    /** Remove a waiter and preserve progress when the active retry or paused successor leaves. */
    void removeAndWakeNext(GlobalQueueEntry entry) {
        removeEndpointWaiterAndWakeNext(entry);
        clearSelector(entry);
        entry.blockedKey = null;
        entry.blockedEndpoint = null;
    }

    private void clearSelector(GlobalQueueEntry entry) {
        SelectorWait waiter = selectorWaiters.remove(entry);
        if (waiter == null) {
            return;
        }
        waiter.domain().remove(entry, waiter.batch());
        if (waiter.domain().size == 0) {
            selectorDomains.remove(entry.blockedKey, waiter.domain());
        }
    }

    /**
     * Start one ordered retry chain. Multiple releases may share one notification;
     * successful admission advances the chain until the next waiter must wait for capacity.
     * Repeated notifications do not wake a second request while a retry is active.
     */
    void capacityChanged(PlacementKey event) {
        releaseSelectors(event);
        forEachExactEndpoint(event, endpoint -> {
            EndpointWaiters waiters = waitersByEndpoint.get(endpoint);
            if (waiters != null) {
                waiters.pausedSuccessor = null;
                wakeNextWaiter(endpoint);
            }
        });
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
        if (event.endpoint() != null) {
            releaseSelectorDomain(event);
        }
        releaseSelectorDomain(PlacementKey.anyGroup(event.role()));
        if (event.group() != null) {
            releaseSelectorDomain(new PlacementKey(event.role(), event.group()));
        }
    }

    /** Release one ordered retry chain per waiting priority without walking its requests. */
    private void releaseSelectorDomain(PlacementKey key) {
        SelectorDomain domain = selectorDomains.get(key);
        if (domain != null) {
            domain.releaseWaitingBatches();
        }
    }

    void clear() {
        for (GlobalQueueEntry entry : selectorWaiters.keySet()) {
            entry.blockedKey = null;
        }
        selectorWaiters.clear();
        selectorDomains.clear();
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
        boolean wasPausedSuccessor = waiters.pausedSuccessor == entry;
        if (wasActiveRetry) {
            waiters.activeRetry = null;
        }
        waiters.entries.remove(entry);
        if (wasPausedSuccessor) {
            waiters.pausedSuccessor = null;
        }
        if (waiters.entries.isEmpty()) {
            unregister(waiters);
        } else if ((wasActiveRetry
                && (waiters.pausedSuccessor == null
                    || firstLiveSuccessor(waiters, null) != waiters.pausedSuccessor))
                || (wasPausedSuccessor && waiters.activeRetry == null)) {
            wakeNextWaiter(waiters.endpoint);
        }
    }

    private GlobalQueueEntry firstLiveSuccessor(EndpointWaiters waiters, GlobalQueueEntry active) {
        for (GlobalQueueEntry candidate : waiters.entries) {
            if (candidate != active && !candidate.removed && !candidate.future.isDone()) {
                return candidate;
            }
        }
        return null;
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
            waiters.pausedSuccessor = null;
            next.blockedKey = null;
            next.blockedEndpoint = null;
            onWake.accept(next);
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
            onWake.accept(entry);
        }
        waiters.entries.clear();
        waiters.activeRetry = null;
        waiters.pausedSuccessor = null;
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

    private final class SelectorDomain {
        private final Map<Integer, SelectorBatch> waitingBatches = new HashMap<>();
        private int size;

        SelectorBatch add(GlobalQueueEntry entry) {
            SelectorBatch batch = waitingBatches.computeIfAbsent(
                    priorityOrdering ? entry.priority : 0, ignored -> new SelectorBatch());
            batch.entries.add(entry);
            size++;
            return batch;
        }

        void remove(GlobalQueueEntry entry, SelectorBatch batch) {
            boolean wasFirst = batch.entries.first() == entry;
            batch.entries.remove(entry);
            size--;
            if (batch.entries.isEmpty()) {
                waitingBatches.remove(priorityOrdering ? entry.priority : 0, batch);
            } else if (batch.ready && wasFirst) {
                onWake.accept(batch.entries.first());
            }
        }

        void releaseWaitingBatches() {
            for (SelectorBatch batch : waitingBatches.values()) {
                batch.ready = true;
                onWake.accept(batch.entries.first());
            }
            waitingBatches.clear();
        }
    }

    /** Requests parked at one priority since the last capacity notification. */
    private final class SelectorBatch {
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private boolean ready;
    }

    private record SelectorWait(SelectorDomain domain, SelectorBatch batch) {
    }

    private final class EndpointWaiters {
        private final WorkerEndpoint endpoint;
        private final NavigableSet<GlobalQueueEntry> entries = new TreeSet<>(order);
        private PlacementKey blocker;
        private GlobalQueueEntry activeRetry;
        private GlobalQueueEntry pausedSuccessor;

        private EndpointWaiters(
                WorkerEndpoint endpoint,
                PlacementKey blocker) {
            this.endpoint = endpoint;
            this.blocker = blocker;
        }
    }
}
