package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Versioned, O(1) notification edge for logical placement capacity.
 *
 * <p>It never owns or iterates requests. Exact-endpoint changes also advance
 * their group and role-wide keys, while exact waiters consume only the exact
 * edge. This prevents one worker's release from waking requests parked on
 * every worker in the same group.</p>
 */
@Component
public final class PlacementAvailability {

    enum ChangeKind {
        CAPACITY,
        TOPOLOGY
    }

    record Event(PlacementKey key, ChangeKind kind) {

        Event {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(kind, "kind");
        }
    }

    @FunctionalInterface
    interface Listener {
        void onAvailabilityChanged(Event event);
    }

    private final AtomicLong sequence = new AtomicLong();
    private final ConcurrentMap<PlacementKey, Long> lastChanged =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<Listener, Boolean> listeners =
            new ConcurrentHashMap<>();

    private static final class GroupChanges {
        private final AtomicLong version = new AtomicLong();
        private final Set<Runnable> listeners = ConcurrentHashMap.newKeySet();

        void signal() {
            version.incrementAndGet();
            for (Runnable listener : listeners) {
                try {
                    listener.run();
                } catch (Throwable failure) {
                    Logger.warn("Group capacity listener failed", failure);
                }
            }
        }
    }

    // Priority zero observes only actual capacity. A new victim may wake strictly higher priorities.
    private final ConcurrentMap<PlacementKey, ConcurrentSkipListMap<Integer, GroupChanges>> dispatchGroups =
            new ConcurrentHashMap<>();

    /** Capture before trying candidates so an edge before listener registration is still observable. */
    public CapacityBoundary.Availability observeGroup(RoleType role, String group, int preemptionPriority) {
        var priorities = dispatchGroups.computeIfAbsent(new PlacementKey(role, group),
                ignored -> new ConcurrentSkipListMap<>());
        GroupChanges changes = priorities.computeIfAbsent(preemptionPriority, ignored -> new GroupChanges());
        long observed = changes.version.get();
        return new CapacityBoundary.Availability() {
            @Override public boolean isAvailable() { return changes.version.get() != observed; }
            @Override public void addListener(Runnable listener) { changes.listeners.add(listener); }
            @Override public void removeListener(Runnable listener) { changes.listeners.remove(listener); }
        };
    }

    /** Engine-facing ownership was published without releasing capacity. */
    public void decodeVictimAvailable(String group, int victimPriority) {
        if (PriorityNormalizer.hasPriority(victimPriority)) {
            signalGroups(RoleType.DECODE, group, victimPriority);
        }
    }

    private void signalGroups(RoleType role, String group, int victimPriority) {
        if (dispatchGroups.isEmpty()) { return; }
        signalGroup(new PlacementKey(role, group), victimPriority);
        if (group != null) { signalGroup(PlacementKey.anyGroup(role), victimPriority); }
    }

    private void signalGroup(PlacementKey key, int victimPriority) {
        var priorities = dispatchGroups.get(key);
        if (priorities == null) { return; }
        var eligible = PriorityNormalizer.hasPriority(victimPriority)
                ? priorities.tailMap(victimPriority, false) : priorities;
        eligible.values().forEach(GroupChanges::signal);
    }

    void addListener(Listener candidate) {
        Objects.requireNonNull(candidate, "listener");
        listeners.put(candidate, Boolean.TRUE);
    }

    void removeListener(Listener candidate) {
        if (candidate != null) {
            listeners.remove(candidate);
        }
    }

    /** Notify that a fresh placement in this domain may now succeed. */
    public void capacityChanged(PlacementKey key) {
        publish(key, ChangeKind.CAPACITY);
    }

    /** Notify that an endpoint generation was published, replaced, or removed. */
    public void topologyChanged(PlacementKey key) {
        publish(key, ChangeKind.TOPOLOGY);
    }

    private void publish(PlacementKey key, ChangeKind kind) {
        Objects.requireNonNull(key, "key");
        long next = sequence.incrementAndGet();
        // Publishers may reach these keys out of sequence; every edge retains
        // the newest version even when an older publication finishes later.
        lastChanged.merge(key, next, Math::max);
        if (key.endpoint() != null) {
            // Exact waiters follow role/address across topology group changes.
            lastChanged.merge(PlacementKey.exact(key.role(), null, key.endpoint()), next, Math::max);
            lastChanged.merge(new PlacementKey(key.role(), key.group()), next, Math::max);
        }
        if (key.group() != null) {
            lastChanged.merge(PlacementKey.anyGroup(key.role()), next, Math::max);
        }
        signalGroups(key.role(), key.group(), PriorityNormalizer.NO_PRIORITY);
        Event event = new Event(key, kind);
        // One physical capacity edge produces one callback. The exact key is
        // sufficient for group/role waiters through their relevance match and
        // avoids three global-lock acquisitions for every endpoint release.
        for (Listener listener : listeners.keySet()) {
            try {
                listener.onAvailabilityChanged(event);
            } catch (Throwable failure) {
                Logger.warn(
                        "Placement availability listener failed", failure);
            }
        }
    }

    public void capacityChanged(
            RoleType role,
            String group,
            String endpoint) {
        capacityChanged(PlacementKey.exact(role, group, endpoint));
    }

    public void topologyChanged(
            RoleType role,
            String group,
            String endpoint) {
        topologyChanged(PlacementKey.exact(role, group, endpoint));
    }

    long sequence() {
        return sequence.get();
    }

    long lastChangedSequence(PlacementKey key) {
        return lastChanged.getOrDefault(key, 0L);
    }

}
