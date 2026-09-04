package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
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

    record Event(PlacementKey key, long sequence, ChangeKind kind) {

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
        lastChanged.put(key, next);
        if (key.endpoint() != null) {
            lastChanged.put(new PlacementKey(key.role(), key.group()), next);
        }
        if (key.group() != null) {
            lastChanged.put(PlacementKey.anyGroup(key.role()), next);
        }
        Event event = new Event(key, next, kind);
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

    public void capacityChanged(RoleType role, String group) {
        capacityChanged(new PlacementKey(role, group));
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
