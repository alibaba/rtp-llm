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
 * <p>It never owns or iterates requests. Exact-group changes also advance the
 * role-wide key used by requests whose routing policy has not selected a
 * group yet.</p>
 */
@Component
public final class PlacementAvailability {

    @FunctionalInterface
    interface Listener {
        void onCapacityChanged(PlacementKey key, long sequence);
    }

    private final AtomicLong sequence = new AtomicLong();
    private final ConcurrentMap<PlacementKey, Long> lastChanged =
            new ConcurrentHashMap<>();
    private volatile Listener listener;

    synchronized void addListener(Listener candidate) {
        Objects.requireNonNull(candidate, "listener");
        if (listener != null && listener != candidate) {
            throw new IllegalStateException(
                    "placement availability already has a listener");
        }
        listener = candidate;
    }

    synchronized void removeListener(Listener candidate) {
        if (listener == candidate) {
            listener = null;
        }
    }

    /** Notify that a fresh placement in this domain may now succeed. */
    public void capacityChanged(PlacementKey key) {
        Objects.requireNonNull(key, "key");
        publish(key);
        if (key.group() != null) {
            publish(PlacementKey.anyGroup(key.role()));
        }
    }

    public void capacityChanged(RoleType role, String group) {
        capacityChanged(new PlacementKey(role, group));
    }

    long sequence() {
        return sequence.get();
    }

    long lastChangedSequence(PlacementKey key) {
        return lastChanged.getOrDefault(key, 0L);
    }

    private void publish(PlacementKey key) {
        long next = sequence.incrementAndGet();
        lastChanged.put(key, next);
        Listener current = listener;
        if (current == null) {
            return;
        }
        try {
            current.onCapacityChanged(key, next);
        } catch (Throwable failure) {
            Logger.warn("Placement availability listener failed", failure);
        }
    }
}
