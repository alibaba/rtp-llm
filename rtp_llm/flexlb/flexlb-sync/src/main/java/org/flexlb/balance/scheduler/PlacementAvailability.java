package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.Set;
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

    @FunctionalInterface
    interface Listener {
        void onAvailabilityChanged(PlacementKey key);
    }

    private final AtomicLong sequence = new AtomicLong();
    private final ConcurrentMap<PlacementKey, Long> lastChanged =
            new ConcurrentHashMap<>();
    private final Set<Listener> listeners = ConcurrentHashMap.newKeySet();

    void addListener(Listener candidate) {
        Objects.requireNonNull(candidate, "listener");
        listeners.add(candidate);
    }

    void removeListener(Listener candidate) {
        if (candidate != null) {
            listeners.remove(candidate);
        }
    }

    /** Notify that capacity or endpoint topology changed in this placement domain. */
    public void changed(PlacementKey key) {
        Objects.requireNonNull(key, "key");
        long next = sequence.incrementAndGet();
        // Publishers may reach these keys out of sequence; every edge retains
        // the newest version even when an older publication finishes later.
        lastChanged.merge(key.capacityDomain(), next, Math::max);
        if (key.endpoint() != null) {
            lastChanged.merge(new PlacementKey(key.role(), key.group(), null), next, Math::max);
        }
        if (key.group() != null) {
            lastChanged.merge(PlacementKey.anyGroup(key.role()), next, Math::max);
        }
        // One physical capacity edge produces one callback. The exact key is
        // sufficient for group/role waiters through their relevance match and
        // avoids three global-lock acquisitions for every endpoint release.
        for (Listener listener : listeners) {
            try {
                listener.onAvailabilityChanged(key);
            } catch (Throwable failure) {
                Logger.warn(
                        "Placement availability listener failed", failure);
            }
        }
    }

    public void changed(RoleType role, String group, String endpoint) {
        changed(PlacementKey.exact(role, group, endpoint));
    }

    long sequence() {
        return sequence.get();
    }

    long lastChangedSequence(PlacementKey key) {
        return lastChanged.getOrDefault(key.capacityDomain(), 0L);
    }

}
