package org.flexlb.balance.scheduler;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.base.Preconditions.checkState;

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
    private final AtomicLong forgottenThrough = new AtomicLong();
    private final ConcurrentMap<PlacementKey, Long> lastChanged = new ConcurrentHashMap<>();
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

    /**
     * Publish one worker's edge without retaining history from a retired generation.
     */
    public void changed(WorkerStatus worker, String address) {
        changed(
                PlacementKey.exact(worker.getRole(), worker.topologySnapshot().group(), address),
                worker);
    }

    /**
     * The registry calls this after drain, while it still owns the exact status identity.
     * Shared group history can also be discarded: older plans retry using the floor,
     * and active members publish subsequent edges normally. No waiters are broadcast.
     */
    public void clearRetired(WorkerStatus worker, String address) {
        checkState(!worker.isActiveGeneration(), "Cannot clear an active worker generation");
        discardHistory(
                PlacementKey.exact(worker.getRole(), worker.topologySnapshot().group(), address),
                sequence.incrementAndGet());
    }

    public void changed(PlacementKey key) {
        changed(key, null);
    }

    private void changed(PlacementKey key, WorkerStatus worker) {
        Objects.requireNonNull(key, "key");
        long next = sequence.incrementAndGet();
        if (worker != null && !worker.isActiveGeneration()) {
            forgottenThrough.accumulateAndGet(next, Math::max);
        } else {
            // An older publisher may finish after retirement and replacement publication.
            lastChanged.merge(key.capacityDomain(), next, Math::max);
            if (key.endpoint() != null) {
                lastChanged.merge(new PlacementKey(key.role(), key.group(), null), next, Math::max);
            }
            if (key.group() != null) {
                lastChanged.merge(PlacementKey.anyGroup(key.role()), next, Math::max);
            }
            if (worker != null && !worker.isActiveGeneration()) {
                discardHistory(key, next);
            }
        }
        // One physical capacity edge produces one callback. The exact key is
        // sufficient for group/role waiters through their relevance match and
        // avoids three global-lock acquisitions for every endpoint release.
        for (Listener listener : listeners) {
            try {
                listener.onAvailabilityChanged(key);
            } catch (Throwable failure) {
                Logger.warn("Placement availability listener failed", failure);
            }
        }
    }

    private void discardHistory(PlacementKey key, long through) {
        // Publish the floor before deletion so a plan cannot mistake a removed edge for no change.
        forgottenThrough.accumulateAndGet(through, Math::max);
        lastChanged.computeIfPresent(
                key.capacityDomain(), (ignored, version) -> version <= through ? null : version);
        if (key.group() != null) {
            lastChanged.computeIfPresent(
                    new PlacementKey(key.role(), key.group(), null),
                    (ignored, version) -> version <= through ? null : version);
        }
    }

    long sequence() {
        return sequence.get();
    }

    long lastChangedSequence(PlacementKey key) {
        long version = lastChanged.getOrDefault(key.capacityDomain(), 0L);
        return Math.max(version, forgottenThrough.get());
    }
}
