package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;
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
 * <p>It never owns or iterates requests. A worker change advances its group and role versions.
 * Listeners receive the exact worker for route invalidation; waiting requests retry selection
 * across the allowed group instead of retaining a worker-specific capacity promise.</p>
 */
@Component
public final class PlacementAvailability {
    private final AtomicLong sequence = new AtomicLong();
    private final AtomicLong forgottenThrough = new AtomicLong();
    private final ConcurrentMap<PlacementKey, Long> lastChanged = new ConcurrentHashMap<>();
    private final Set<Listener> listeners = ConcurrentHashMap.newKeySet();

    public void addListener(Listener candidate) {
        Objects.requireNonNull(candidate, "listener");
        listeners.add(candidate);
    }

    public void removeListener(Listener candidate) {
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
                worker,
                null);
    }

    public void changed(WorkerEndpoint endpoint) {
        var status = Objects.requireNonNull(endpoint, "endpoint").getStatus();
        changed(
                PlacementKey.exact(
                        status.getRole(), status.topologySnapshot().group(), endpoint.ipPort()),
                status,
                endpoint);
    }

    /**
     * The registry calls this after drain, while it still owns the exact status identity.
     * Shared group history can also be discarded: older plans retry using the floor,
     * and active members publish subsequent edges normally. No waiters are broadcast.
     */
    public void clearRetired(WorkerStatus worker) {
        checkState(!worker.isActiveGeneration(), "Cannot clear an active worker generation");
        discardHistory(
                new PlacementKey(worker.getRole(), worker.topologySnapshot().group(), null),
                sequence.incrementAndGet());
    }

    long sequence() {
        return sequence.get();
    }

    long lastChangedSequence(PlacementKey key) {
        long version = lastChanged.getOrDefault(key.waitingDomain(), 0L);
        return Math.max(version, forgottenThrough.get());
    }

    private void changed(PlacementKey key, WorkerStatus worker, WorkerEndpoint endpoint) {
        Objects.requireNonNull(key, "key");
        long next = sequence.incrementAndGet();
        if (!worker.isActiveGeneration()) {
            forgottenThrough.accumulateAndGet(next, Math::max);
        } else {
            // An older publisher may finish after retirement and replacement publication.
            lastChanged.merge(key.waitingDomain(), next, Math::max);
            if (key.group() != null) {
                lastChanged.merge(PlacementKey.anyGroup(key.role()), next, Math::max);
            }
            if (!worker.isActiveGeneration()) {
                discardHistory(key, next);
            }
        }
        // Preserve one callback per physical change, including the worker identity needed
        // to invalidate existing routes. Wait queues use only the role and group.
        for (Listener listener : listeners) {
            try {
                if (endpoint == null) listener.onAvailabilityChanged(key);
                else listener.onEndpointChanged(endpoint);
            } catch (Throwable failure) {
                Logger.warn("Placement availability listener failed", failure);
            }
        }
    }

    private void discardHistory(PlacementKey key, long through) {
        // Publish the floor before deletion so a plan cannot mistake a removed edge for no change.
        forgottenThrough.accumulateAndGet(through, Math::max);
        lastChanged.computeIfPresent(
                key.waitingDomain(), (ignored, version) -> version <= through ? null : version);
    }

    @FunctionalInterface
    public interface Listener {
        default void onEndpointChanged(WorkerEndpoint endpoint) {
            var status = endpoint.getStatus();
            onAvailabilityChanged(
                    PlacementKey.exact(
                            status.getRole(),
                            status.topologySnapshot().group(),
                            endpoint.ipPort()));
        }

        void onAvailabilityChanged(PlacementKey key);
    }
}
