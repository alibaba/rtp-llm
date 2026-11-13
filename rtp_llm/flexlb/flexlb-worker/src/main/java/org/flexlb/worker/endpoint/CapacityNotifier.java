package org.flexlb.worker.endpoint;

import org.flexlb.util.Logger;
import org.flexlb.worker.status.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

import static com.google.common.base.Preconditions.checkState;

/**
 * Notifies listeners when worker capacity or availability changes and records role/group versions.
 * Waiting queues compare these versions to avoid missing a change while registering a request.
 * A notification prompts another capacity check; it does not reserve capacity for the request.
 */
@Component
public final class CapacityNotifier {
    private final AtomicLong sequence = new AtomicLong();
    private final AtomicLong forgottenThrough = new AtomicLong();
    private final ConcurrentMap<CapacityKey, Long> lastChanged = new ConcurrentHashMap<>();
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
     * Publish one worker's edge without retaining history from a closed generation.
     */
    public void notifyChanged(WorkerStatus worker, String address) {
        notifyChanged(
                CapacityKey.exact(worker.getRole(), worker.topologySnapshot().group(), address),
                worker,
                null);
    }

    public void notifyChanged(WorkerEndpoint endpoint) {
        var status = Objects.requireNonNull(endpoint, "endpoint").getStatus();
        notifyChanged(
                CapacityKey.exact(
                        status.getRole(), status.topologySnapshot().group(), endpoint.ipPort()),
                status,
                endpoint);
    }

    /**
     * The registry calls this after drain, while it still owns the exact status identity.
     * Shared group history can also be discarded: older plans retry using the floor,
     * and active members publish subsequent edges normally. No waiters are broadcast.
     */
    public void removeClosedWorker(WorkerStatus worker) {
        checkState(!worker.isActiveGeneration(), "Cannot clear an active worker generation");
        discardHistory(
                new CapacityKey(worker.getRole(), worker.topologySnapshot().group(), null),
                sequence.incrementAndGet());
    }

    public long sequence() {
        return sequence.get();
    }

    public long lastChangedSequence(CapacityKey key) {
        long version = lastChanged.getOrDefault(key.waitingDomain(), 0L);
        return Math.max(version, forgottenThrough.get());
    }

    private void notifyChanged(CapacityKey key, WorkerStatus worker, WorkerEndpoint endpoint) {
        Objects.requireNonNull(key, "key");
        long next = sequence.incrementAndGet();
        if (!worker.isActiveGeneration()) {
            forgottenThrough.accumulateAndGet(next, Math::max);
        } else {
            // An older publisher may finish after cleanup and replacement publication.
            lastChanged.merge(key.waitingDomain(), next, Math::max);
            if (key.group() != null) {
                lastChanged.merge(CapacityKey.anyGroup(key.role()), next, Math::max);
            }
            if (!worker.isActiveGeneration()) {
                discardHistory(key, next);
            }
        }
        // Preserve one callback per physical change, including the worker identity needed
        // to invalidate existing routes. Wait queues use only the role and group.
        for (Listener listener : listeners) {
            try {
                if (endpoint == null) listener.onCapacityChanged(key);
                else listener.onEndpointChanged(endpoint);
            } catch (Throwable failure) {
                Logger.warn("Placement capacityNotifier listener failed", failure);
            }
        }
    }

    private void discardHistory(CapacityKey key, long through) {
        // Publish the floor before deletion so a plan cannot mistake a removed edge for no change.
        forgottenThrough.accumulateAndGet(through, Math::max);
        lastChanged.computeIfPresent(
                key.waitingDomain(), (ignored, version) -> version <= through ? null : version);
    }

    /**
     * Receives capacity changes, including the exact endpoint when available.
     */
    @FunctionalInterface
    public interface Listener {
        default void onEndpointChanged(WorkerEndpoint endpoint) {
            var status = endpoint.getStatus();
            onCapacityChanged(
                    CapacityKey.exact(
                            status.getRole(),
                            status.topologySnapshot().group(),
                            endpoint.ipPort()));
        }

        void onCapacityChanged(CapacityKey key);
    }
}
