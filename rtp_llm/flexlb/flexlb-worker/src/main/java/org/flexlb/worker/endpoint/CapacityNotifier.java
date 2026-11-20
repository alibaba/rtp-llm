package org.flexlb.worker.endpoint;

import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Broadcasts endpoint changes outside resource locks. Listeners recheck current capacity;
 * the notifier stores neither request waiters nor capacity history.
 */
@Component
public final class CapacityNotifier {
    private final Set<Listener> listeners = ConcurrentHashMap.newKeySet();

    public void addListener(Listener listener) {
        listeners.add(Objects.requireNonNull(listener, "listener"));
    }

    public void removeListener(Listener listener) {
        if (listener != null) listeners.remove(listener);
    }

    public void notifyChanged(WorkerEndpoint endpoint) {
        Objects.requireNonNull(endpoint, "endpoint");
        for (var listener : listeners) {
            try {
                listener.onEndpointChanged(endpoint);
            } catch (Throwable failure) {
                Logger.warn("Capacity notification listener failed", failure);
            }
        }
    }

    /**
     * Receives a capacity change outside endpoint resource locks and rechecks current availability.
     */
    @FunctionalInterface
    public interface Listener {
        void onEndpointChanged(WorkerEndpoint endpoint);
    }
}
