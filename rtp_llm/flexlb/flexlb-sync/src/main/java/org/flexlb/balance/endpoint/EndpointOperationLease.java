package org.flexlb.balance.endpoint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;

/**
 * A deadlock-free READY-operation lease across one or more endpoint generations.
 *
 * <p>Locks are acquired in {@link EndpointId} order and released in reverse.
 * Retirement first changes READY to RETIRING and then waits for the endpoint's
 * operation write barrier, so a transaction either obtains every READY lease
 * before retirement, or performs no endpoint side effect.
 */
public final class EndpointOperationLease implements AutoCloseable {

    private final List<WorkerEndpoint> endpoints;
    private boolean closed;

    private EndpointOperationLease(List<WorkerEndpoint> endpoints) {
        this.endpoints = endpoints;
    }

    public static Optional<EndpointOperationLease> acquire(WorkerEndpoint... endpoints) {
        return acquire(Arrays.asList(endpoints));
    }

    public static Optional<EndpointOperationLease> acquire(Collection<? extends WorkerEndpoint> candidates) {
        IdentityHashMap<WorkerEndpoint, Boolean> unique = new IdentityHashMap<>();
        for (WorkerEndpoint endpoint : candidates) {
            if (endpoint != null) {
                unique.put(endpoint, Boolean.TRUE);
            }
        }
        List<WorkerEndpoint> ordered = new ArrayList<>(unique.keySet());
        ordered.sort(Comparator.comparing(WorkerEndpoint::getEndpointId));

        List<WorkerEndpoint> acquired = new ArrayList<>(ordered.size());
        for (WorkerEndpoint endpoint : ordered) {
            endpoint.lockOperationRead();
            acquired.add(endpoint);
        }
        for (WorkerEndpoint endpoint : acquired) {
            if (!endpoint.isReady()) {
                unlockReverse(acquired);
                return Optional.empty();
            }
        }
        return Optional.of(new EndpointOperationLease(List.copyOf(acquired)));
    }

    public boolean contains(WorkerEndpoint endpoint) {
        return endpoints.stream().anyMatch(candidate -> candidate == endpoint);
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        unlockReverse(endpoints);
    }

    private static void unlockReverse(List<WorkerEndpoint> endpoints) {
        for (int i = endpoints.size() - 1; i >= 0; i--) {
            endpoints.get(i).unlockOperationRead();
        }
    }
}
