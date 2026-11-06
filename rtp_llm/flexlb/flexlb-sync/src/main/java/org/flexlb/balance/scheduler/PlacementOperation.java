package org.flexlb.balance.scheduler;

import java.util.Objects;

/**
 * Stable identity of one placement attempt; its current facts live in RequestContext.
 * Context compares attempts by reference, so equal record values cannot complete another attempt.
 */
public record PlacementOperation(RequestContext owner, RouteExecution withdrawingRoute)
        implements AutoCloseable {
    public PlacementOperation {
        Objects.requireNonNull(owner, "owner");
    }

    @Override
    public void close() {
        owner.requirePlacementCompletion(this);
    }
}
