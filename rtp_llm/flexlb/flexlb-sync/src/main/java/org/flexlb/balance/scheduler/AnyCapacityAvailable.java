package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;

import java.util.List;

/**
 * Wakes a queue when any of its blocked delivery candidates has capacity.
 */
record AnyCapacityAvailable(List<CapacityBoundary.Availability> sources) implements CapacityBoundary.Availability {
    AnyCapacityAvailable {
        sources = List.copyOf(sources);
    }

    @Override
    public boolean isAvailable() {
        return sources.stream().anyMatch(CapacityBoundary.Availability::isAvailable);
    }

    @Override
    public void addListener(Runnable listener) {
        for (var source : sources) {
            source.addListener(listener);
        }
    }

    @Override
    public void removeListener(Runnable listener) {
        for (var source : sources) {
            source.removeListener(listener);
        }
    }
}
