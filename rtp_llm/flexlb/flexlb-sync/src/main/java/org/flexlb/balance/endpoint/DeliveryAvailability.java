package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;

/**
 * Wait condition for one more route to use a Prefill endpoint's dispatch capacity.
 */
final class DeliveryAvailability implements CapacityBoundary.Availability {
    private final PrefillState prefillState;
    private final long maxDispatchLimit;

    DeliveryAvailability(PrefillState prefillState, long maxDispatchLimit) {
        this.prefillState = prefillState;
        this.maxDispatchLimit = maxDispatchLimit;
    }

    @Override
    public boolean isAvailable() {
        return prefillState.canDeliver(maxDispatchLimit, 1);
    }

    @Override
    public void addListener(Runnable listener) {
        // PrefillState already notifies its WorkerBatcher when delivery capacity changes.
    }

    @Override
    public void removeListener(Runnable listener) {
    }
}
