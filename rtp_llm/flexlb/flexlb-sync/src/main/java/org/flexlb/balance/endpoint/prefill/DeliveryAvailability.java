package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.delivery.DeliveryBlockReason;

/**
 * Wait condition for one more route to use a Prefill endpoint's dispatch capacity.
 */
final class DeliveryAvailability implements DeliveryBlockReason.Availability {
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
