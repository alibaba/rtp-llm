package org.flexlb.balance.composition;

import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTelemetry;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.delivery.RouteDeliveryStrategy;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.PrefillDeliveryStrategyBinding;

/** Composition of direct route delivery with its exact admission resources. */
public final class RouteDeliveryBinding
        implements PrefillDeliveryStrategyBinding {

    private final DeliveryStrategy strategy;

    public RouteDeliveryBinding(
            PrefillAdmissionPort admission,
            SlotDeliveryPort slots,
            DeliveryTelemetry telemetry) {
        this.strategy = new RouteDeliveryStrategy(
                admission, slots, telemetry);
    }

    @Override
    public DeliveryStrategy strategy() {
        return strategy;
    }
}
