package org.flexlb.balance.composition;

import org.flexlb.balance.delivery.BatchDeliveryStrategy;
import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.delivery.DeliveryTelemetry;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.PrefillDeliveryStrategyBinding;

/** Composition of the batch transport with its exact admission resources. */
public final class BatchDeliveryBinding
        implements PrefillDeliveryStrategyBinding {

    private final DeliveryStrategy strategy;

    public BatchDeliveryBinding(
            BatchSubmissionPort submission,
            PrefillAdmissionPort admission,
            SlotDeliveryPort slots,
            DeliveryTelemetry telemetry) {
        this.strategy = new BatchDeliveryStrategy(
                submission, admission, slots, telemetry);
    }

    @Override
    public DeliveryStrategy strategy() {
        return strategy;
    }
}
