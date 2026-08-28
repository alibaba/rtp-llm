package org.flexlb.balance.delivery;

import java.util.List;

/**
 * Reporting boundary for already-committed delivery decisions.
 *
 * <p>Implementations must contain and report their own failures. Neither
 * method may throw into delivery or alter an already-committed business
 * outcome.
 */
public interface DeliveryTelemetry {

    void routesDelivered(
            DeliveryMetadata metadata,
            List<DeliveryItem> exactItems);

    void batchDispatched(
            long batchId,
            DeliveryMetadata metadata,
            List<DeliveryItem> dispatched,
            long predictedMs);
}
