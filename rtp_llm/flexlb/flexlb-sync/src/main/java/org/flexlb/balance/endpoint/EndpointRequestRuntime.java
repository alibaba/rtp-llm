package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.SlotDeliveryPort;

/**
 * Endpoint-facing request runtime.
 *
 * <p>This is the only request-lifecycle capability exposed to endpoint
 * composition. The public scheduler facade is deliberately not an endpoint
 * callback target.
 */
public interface EndpointRequestRuntime
        extends DeliveryLifecyclePort, SlotDeliveryPort, EndpointEventSink {
}
