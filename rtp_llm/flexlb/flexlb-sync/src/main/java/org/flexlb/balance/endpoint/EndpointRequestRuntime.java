package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryLifecyclePort;
import org.flexlb.balance.delivery.SlotDeliveryPort;

/**
 * Endpoint-facing request runtime.
 *
 * <p>This is the only request-lifecycle capability exposed to endpoint
 * composition. The public scheduler facade is deliberately not an endpoint
 * callback target.
 *
 * <p>Stage-2 T7 S3: it also carries the
 * {@link DecodePlacementAuthorityPort} so a {@code DecodeEndpoint} can
 * project its decode-admission flips onto the slot-side authority through
 * the same injected runtime (the endpoint's constructor discovers the port
 * on its event sink; endpoints composed without a port keep the legacy
 * bare-flip semantics).
 */
public interface EndpointRequestRuntime
        extends DeliveryLifecyclePort, SlotDeliveryPort, EndpointEventSink,
        DecodePlacementAuthorityPort {
}
