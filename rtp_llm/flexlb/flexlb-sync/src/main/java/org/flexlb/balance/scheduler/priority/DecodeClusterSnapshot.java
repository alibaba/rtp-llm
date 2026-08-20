package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.FlexlbConfig;

import java.util.HashMap;
import java.util.Map;

/**
 * Detailed Decode cluster view captured only after a Decode-capacity route
 * failure, immediately before classification or eviction planning.
 *
 * <p>Keeping this type distinct from {@link ClusterSnapshot} prevents a
 * lightweight normal-path summary from accidentally reaching a decision that
 * requires the full reserved/accepted/running layers.
 */
public record DecodeClusterSnapshot(Map<String, DecodeEndpointSnapshot> decodes) {

    public static DecodeClusterSnapshot capture(EndpointRegistry registry,
                                                FlexlbConfig config) {
        Long configuredDecodeLimit = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        long decodeConcurrencyLimit = configuredDecodeLimit == null
                ? 0 : configuredDecodeLimit;
        Map<String, DecodeEndpointSnapshot> decodes = new HashMap<>();
        registry.getDecodeEndpoints().forEach((key, ep) ->
                decodes.put(key, DecodeEndpointSnapshot.capture(ep, decodeConcurrencyLimit)));
        return new DecodeClusterSnapshot(decodes);
    }
}
