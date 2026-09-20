package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.FlexlbConfig;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Endpoint observations used for eviction planning.
 * Snapshots do not reserve capacity; commit must revalidate selected victims.
 *
 * @param prefills prefill endpoint snapshots keyed by "ip:httpPort"
 * @param decodes  decode endpoint snapshots keyed by "ip:httpPort"
 */
public record ClusterSnapshot(
        Map<String, PrefillEndpointSnapshot> prefills,
        Map<String, DecodeEndpointSnapshot> decodes) {

    /** Capture victim candidates only when entering eviction planning. */
    public static ClusterSnapshot captureDecode(EndpointRegistry registry, String group, long concurrencyLimit) {
        Map<String, DecodeEndpointSnapshot> decodes = new HashMap<>();
        registry.getDecodeEndpoints().forEach((id, endpoint) -> {
            if ((group == null || group.equals(endpoint.getStatus().getGroup())) && endpoint.getStatus().isAlive()) {
                decodes.put(id, DecodeEndpointSnapshot.capture(endpoint, concurrencyLimit));
            }
        });
        return new ClusterSnapshot(Map.of(), Collections.unmodifiableMap(decodes));
    }

    public static ClusterSnapshot capture(EndpointRegistry registry, FlexlbConfig config) {
        int prefillQueueCapacity = config.isBatchDispatch()
                ? config.batchDispatcher().getMaxWaitingRequestsPerPrefillWorker()
                : config.getInternalRuntime().getNonBatchWaitingRequestsPerPrefillWorker();
        Long configuredDecodeConcurrency = config.getRouter().getRoles().getDecode()
                .getAvailability().getMaxEngineRequests();
        long decodeConcurrencyLimit = configuredDecodeConcurrency == null
                ? 0L : configuredDecodeConcurrency;
        Map<String, PrefillEndpointSnapshot> prefills = new HashMap<>();
        registry.getPrefillEndpoints().forEach((key, ep) ->
                prefills.put(key, PrefillEndpointSnapshot.capture(ep, prefillQueueCapacity)));
        Map<String, DecodeEndpointSnapshot> decodes = new HashMap<>();
        registry.getDecodeEndpoints().forEach((key, ep) ->
                decodes.put(key, DecodeEndpointSnapshot.capture(ep, decodeConcurrencyLimit)));
        return new ClusterSnapshot(prefills, decodes);
    }
}
