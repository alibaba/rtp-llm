package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.FlexlbConfig;

import java.util.HashMap;
import java.util.Map;

/**
 * Read-only cluster view captured before building an admission plan.
 *
 * <p>Versions captured here (prefill queue versions) are validated at commit
 * time. Decode admission versions are captured later, right after the
 * scheduler's own reservation, so that only interference between plan build
 * and commit is detected.
 *
 * @param prefills prefill endpoint snapshots keyed by "ip:httpPort"
 * @param decodes  decode endpoint snapshots keyed by "ip:httpPort"
 */
public record ClusterSnapshot(
        Map<String, PrefillEndpointSnapshot> prefills,
        Map<String, DecodeEndpointSnapshot> decodes) {

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
