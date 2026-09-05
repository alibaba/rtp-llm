package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;

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
        EngineWorkerStatus health = new EngineWorkerStatus(registry);
        Map<String, PrefillEndpointSnapshot> prefills = new HashMap<>();
        health.selectRoutableModelWorkerStatus(RoleType.PREFILL, null).forEach((key, ep) ->
                prefills.put(key, PrefillEndpointSnapshot.capture((PrefillEndpoint) ep, prefillQueueCapacity)));
        Map<String, DecodeEndpointSnapshot> decodes = new HashMap<>();
        health.selectRoutableModelWorkerStatus(RoleType.DECODE, null).forEach((key, ep) ->
                decodes.put(key, DecodeEndpointSnapshot.capture((DecodeEndpoint) ep, decodeConcurrencyLimit)));
        return new ClusterSnapshot(prefills, decodes);
    }
}
