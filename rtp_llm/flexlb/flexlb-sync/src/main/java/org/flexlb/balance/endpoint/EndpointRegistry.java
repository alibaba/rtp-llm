package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final FlexlbBatchScheduler batchScheduler;

    public EndpointRegistry(ConfigService configService,
                            @Lazy FlexlbBatchScheduler batchScheduler) {
        this.configService = configService;
        this.batchScheduler = batchScheduler;
    }

    public WorkerEndpoint get(String ipPort) {
        WorkerEndpoint ep = prefillEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        return decodeEndpoints.get(ipPort);
    }

    public PrefillEndpoint getPrefill(String ipPort) {
        return prefillEndpoints.get(ipPort);
    }

    public DecodeEndpoint getDecode(String ipPort) {
        return decodeEndpoints.get(ipPort);
    }

    public PrefillEndpoint getOrCreatePrefill(String ipPort, java.util.function.Function<String, PrefillEndpoint> factory) {
        return prefillEndpoints.computeIfAbsent(ipPort, factory);
    }

    public PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status) {
        return prefillEndpoints.computeIfAbsent(ipPort,
                k -> new PrefillEndpoint(status, configService.loadBalanceConfig(), batchScheduler));
    }

    public DecodeEndpoint ensureDecodeEndpoint(String ipPort, WorkerStatus status) {
        return decodeEndpoints.computeIfAbsent(ipPort,
                k -> new DecodeEndpoint(status));
    }

    public Map<String, PrefillEndpoint> getPrefillEndpoints(String group) {
        return getEndpointsByGroup(prefillEndpoints, group);
    }

    public Map<String, DecodeEndpoint> getDecodeEndpoints(String group) {
        return getEndpointsByGroup(decodeEndpoints, group);
    }

    public void removePrefill(String ipPort) {
        prefillEndpoints.remove(ipPort);
    }

    public void removeDecode(String ipPort) {
        decodeEndpoints.remove(ipPort);
    }

    public void close() {
        prefillEndpoints.values().forEach(WorkerEndpoint::close);
        decodeEndpoints.values().forEach(WorkerEndpoint::close);
    }

    private <T extends WorkerEndpoint> Map<String, T> getEndpointsByGroup(ConcurrentHashMap<String, T> endpoints, String group) {
        Map<String, T> result = new LinkedHashMap<>();
        for (Map.Entry<String, T> entry : endpoints.entrySet()) {
            T ep = entry.getValue();
            if (group == null || group.equals(ep.getStatus().getGroup())) {
                result.put(entry.getKey(), ep);
            }
        }
        return result;
    }
}
