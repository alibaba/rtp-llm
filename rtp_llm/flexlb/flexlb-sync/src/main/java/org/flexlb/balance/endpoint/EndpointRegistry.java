package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final EngineWorkerStatus engineWorkerStatus;

    public EndpointRegistry(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
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

    public DecodeEndpoint getOrCreateDecode(String ipPort, java.util.function.Function<String, DecodeEndpoint> factory) {
        return decodeEndpoints.computeIfAbsent(ipPort, factory);
    }

    public Map<String, PrefillEndpoint> getPrefillEndpoints(String group) {
        return getEndpointsByRole(prefillEndpoints, RoleType.PREFILL, group);
    }

    public Map<String, DecodeEndpoint> getDecodeEndpoints(String group) {
        return getEndpointsByRole(decodeEndpoints, RoleType.DECODE, group);
    }

    public void removePrefill(String ipPort) {
        prefillEndpoints.remove(ipPort);
    }

    public void removeDecode(String ipPort) {
        decodeEndpoints.remove(ipPort);
    }

    /**
     * Backward-compatible get that returns any endpoint type.
     */
    public WorkerEndpoint get(String ipPort) {
        PrefillEndpoint pep = prefillEndpoints.get(ipPort);
        if (pep != null) {
            return pep;
        }
        return decodeEndpoints.get(ipPort);
    }

    public void remove(String ipPort) {
        prefillEndpoints.remove(ipPort);
        decodeEndpoints.remove(ipPort);
    }

    private <T> Map<String, T> getEndpointsByRole(ConcurrentHashMap<String, T> endpoints,
                                                    RoleType roleType, String group) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (workerStatusMap == null || workerStatusMap.isEmpty()) {
            return Map.of();
        }
        Map<String, T> result = new LinkedHashMap<>();
        for (String ipPort : workerStatusMap.keySet()) {
            T ep = endpoints.get(ipPort);
            if (ep != null) {
                result.put(ipPort, ep);
            }
        }
        return result;
    }
}
