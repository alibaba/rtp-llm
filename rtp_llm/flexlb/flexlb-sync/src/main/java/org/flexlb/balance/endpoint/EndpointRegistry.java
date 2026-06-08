package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, WorkerEndpoint> endpoints = new ConcurrentHashMap<>();
    private final EngineWorkerStatus engineWorkerStatus;

    public EndpointRegistry(EngineWorkerStatus engineWorkerStatus) {
        this.engineWorkerStatus = engineWorkerStatus;
    }

    public WorkerEndpoint get(String ipPort) {
        return endpoints.get(ipPort);
    }

    public WorkerEndpoint getOrCreate(String ipPort, Function<String, WorkerEndpoint> factory) {
        return endpoints.computeIfAbsent(ipPort, factory);
    }

    public Map<String, WorkerEndpoint> getEndpoints(RoleType roleType, String group) {
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (workerStatusMap == null || workerStatusMap.isEmpty()) {
            return Map.of();
        }
        Map<String, WorkerEndpoint> result = new LinkedHashMap<>();
        for (String ipPort : workerStatusMap.keySet()) {
            WorkerEndpoint ep = endpoints.get(ipPort);
            if (ep != null) {
                result.put(ipPort, ep);
            }
        }
        return result;
    }

    public void remove(String ipPort) {
        endpoints.remove(ipPort);
    }
}
