package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiConsumer;

@Component
public class EngineWorkerStatus {

    public static final ModelWorkerStatus MODEL_ROLE_WORKER_STATUS = new ModelWorkerStatus();

    private final EndpointRegistry endpointRegistry;

    public EngineWorkerStatus(EndpointRegistry endpointRegistry) {
        this.endpointRegistry = endpointRegistry;
    }

    /**
     * Select workers for a given role and group, returning
     * {@link WorkerEndpoint} instances so callers can access both
     * engine status and endpoint-local methods (reserve / release / …).
     */
    public Map<String/*ipPort*/, WorkerEndpoint> selectModelWorkerStatus(RoleType roleType, String group) {

        Map<String, WorkerEndpoint> result = new LinkedHashMap<>();
        forEachModelWorkerEndpoint(roleType, group, result::put);
        return result;
    }

    /**
     * Visit registered endpoints without materializing a temporary map.
     *
     * @return number of endpoints passed to {@code action}
     */
    public int forEachModelWorkerEndpoint(RoleType roleType, String group,
                                          BiConsumer<String, WorkerEndpoint> action) {
        int visited = 0;
        for (Map.Entry<String, ? extends WorkerEndpoint> entry
                : endpointRegistry.getEndpoints(roleType).entrySet()) {
            WorkerEndpoint endpoint = entry.getValue();
            WorkerStatus ws = endpoint.getStatus();
            if (ws == null) {
                continue;
            }
            if (group != null && !group.equals(ws.getGroup())) {
                continue;
            }
            action.accept(entry.getKey(), endpoint);
            visited++;
        }
        return visited;
    }

    public int getModelWorkerCapacity(RoleType roleType) {
        Map<String, WorkerStatus> roleStatusMap = MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);
        int statusCount = roleStatusMap == null ? 0 : roleStatusMap.size();
        return Math.max(statusCount, endpointRegistry.getEndpointCount(roleType));
    }

}
