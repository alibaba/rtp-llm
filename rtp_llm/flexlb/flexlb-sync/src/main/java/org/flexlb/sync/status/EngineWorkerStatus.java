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

    /** Select logical endpoints only when all engines of their physical frontend are healthy. */
    public Map<String, WorkerEndpoint> selectRoutableModelWorkerStatus(RoleType roleType, String group) {
        Map<String, WorkerEndpoint> candidates = selectModelWorkerStatus(roleType, group);
        candidates.entrySet().removeIf(entry -> !isPhysicalGroupHealthy(entry.getValue()));
        return candidates;
    }

    /** Recheck the current sibling endpoints before committing a routing choice. */
    public boolean isPhysicalGroupHealthy(WorkerEndpoint endpoint) {
        WorkerStatus worker = endpoint == null ? null : endpoint.getStatus();
        if (worker == null || !worker.isAlive() || worker.getMultiEngineNum() < 1) {
            return false;
        }
        int expected = worker.getMultiEngineNum();
        boolean[] indexes = new boolean[expected];
        int count = 0;
        boolean currentEndpoint = false;
        for (WorkerEndpoint siblingEndpoint : endpointRegistry.getEndpoints(worker.getRole()).values()) {
            WorkerStatus sibling = siblingEndpoint.getStatus();
            if (sibling == null || !worker.getPhysicalGroupKey().equals(sibling.getPhysicalGroupKey())) {
                continue;
            }
            int index = sibling.getEngineIndex();
            if (sibling.getMultiEngineNum() != expected || index < 0 || index >= expected
                    || indexes[index] || !sibling.isAlive()) {
                return false;
            }
            indexes[index] = true;
            count++;
            currentEndpoint |= siblingEndpoint == endpoint;
        }
        return currentEndpoint && count == expected;
    }

}
