package org.flexlb.sync.status;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
        Map<String, List<WorkerEndpoint>> groups = new HashMap<>();
        for (WorkerEndpoint endpoint : candidates.values()) {
            groups.computeIfAbsent(endpoint.getStatus().getPhysicalGroupKey(), key -> new ArrayList<>())
                    .add(endpoint);
        }
        Set<WorkerEndpoint> healthy = new HashSet<>();
        for (List<WorkerEndpoint> siblings : groups.values()) {
            if (isHealthySiblingGroup(siblings)) {
                healthy.addAll(siblings);
            }
        }
        candidates.entrySet().removeIf(entry -> !healthy.contains(entry.getValue()));
        return candidates;
    }

    /** Recheck the current sibling endpoints before committing a routing choice. */
    public boolean isPhysicalGroupHealthy(WorkerEndpoint endpoint) {
        WorkerStatus worker = endpoint == null ? null : endpoint.getStatus();
        if (worker == null || !worker.isAlive() || worker.getMultiEngineNum() < 1) {
            return false;
        }
        String groupKey = worker.getPhysicalGroupKey();
        List<WorkerEndpoint> siblings = new ArrayList<>();
        for (WorkerEndpoint sibling : endpointRegistry.getEndpoints(worker.getRole()).values()) {
            if (sibling.getStatus() != null
                    && groupKey.equals(sibling.getStatus().getPhysicalGroupKey())) {
                siblings.add(sibling);
            }
        }
        return siblings.contains(endpoint) && isHealthySiblingGroup(siblings);
    }

    private boolean isHealthySiblingGroup(List<WorkerEndpoint> siblings) {
        if (siblings.isEmpty()) {
            return false;
        }
        int expected = siblings.getFirst().getStatus().getMultiEngineNum();
        if (expected < 1 || siblings.size() != expected) {
            return false;
        }
        boolean[] indexes = new boolean[expected];
        for (WorkerEndpoint endpoint : siblings) {
            WorkerStatus sibling = endpoint.getStatus();
            int index = sibling.getEngineIndex();
            if (sibling.getMultiEngineNum() != expected || index < 0 || index >= expected
                    || indexes[index] || !sibling.isAlive()) {
                return false;
            }
            indexes[index] = true;
        }
        return true;
    }

}
