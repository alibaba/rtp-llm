package org.flexlb.sync.status;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

@Slf4j
@Data
@Component
public class EngineWorkerStatus {

    public static final ModelWorkerStatus MODEL_ROLE_WORKER_STATUS = new ModelWorkerStatus();

    public final ModelMetaConfig modelMetaConfig;
    private final EndpointRegistry endpointRegistry;

    public EngineWorkerStatus(ModelMetaConfig modelMetaConfig, EndpointRegistry endpointRegistry) {
        this.modelMetaConfig = modelMetaConfig;
        this.endpointRegistry = endpointRegistry;
    }

    /**
     * Select workers for a given role and group, returning
     * {@link WorkerEndpoint} instances so callers can access both
     * engine status and endpoint-local methods (reserve / release / …).
     */
    public Map<String/*ipPort*/, WorkerEndpoint> selectModelWorkerStatus(RoleType roleType, String group) {

        Map<String/*ip:port*/, WorkerStatus> roleStatusMap = MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);

        if (roleStatusMap == null) {
            return Map.of();
        }

        Map<String, WorkerEndpoint> result = new LinkedHashMap<>();
        for (Map.Entry<String, WorkerStatus> entry : roleStatusMap.entrySet()) {
            WorkerStatus ws = entry.getValue();
            if (ws == null) {
                continue;
            }
            if (group != null && !group.equals(ws.getGroup())) {
                continue;
            }
            WorkerEndpoint ep = endpointRegistry.get(entry.getKey());
            if (ep != null) {
                result.put(entry.getKey(), ep);
            }
        }
        return result;
    }

}
