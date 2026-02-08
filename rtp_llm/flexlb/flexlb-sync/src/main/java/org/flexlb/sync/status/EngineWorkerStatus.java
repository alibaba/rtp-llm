package org.flexlb.sync.status;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
@Data
@Component
public class EngineWorkerStatus {

    public static final Map<String/*modelName*/, ModelWorkerStatus> MODEL_ROLE_WORKER_STATUS_MAP = new ConcurrentHashMap<>();

    public final ModelMetaConfig modelMetaConfig;

    public EngineWorkerStatus(ModelMetaConfig modelMetaConfig) {
        this.modelMetaConfig = modelMetaConfig;
    }

    public Map<String/*ipPort*/, WorkerStatus> selectModelWorkerStatus(String modelName, RoleType roleType, String group) {

        ModelWorkerStatus modelWorkerStatus = MODEL_ROLE_WORKER_STATUS_MAP.get(modelName);
        if (modelWorkerStatus == null) {
            return Map.of();
        }
        Map<String/*ip:port*/, WorkerStatus> roleStatusMap = modelWorkerStatus.getRoleStatusMap(roleType);

        if (group == null) {
            return roleStatusMap;
        }

        return roleStatusMap.entrySet()
                .stream()
                .filter(entry -> {
                    WorkerStatus workerStatus = entry.getValue();
                    return workerStatus != null && workerStatus.getGroup() != null && workerStatus.getGroup().equals(group);
                })
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

}
