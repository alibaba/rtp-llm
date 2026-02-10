package org.flexlb.sync.status;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Data
@Component
public class EngineWorkerStatus {

    public static final ModelWorkerStatus MODEL_ROLE_WORKER_STATUS = new ModelWorkerStatus();

    public final ModelMetaConfig modelMetaConfig;

    public EngineWorkerStatus(ModelMetaConfig modelMetaConfig) {
        this.modelMetaConfig = modelMetaConfig;
    }

    public Map<String/*ipPort*/, WorkerStatus> selectModelWorkerStatus(RoleType roleType, String group) {

        Map<String/*ip:port*/, WorkerStatus> roleStatusMap = MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);

        if (roleStatusMap == null) {
            return Map.of();
        }

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
