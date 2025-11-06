package org.flexlb.sync.status;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;
import reactor.core.Disposable;

@Slf4j
@Getter
@Setter
@Component
public class EngineWorkerStatus {
    private Map<String/*modelName*/, ModelWorkerStatus> modelRoleWorkerStatusMap = new ConcurrentHashMap<>();
    public static final Map<String/*ip+时间戳*/, Disposable> queryStatusDisposableMap = new ConcurrentHashMap<>();

    public final ModelMetaConfig modelMetaConfig;

    public EngineWorkerStatus(ModelMetaConfig modelMetaConfig) {
        this.modelMetaConfig = modelMetaConfig;
    }

    public ConcurrentHashMap<String/*ipPort*/, WorkerStatus> selectModelWorkerStatus(
        String modelName, RoleType roleType, String group) {

        ModelWorkerStatus modelWorkerStatus = modelRoleWorkerStatusMap.get(modelName);
        ConcurrentHashMap<String/*ip:port*/, WorkerStatus> roleStatusMap = modelWorkerStatus.getRoleStatusMap(roleType);

        if (group != null) {
            Map<String, WorkerStatus> filterMap = roleStatusMap.entrySet()
                .stream()
                .filter(entry -> {
                    WorkerStatus workerStatus = entry.getValue();
                    return workerStatus != null && group != null && group.equals(workerStatus.getGroup());
                })
                .collect(Collectors.toConcurrentMap(Map.Entry::getKey, Map.Entry::getValue));
            roleStatusMap.clear();
            roleStatusMap.putAll(filterMap);
        }
        return roleStatusMap;
    }

}
