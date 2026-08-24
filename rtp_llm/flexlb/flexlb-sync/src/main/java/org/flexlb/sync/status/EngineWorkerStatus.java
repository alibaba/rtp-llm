package org.flexlb.sync.status;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
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

    /**
     * Returns workers keyed by logical identity in {@code ip:port@engineIndex} format.
     * The index identifies one independently routable engine behind the physical frontend.
     */
    public Map<String, WorkerStatus> selectModelWorkerStatus(RoleType roleType, String group) {

        Map<String, WorkerStatus> roleStatusMap = MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);

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

    /**
     * Returns logical workers whose complete physical sibling group is healthy.
     * Resource and cache state are deliberately not aggregated here.
     * Map keys use {@code ip:port@engineIndex}; the index identifies one independently
     * routable engine behind the physical frontend.
     */
    public Map<String, WorkerStatus> selectRoutableModelWorkerStatus(RoleType roleType, String group) {
        Map<String, WorkerStatus> candidates = selectModelWorkerStatus(roleType, group);
        if (candidates.isEmpty()) {
            return Map.of();
        }

        Map<String, Map<String, WorkerStatus>> groups = candidates.values().stream()
                .filter(Objects::nonNull)
                .collect(Collectors.groupingBy(
                        WorkerStatus::getPhysicalGroupKey,
                        Collectors.toMap(WorkerStatus::getLogicalIpPort, Function.identity(), (left, right) -> left)));
        Set<String> healthyGroups = groups.entrySet().stream()
                .filter(entry -> isHealthyPhysicalGroup(entry.getValue()))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());

        return candidates.entrySet().stream()
                .filter(entry -> entry.getValue() != null)
                .filter(entry -> healthyGroups.contains(entry.getValue().getPhysicalGroupKey()))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    private boolean isHealthyPhysicalGroup(Map<String, WorkerStatus> siblings) {
        if (siblings.isEmpty()) {
            return false;
        }
        WorkerStatus first = siblings.values().iterator().next();
        int expected = first.getMultiEngineNum();
        if (expected < 1 || siblings.size() != expected) {
            return false;
        }
        boolean[] indexes = new boolean[expected];
        for (WorkerStatus sibling : siblings.values()) {
            int engineIndex = sibling.getEngineIndex();
            if (sibling.getMultiEngineNum() != expected
                    || engineIndex < 0
                    || engineIndex >= expected
                    || indexes[engineIndex]
                    || !sibling.isAlive()) {
                return false;
            }
            indexes[engineIndex] = true;
        }
        return true;
    }

}
