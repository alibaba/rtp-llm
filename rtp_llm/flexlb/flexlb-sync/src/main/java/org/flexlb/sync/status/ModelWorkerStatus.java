package org.flexlb.sync.status;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.LoggingUtils;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * @author zjw
 * description:
 * date: 2025/4/24
 */
@Data
@NoArgsConstructor
public class ModelWorkerStatus {

    /**
     * 非PD分离模式
     */
    private Map<String/*ipPort*/, WorkerStatus> pdFusionStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> prefillStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> decodeStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> vitStatusMap = new ConcurrentHashMap<>();

    public Map<String, WorkerStatus> getRoleStatusMap(RoleType roleType) {

        if (roleType == RoleType.DECODE) {
            return decodeStatusMap;
        } else if (roleType == RoleType.PREFILL) {
            return prefillStatusMap;
        } else if (roleType == RoleType.PDFUSION) {
            return pdFusionStatusMap;
        } else if (roleType == RoleType.VIT) {
            return vitStatusMap;
        }

        return Map.of();
    }

    public List<RoleType> getRoleTypeList() {
        List<RoleType> roleTypeList = new ArrayList<>();
        if (!pdFusionStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.PDFUSION);
        }
        if (!decodeStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.DECODE);
        }
        if (!prefillStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.PREFILL);
        }
        if (!vitStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.VIT);
        }
        return roleTypeList;
    }

    public Map<String, WorkerStatus> getRoleStatusMap(RoleType roleType, String group) {
        Map<String, WorkerStatus> roleStatusMap = getRoleStatusMap(roleType);
        if (CollectionUtils.isEmpty(roleStatusMap)) {
            LoggingUtils.warn("roleStatusMap is empty, role: {}", roleType.toString());
            return Map.of();
        }

        Map<String, WorkerStatus> filteredMap = roleStatusMap.entrySet()
                .stream()
                .filter(entry -> entry.getValue().getGroup().equals(group))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        if (CollectionUtils.isEmpty(filteredMap)) {
            LoggingUtils.warn("roleStatusMap is empty, role: {}, group:{}, ", roleType.toString(), group);
        }

        return filteredMap;
    }

    public int getWorkerTotalCount() {
        return pdFusionStatusMap.size() + decodeStatusMap.size() + prefillStatusMap.size() + vitStatusMap.size();
    }
}
