package org.flexlb.sync.status;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.utils.LoggingUtils;


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
    private ConcurrentHashMap<String/*ipPort*/, WorkerStatus> pdFusionStatusMap = new ConcurrentHashMap<>();

    private ConcurrentHashMap<String/*ipPort*/, WorkerStatus> prefillStatusMap = new ConcurrentHashMap<>();

    private ConcurrentHashMap<String/*ipPort*/, WorkerStatus> decodeStatusMap = new ConcurrentHashMap<>();

    private ConcurrentHashMap<String/*ipPort*/, WorkerStatus> vitStatusMap = new ConcurrentHashMap<>();

    private EngineMetric engineMetric;

    public ConcurrentHashMap<String, WorkerStatus> getRoleStatusMap(RoleType roleType) {
        if (roleType == RoleType.DECODE) {
            return decodeStatusMap;
        } else if (roleType == RoleType.PREFILL) {
            return prefillStatusMap;
        } else if (roleType == RoleType.PDFUSION) {
            return pdFusionStatusMap;
        } else if (roleType == RoleType.VIT) {
            return vitStatusMap;
        }
        return null;
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
        if(!vitStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.VIT);
        }
        return roleTypeList;
    }

    public ConcurrentHashMap<String, WorkerStatus> getRoleStatusMap(RoleType roleType, String group) {
        ConcurrentHashMap<String, WorkerStatus> roleStatusMap = getRoleStatusMap(roleType);
        if (roleStatusMap.isEmpty()) {
            LoggingUtils.warn("roleStatusMap is empty, role: {}", roleType.toString());
            return new ConcurrentHashMap<>();
        }
        ConcurrentHashMap<String, WorkerStatus> filteredMap = new ConcurrentHashMap<>();
        roleStatusMap.entrySet()
                .stream()
                .filter(entry -> entry.getValue().getGroup().equals(group))
                .forEach(entry -> filteredMap.put(entry.getKey(), entry.getValue()));
        if (filteredMap.isEmpty()) {
            LoggingUtils.warn("roleStatusMap is empty, role: {}, group:{}, ", roleType.toString(), group);
        }
        return filteredMap;
    }

}
