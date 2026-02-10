package org.flexlb.sync.status;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Data
@NoArgsConstructor
public class ModelWorkerStatus {

    /**
     * Non-PD separation mode
     */
    private Map<String/*ipPort*/, WorkerStatus> pdFusionStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> prefillStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> decodeStatusMap = new ConcurrentHashMap<>();

    private Map<String/*ipPort*/, WorkerStatus> vitStatusMap = new ConcurrentHashMap<>();

    public Map<String, WorkerStatus> getRoleStatusMap(RoleType roleType) {
        return switch (roleType) {
            case DECODE -> decodeStatusMap;
            case PREFILL -> prefillStatusMap;
            case PDFUSION -> pdFusionStatusMap;
            case VIT -> vitStatusMap;
            case null -> Map.of();
        };
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

    public int getWorkerTotalCount() {
        return pdFusionStatusMap.size() + decodeStatusMap.size() + prefillStatusMap.size() + vitStatusMap.size();
    }
}
