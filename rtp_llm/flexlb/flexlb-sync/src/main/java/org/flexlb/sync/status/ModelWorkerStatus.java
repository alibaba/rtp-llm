package org.flexlb.sync.status;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

@Data
@NoArgsConstructor
public class ModelWorkerStatus {

    /**
     * Non-PD separation mode
     */
    private ConcurrentMap<String/*ipPort*/, WorkerStatus> pdFusionStatusMap = new ConcurrentHashMap<>();

    private ConcurrentMap<String/*ipPort*/, WorkerStatus> prefillStatusMap = new ConcurrentHashMap<>();

    private ConcurrentMap<String/*ipPort*/, WorkerStatus> decodeStatusMap = new ConcurrentHashMap<>();

    private ConcurrentMap<String/*ipPort*/, WorkerStatus> vitStatusMap = new ConcurrentHashMap<>();

    private ConcurrentMap<String/*ipPort*/, WorkerStatus> frontendStatusMap = new ConcurrentHashMap<>();

    public ConcurrentMap<String, WorkerStatus> getRoleStatusMap(RoleType roleType) {
        if (roleType == null) {
            throw new IllegalArgumentException("roleType must not be null");
        }
        return switch (roleType) {
            case DECODE -> decodeStatusMap;
            case PREFILL -> prefillStatusMap;
            case PDFUSION -> pdFusionStatusMap;
            case VIT -> vitStatusMap;
            case FRONTEND -> frontendStatusMap;
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
        if (!frontendStatusMap.isEmpty()) {
            roleTypeList.add(RoleType.FRONTEND);
        }
        return roleTypeList;
    }

    public int getWorkerTotalCount() {
        return pdFusionStatusMap.size() + decodeStatusMap.size() + prefillStatusMap.size() + vitStatusMap.size() + frontendStatusMap.size();
    }
}
