package org.flexlb.sync.status;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class ModelWorkerStatus {

    /**
     * Non-PD separation mode
     */
    private final Map<String/*ipPort*/, WorkerStatus> pdFusionStatusMap =
            new ConcurrentHashMap<>();

    private final Map<String/*ipPort*/, WorkerStatus> prefillStatusMap =
            new ConcurrentHashMap<>();

    private final Map<String/*ipPort*/, WorkerStatus> decodeStatusMap =
            new ConcurrentHashMap<>();

    private final Map<String/*ipPort*/, WorkerStatus> vitStatusMap =
            new ConcurrentHashMap<>();

    private final Map<String/*ipPort*/, WorkerStatus> frontendStatusMap =
            new ConcurrentHashMap<>();

    public Map<String, WorkerStatus> getPdFusionStatusMap() {
        return pdFusionStatusMap;
    }

    public Map<String, WorkerStatus> getPrefillStatusMap() {
        return prefillStatusMap;
    }

    public Map<String, WorkerStatus> getDecodeStatusMap() {
        return decodeStatusMap;
    }

    public Map<String, WorkerStatus> getVitStatusMap() {
        return vitStatusMap;
    }

    public Map<String, WorkerStatus> getFrontendStatusMap() {
        return frontendStatusMap;
    }

    public Map<String, WorkerStatus> getRoleStatusMap(RoleType roleType) {
        return switch (roleType) {
            case DECODE -> decodeStatusMap;
            case PREFILL -> prefillStatusMap;
            case PDFUSION -> pdFusionStatusMap;
            case VIT -> vitStatusMap;
            case FRONTEND -> frontendStatusMap;
            case null -> Map.of();
        };
    }

    public int getWorkerTotalCount() {
        return pdFusionStatusMap.size() + decodeStatusMap.size() + prefillStatusMap.size() + vitStatusMap.size() + frontendStatusMap.size();
    }
}
