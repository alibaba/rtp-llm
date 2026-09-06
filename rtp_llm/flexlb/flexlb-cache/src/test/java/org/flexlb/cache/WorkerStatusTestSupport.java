package org.flexlb.cache;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.KvCacheGroupMode;

import java.util.Map;

public final class WorkerStatusTestSupport {

    private WorkerStatusTestSupport() {
    }

    public static WorkerStatus workerStatus(String ip, int port, RoleType role) {
        return workerStatus(ip, port, role, false, null, 0);
    }

    public static WorkerStatus workerStatus(
            String ip, int port, RoleType role, boolean alive, CacheStatus cacheStatus) {
        return workerStatus(ip, port, role, alive, cacheStatus, 0);
    }

    public static WorkerStatus workerStatus(
            String ip,
            int port,
            RoleType role,
            boolean alive,
            CacheStatus cacheStatus,
            int cacheMatchRollbackBlocks) {
        WorkerStatus workerStatus = WorkerStatus.createDiscovered(
                role, "default", ip, port, port + 1, "");
        publishObservation(workerStatus, role, alive, cacheMatchRollbackBlocks);
        if (cacheStatus != null) {
            workerStatus.publishCacheStatus(cacheStatus);
        }
        return workerStatus;
    }

    private static void publishObservation(
            WorkerStatus workerStatus,
            RoleType role,
            boolean alive,
            int cacheMatchRollbackBlocks) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(alive);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setKvCacheGroupMode(KvCacheGroupMode.UNSPECIFIED);
        response.setCacheMatchRollbackBlocks(cacheMatchRollbackBlocks);

        workerStatus.lock.lock();
        try {
            WorkerStatus.StatusObservation observation =
                    workerStatus.freezeStatusResponse(response);
            WorkerStatus.PreparedStatus prepared =
                    workerStatus.prepareNewStatus(observation);
            workerStatus.publishPreparedStatus(prepared);
            workerStatus.recordSuccessfulPoll(alive);
        } finally {
            workerStatus.lock.unlock();
        }
    }
}
