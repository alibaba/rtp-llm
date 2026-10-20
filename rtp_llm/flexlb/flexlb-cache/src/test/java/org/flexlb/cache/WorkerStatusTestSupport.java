package org.flexlb.cache;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.testfixture.WorkerStatusResponse;
import org.flexlb.testfixture.WorkerStatusTestFixtures;

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
            workerStatus.cacheMutationLock.lock();
            try {
                workerStatus.publishCacheStatus(cacheStatus);
            } finally {
                workerStatus.cacheMutationLock.unlock();
            }
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
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setKvCacheGroupMode(KvCacheGroupMode.UNSPECIFIED);
        response.setCacheMatchRollbackBlocks(cacheMatchRollbackBlocks);

        workerStatus.lock.lock();
        try {
            WorkerStatus.StatusObservation observation =
                    WorkerStatusTestFixtures.observation(workerStatus, response);
            WorkerStatus.PreparedStatus prepared =
                    WorkerStatusTestFixtures.prepare(workerStatus, observation);
            workerStatus.publishPreparedStatus(prepared);
            WorkerStatusTestFixtures.receive(workerStatus, alive, null);
        } finally {
            workerStatus.lock.unlock();
        }
    }
}
