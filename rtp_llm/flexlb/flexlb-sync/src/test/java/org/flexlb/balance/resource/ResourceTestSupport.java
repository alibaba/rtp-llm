package org.flexlb.balance.resource;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;

import java.util.Map;

/** Package-local frozen-status fixtures for resource decisions. */
final class ResourceTestSupport {

    private ResourceTestSupport() {
    }

    static WorkerStatus worker(
            RoleType role,
            long totalKv,
            long availableKv,
            Map<String, TaskInfo> runningTasks) {
        WorkerStatus worker = WorkerStatus.createDiscovered(
                role, null, "127.0.0.1", 8080, 9090, "test-site");
        publish(worker, true, totalKv, availableKv, runningTasks);
        return worker;
    }

    static void publish(
            WorkerStatus worker,
            boolean alive,
            long totalKv,
            long availableKv,
            Map<String, TaskInfo> runningTasks) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(worker.getRole());
        response.setAlive(alive);
        response.setTotalKvCacheTokens(totalKv);
        response.setAvailableKvCacheTokens(availableKv);
        response.setRunningTaskInfo(runningTasks);
        response.setStatusVersion(
                Math.max(1L, worker.appliedStatusCursor().statusVersion() + 1L));
        response.setLatestFinishedVersion(
                worker.appliedStatusCursor().latestFinishedTaskVersion());
        worker.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = worker.prepareNewStatus(
                    worker.freezeStatusResponse(response));
            worker.publishPreparedStatus(prepared);
        } finally {
            worker.lock.unlock();
        }
    }
}
