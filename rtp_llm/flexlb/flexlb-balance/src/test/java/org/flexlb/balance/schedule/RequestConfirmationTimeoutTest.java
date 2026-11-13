package org.flexlb.balance.schedule;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Request TTL reclaims real endpoint capacity locally, without an Engine Cancel channel.
 */
class RequestConfirmationTimeoutTest {
    private static final long REQUEST_ID = 101L;
    private static final long TIMEOUT_MS = TimeUnit.HOURS.toMillis(1L);

    private static WorkerStatus worker(RoleType role, String ip) {
        return WorkerStatus.createDiscovered(role, "g1", ip, 8080, 8081, "test");
    }

    private static WorkerStatusResponse status(RoleType role) {
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(1L);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setRunningQueryLen(0L);
        response.setFinishedTaskInfo(Map.of());
        response.setTotalKvCacheTokens(10_000L);
        response.setAvailableKvCacheTokens(10_000L);
        return response;
    }

    private static void applyStatus(WorkerEndpoint endpoint, WorkerStatusResponse response) {
        WorkerStatus worker = endpoint.getStatus();
        WorkerEndpoint.StatusChange projection;
        worker.lock.lock();
        try {
            projection =
                    endpoint.applyWorkerStatus(
                            worker,
                            worker.prepareStatus(
                                    WorkerStatusTestSupport.freezeStatusResponse(
                                            worker, response)));
        } finally {
            worker.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }

    private enum ConfirmationWait {
        MISSING_REPLY,
        UNCERTAIN_REPLY,
        AUTOMATIC_TIMER
    }
}
