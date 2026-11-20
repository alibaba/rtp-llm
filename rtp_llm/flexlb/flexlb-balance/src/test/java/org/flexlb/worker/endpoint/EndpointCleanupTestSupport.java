package org.flexlb.worker.endpoint;

import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatusResponse;

import java.util.Map;

/**
 * Real endpoint ledgers for scheduler-directory cleanup tests across package boundaries.
 */
public final class EndpointCleanupTestSupport {
    private EndpointCleanupTestSupport() {}

    public static void finishDecode(DecodeEndpoint endpoint, long requestId) {
        var finished = new TaskInfo();
        finished.setRequestId(String.valueOf(requestId));
        var response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of(String.valueOf(requestId), finished));
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    public static void confirmDecode(DecodeEndpoint endpoint, long requestId) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(String.valueOf(requestId));
        running.setPhase(TaskPhase.KV_ALLOCATED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of(Long.toString(requestId), running));
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }
}
