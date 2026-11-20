package org.flexlb.worker.status;

import org.flexlb.config.service.RoleType;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Converts mutable test observations into the production immutable status protocol.
 */
public final class WorkerStatusTestSupport {
    private WorkerStatusTestSupport() {}

    public static WorkerStatus createDiscovered(
            RoleType role, String group, String ip, int port, int grpcPort, String site) {
        return createDiscovered(role, group, ip, port, grpcPort, site, null);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName) {
        return createDiscovered(role, group, ip, port, grpcPort, site, deploymentName, 0, 1);
    }

    public static WorkerStatus createDiscovered(
            RoleType role,
            String group,
            String ip,
            int port,
            int grpcPort,
            String site,
            String deploymentName,
            int engineIndex,
            int multiEngineNum) {
        Objects.requireNonNull(ip, "ip");
        if (port <= 0 || grpcPort <= 0) {
            throw new IllegalArgumentException("worker ports must be positive");
        }
        if (engineIndex < 0 || multiEngineNum <= 0 || engineIndex >= multiEngineNum) {
            throw new IllegalArgumentException("invalid logical worker index");
        }
        return new WorkerStatus(
                new WorkerStatus.TopologySnapshot(
                        group,
                        ip,
                        port,
                        grpcPort,
                        site,
                        deploymentName,
                        engineIndex,
                        multiEngineNum),
                role);
    }

    public static boolean isEngineConfirmed(DecodeTaskPhase phase) {
        return phase == DecodeTaskPhase.ACCEPTED_NOT_RUNNING || phase == DecodeTaskPhase.RUNNING;
    }

    public static long hardKvAvailable(
            org.flexlb.worker.endpoint.DecodeResources.CapacityUsage usage) {
        return Math.max(0L, usage.availableKvTokens() - usage.hardReservedKvTokens());
    }

    public static WorkerStatus.PollHealth recordSuccessfulPoll(WorkerStatus status, boolean alive) {
        try (var lease = Objects.requireNonNull(status.tryBeginStatusPoll(0L), "status poll slot")) {
            status.recordStatusRpcResult(lease, true, alive);
        }
        return status.pollHealth();
    }

    public static ServerStatus serverStatus(
            org.flexlb.config.service.RoleType role,
            String requestId,
            WorkerStatus.TopologySnapshot topology,
            WorkerStatus.EngineObservation observation) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(role);
        result.setRequestId(requestId);
        result.setGroup(topology.group());
        result.setServerIp(topology.ip());
        result.setHttpPort(topology.port());
        result.setGrpcPort(topology.grpcPort());
        result.setSelectedEngineIndex(topology.engineIndex(), topology.multiEngineNum());
        result.setDpRank(observation.dpRank());
        return result;
    }

    public static WorkerStatus.StatusObservation freezeStatusResponse(
            WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, WorkerStatus.TaskObservation> runningTasks =
                freezeTaskMap(response.getRunningTaskInfo());
        Map<String, WorkerStatus.TaskObservation> finishedTasks =
                freezeTaskMap(response.getFinishedTaskInfo());
        WorkerStatus.EngineObservation engine =
                new WorkerStatus.EngineObservation(
                        response.getRole(),
                        response.getAvailableConcurrency(),
                        response.getAvailableKvCacheTokens(),
                        response.getTotalKvCacheTokens(),
                        runningTasks,
                        response.getStepLatencyMs(),
                        response.getIterateCount(),
                        response.getDpSize(),
                        response.getTpSize(),
                        response.getDpRank(),
                        response.getBlockSize(),
                        response.getBlockHashLookaheadTokens(),
                        response.getCacheMatchRollbackBlocks(),
                        response.getKvCacheGroupMode(),
                        response.getMaxSeqLen(),
                        response.getMaxBatchTokensSize(),
                        response.getRunningQueryLen(),
                        response.getWaitingQueryLen(),
                        response.getLastStepMetrics());
        return status.bindStatusObservation(
                engine,
                response.isAlive(),
                response.getStatusVersion(),
                response.getLatestFinishedVersion(),
                finishedTasks);
    }

    private static Map<String, WorkerStatus.TaskObservation> freezeTaskMap(
            Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, WorkerStatus.TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), copyOf(entry.getValue()));
            }
        }
        return Map.copyOf(frozen);
    }

    private static WorkerStatus.TaskObservation copyOf(TaskInfo task) {
        return new WorkerStatus.TaskObservation(
                task.getRequestId(),
                task.getPrefixLength(),
                task.getPrefillTime(),
                task.getInputLength(),
                task.getWaitingTime(),
                task.getIterateCount(),
                task.getEndTimeMs(),
                task.getDpRank(),
                task.getErrorCode(),
                task.getErrorMessage(),
                task.getPhase(),
                task.getExecutionTimeMs(),
                task.getPriorityPreemptionProgress(),
                new WorkerStatus.TaskTelemetry(
                        task.isPrefixLengthValid(),
                        task.getRequestReceivedTimeMs(),
                        task.getInputQueueEnqueueTimeMs(),
                        task.getInputQueueDrainTimeMs(),
                        task.getWaitingEnteredTimeMs(),
                        task.getRunningEnteredTimeMs(),
                        task.getRemoteKvWaitMs(),
                        task.getFirstTokenTimeMs(),
                        task.getHbmLocalMatchTokens(),
                        task.getRemoteKvAddedMatchTokens(),
                        task.getFirstPrefillStepId(),
                        task.getLastPrefillStepId(),
                        task.getPrefillStepCount(),
                        task.getPrefillNonfinalChunkTokensMin(),
                        task.getPrefillNonfinalChunkTokensMax(),
                        task.getCompletedPrefillTokens(),
                        task.getRemainingPrefillTokens() < 0
                                ? null
                                : task.getRemainingPrefillTokens(),
                        task.getLastCompletedPrefillStepId()));
    }
}
