package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatus.EngineObservation;
import org.flexlb.dao.master.WorkerStatus.StatusObservation;
import org.flexlb.dao.master.WorkerStatus.TaskObservation;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RequestId;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Converter between gRPC protobuf messages and Java objects for engine status
 */
public class EngineStatusConverter {

    /**
     * Convert one protobuf response into an immutable observation, retaining all
     * active task phases from {@code running_task_info}.
     */
    public static StatusObservation convertToStatusObservation(WorkerStatus owner,
                                                               EngineRpcService.WorkerStatusPB workerStatusPB) {
        Map<String, TaskObservation> activeTasks = convertTasks(
                workerStatusPB.getRunningTaskInfoList());
        Map<String, TaskObservation> finishedTasks = convertTasks(
                workerStatusPB.getFinishedTaskListList());
        EngineObservation engine = new EngineObservation(
                RoleTypeProtoConverter.fromWorkerStatus(workerStatusPB),
                (long) workerStatusPB.getAvailableConcurrency(),
                workerStatusPB.getAvailableKvCache(),
                workerStatusPB.getTotalKvCache(),
                activeTasks,
                workerStatusPB.getStepLatencyMs(),
                workerStatusPB.getIterateCount(),
                workerStatusPB.getDpSize(),
                workerStatusPB.getTpSize(),
                workerStatusPB.getDpRank(),
                workerStatusPB.getBlockSize(),
                workerStatusPB.getBlockHashLookaheadTokens(),
                workerStatusPB.getCacheMatchRollbackBlocks(),
                convertKvCacheGroupMode(workerStatusPB.getKvCacheGroupMode()),
                workerStatusPB.getMaxSeqLen(),
                workerStatusPB.getMaxBatchTokensSize(),
                workerStatusPB.getRunningQueryLen(),
                workerStatusPB.getWaitingQueryLen(),
                convertStepMetrics(workerStatusPB));
        return owner.bindStatusObservation(
                engine,
                workerStatusPB.getAlive(),
                workerStatusPB.getStatusVersion(),
                workerStatusPB.getLatestFinishedVersion(),
                finishedTasks);
    }

    private static WorkerStatus.StepMetrics convertStepMetrics(EngineRpcService.WorkerStatusPB status) {
        if (!status.hasLastStepMetrics()) {
            return null;
        }
        EngineRpcService.WorkerStepMetricsPB step = status.getLastStepMetrics();
        return new WorkerStatus.StepMetrics(step.getStepId(), step.getCompletedTimeMs(),
                step.getTotalScheduledTokens(), step.getPrefillRequestCount(), step.getPrefillTokens(),
                step.getTokenBudget(), step.getBudgetFillRatio());
    }

    private static KvCacheGroupMode convertKvCacheGroupMode(
            EngineRpcService.KvCacheGroupModePB mode) {
        return switch (mode) {
            case KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY -> KvCacheGroupMode.FULL_ATTENTION_ONLY;
            case KV_CACHE_GROUP_MODE_WITH_MAMBA -> KvCacheGroupMode.WITH_MAMBA;
            default -> KvCacheGroupMode.UNSPECIFIED;
        };
    }

    /**
     * Convert CacheStatusPB to CacheStatus
     */
    public static CacheStatus convertToCacheStatus(EngineRpcService.CacheStatusPB cacheStatusPB) {
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(cacheStatusPB.getAvailableKvCache());
        cacheStatus.setTotalKvCache(cacheStatusPB.getTotalKvCache());
        cacheStatus.setBlockSize(cacheStatusPB.getBlockSize());
        cacheStatus.setVersion(cacheStatusPB.getVersion());

        Map<Long, Boolean> cacheKeysMap = cacheStatusPB.getCacheKeysMap();
        // Copy the keySet: protobuf MapField#keySet() returns a VIEW that keeps
        // the whole WorkerStatusPB message graph reachable until the next sync.
        Set<Long> cachedKeysSet = new HashSet<>(cacheKeysMap.keySet());
        cacheStatus.setCachedKeys(cachedKeysSet);
        cacheStatus.setCacheKeySize(cacheKeysMap.size());
        return cacheStatus;
    }

    /**
     * Convert protobuf task values directly into immutable observations.
     */
    private static Map<String, TaskObservation> convertTasks(List<EngineRpcService.TaskInfoPB> taskInfoPBList) {
        if (taskInfoPBList == null || taskInfoPBList.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> tasks = new HashMap<>(
                taskInfoPBList.size());

        for (EngineRpcService.TaskInfoPB task : taskInfoPBList) {
            long errorCode = task.hasErrorInfo()
                    ? task.getErrorInfo().getErrorCode() : 0L;
            String errorMessage = errorCode == 0L
                    ? null : task.getErrorInfo().getErrorMessage();
            TaskObservation observation = new TaskObservation(
                    RequestId.parse(task),
                    task.getPrefixLength(),
                    0L,
                    task.getInputLength(),
                    task.getWaitingTimeMs(),
                    task.getIterateCount(),
                    task.getEndTimeMs(),
                    task.getDpRank(),
                    errorCode,
                    errorMessage,
                    task.getBatchId(),
                    resolvePhase(task),
                    task.getExecutionTimeMs(),
                    switch (task.getPriorityPreemptionProgress()) {
                        case PRIORITY_PREEMPTION_CANCELING ->
                                PriorityPreemptionProgress.CANCELING;
                        case PRIORITY_PREEMPTION_CANCELED ->
                                PriorityPreemptionProgress.CANCELED;
                        case PRIORITY_PREEMPTION_NONE, UNRECOGNIZED ->
                                PriorityPreemptionProgress.NONE;
                    },
                    new WorkerStatus.TaskTelemetry(task.getPrefixLengthValid(),
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
                            task.getPrefillNonfinalChunkTokensMax()));
            tasks.put(observation.requestId(), observation);
        }
        return Map.copyOf(tasks);
    }
    private static TaskPhase resolvePhase(EngineRpcService.TaskInfoPB task) {
        if (task.getPhase() != EngineRpcService.TaskPhase.TASK_PHASE_PENDING) {
            // phase was added after the legacy is_waiting flag and is the
            // authoritative state whenever it carries a non-default value.
            // In particular, the immediately preceding e0 wire schema did not
            // contain field 9, so its RECEIVED/KV_ALLOCATED payloads are read
            // here with is_waiting=false. Treating that default as an explicit
            // value rejects valid status snapshots during a rolling upgrade.
            return convertPhase(task.getPhase());
        }
        // dsv4 writers only sent is_waiting. Because old proto3 writers omit
        // false on the wire, phase=0/is_waiting=false means legacy RUNNING.
        // New PENDING writers dual-write is_waiting=true.
        return task.getIsWaiting() ? TaskPhase.PENDING : TaskPhase.RUNNING;
    }

    private static TaskPhase convertPhase(EngineRpcService.TaskPhase protoPhase) {
        switch (protoPhase) {
            case TASK_PHASE_RECEIVED:
                return TaskPhase.RECEIVED;
            case TASK_PHASE_KV_ALLOCATED:
                return TaskPhase.KV_ALLOCATED;
            case TASK_PHASE_RUNNING:
                return TaskPhase.RUNNING;
            default:
                return TaskPhase.PENDING;
        }
    }
}
