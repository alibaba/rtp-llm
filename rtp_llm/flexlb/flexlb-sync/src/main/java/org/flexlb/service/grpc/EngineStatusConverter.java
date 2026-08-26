package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.TaskPhase;
import org.flexlb.enums.PriorityPreemptionProgress;

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
     * Convert WorkerStatusPB to WorkerStatusResponse
     */
    public static WorkerStatusResponse convertToWorkerStatusResponse(EngineRpcService.WorkerStatusPB workerStatusPB) {
        WorkerStatusResponse response = new WorkerStatusResponse();

        response.setRole(RoleTypeProtoConverter.fromWorkerStatus(workerStatusPB));
        // Compatibility only: LocalRpcServer::GetWorkerStatus does not currently
        // populate this field. Preserve it for protocol compatibility/telemetry,
        // but do not use it as a scheduling or batching limit.
        response.setAvailableConcurrency(workerStatusPB.getAvailableConcurrency());
        response.setRunningQueryLen(workerStatusPB.getRunningQueryLen());
        response.setWaitingQueryLen(workerStatusPB.getWaitingQueryLen());
        response.setStepLatencyMs(workerStatusPB.getStepLatencyMs());
        response.setIterateCount(workerStatusPB.getIterateCount());
        response.setDpSize(workerStatusPB.getDpSize());
        response.setTpSize(workerStatusPB.getTpSize());
        response.setDpRank(workerStatusPB.getDpRank());
        response.setBlockHashLookaheadTokens(workerStatusPB.getBlockHashLookaheadTokens());
        response.setCacheMatchRollbackBlocks(workerStatusPB.getCacheMatchRollbackBlocks());
        response.setKvCacheGroupMode(convertKvCacheGroupMode(workerStatusPB.getKvCacheGroupMode()));
        response.setStatusVersion(workerStatusPB.getStatusVersion());
        response.setLatestFinishedVersion(workerStatusPB.getLatestFinishedVersion());
        response.setAlive(workerStatusPB.getAlive());
        response.setAvailableKvCacheTokens(workerStatusPB.getAvailableKvCache());
        response.setTotalKvCacheTokens(workerStatusPB.getTotalKvCache());
        response.setMaxSeqLen(workerStatusPB.getMaxSeqLen());
        response.setMaxBatchTokensSize(workerStatusPB.getMaxBatchTokensSize());
        if (workerStatusPB.getBlockSize() > 0) {
            response.setCacheStatus(CacheStatus.builder()
                    .availableKvCache(workerStatusPB.getAvailableKvCache())
                    .totalKvCache(workerStatusPB.getTotalKvCache())
                    .blockSize(workerStatusPB.getBlockSize())
                    .version(workerStatusPB.getStatusVersion())
                    .build());
        }

        List<EngineRpcService.TaskInfoPB> srcRunningTaskInfoList = workerStatusPB.getRunningTaskInfoList();
        List<EngineRpcService.TaskInfoPB> waitingTaskInfoList = srcRunningTaskInfoList.stream()
                .filter(taskInfoPB -> resolvePhase(taskInfoPB) != TaskPhase.RUNNING)
                .toList();
        // Convert waiting task info
        response.setWaitingTaskInfo(convertToTaskInfoList(waitingTaskInfoList));

        // Keep the dsv4 running_task_info contract unified. Endpoint calibration
        // and priority scheduling need every active phase; lifecycle consumers
        // use waitingTaskInfo plus a phase-filtered view of this map.
        response.setRunningTaskInfo(convertToTaskInfoList(srcRunningTaskInfoList));

        // Convert finished task list
        response.setFinishedTaskInfo(convertToTaskInfoList(workerStatusPB.getFinishedTaskListList()));

        return response;
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
     * Convert list of TaskInfoPB to list of TaskInfo
     */
    private static Map<String, TaskInfo> convertToTaskInfoList(
            List<EngineRpcService.TaskInfoPB> taskInfoPBList) {
        if (taskInfoPBList == null) {
            return null;
        }
        Map<String, TaskInfo> taskInfoMap = new HashMap<>(taskInfoPBList.size());

        for (EngineRpcService.TaskInfoPB taskInfoPB : taskInfoPBList) {
            TaskInfo taskInfo = new TaskInfo();
            long requestId = taskInfoPB.getRequestId();
            taskInfo.setRequestId(requestId);
            taskInfo.setPrefixLength(taskInfoPB.getPrefixLength());
            taskInfo.setPrefixLengthValid(taskInfoPB.getPrefixLengthValid());
            taskInfo.setInputLength(taskInfoPB.getInputLength());
            taskInfo.setWaitingTime(taskInfoPB.getWaitingTimeMs());
            taskInfo.setIterateCount(taskInfoPB.getIterateCount());
            taskInfo.setEndTimeMs(taskInfoPB.getEndTimeMs());
            taskInfo.setDpRank(taskInfoPB.getDpRank());
            taskInfo.setBatchId(taskInfoPB.getBatchId());
            taskInfo.setExecutionTimeMs(taskInfoPB.getExecutionTimeMs());
            taskInfo.setPhase(resolvePhase(taskInfoPB));
            taskInfo.setPriorityPreemptionProgress(switch (
                    taskInfoPB.getPriorityPreemptionProgress()) {
                case PRIORITY_PREEMPTION_CANCELING -> PriorityPreemptionProgress.CANCELING;
                case PRIORITY_PREEMPTION_CANCELED -> PriorityPreemptionProgress.CANCELED;
                case PRIORITY_PREEMPTION_NONE, UNRECOGNIZED ->
                        PriorityPreemptionProgress.NONE;
            });
            if (taskInfoPB.hasErrorInfo() && taskInfoPB.getErrorInfo().getErrorCode() != 0L) {
                taskInfo.setErrorCode(taskInfoPB.getErrorInfo().getErrorCode());
                taskInfo.setErrorMessage(taskInfoPB.getErrorInfo().getErrorMessage());
            }
            taskInfo.setWaitingEnteredTimeMs(taskInfoPB.getWaitingEnteredTimeMs());
            taskInfo.setRunningEnteredTimeMs(taskInfoPB.getRunningEnteredTimeMs());
            taskInfo.setRequestReceivedTimeMs(taskInfoPB.getRequestReceivedTimeMs());
            taskInfo.setInputQueueEnqueueTimeMs(taskInfoPB.getInputQueueEnqueueTimeMs());
            taskInfo.setInputQueueDrainTimeMs(taskInfoPB.getInputQueueDrainTimeMs());
            taskInfo.setRemoteKvWaitMs(taskInfoPB.getRemoteKvWaitMs());
            taskInfo.setFirstTokenTimeMs(taskInfoPB.getFirstTokenTimeMs());
            taskInfo.setHbmLocalMatchTokens(taskInfoPB.getHbmLocalMatchTokens());
            taskInfo.setRemoteKvAddedMatchTokens(taskInfoPB.getRemoteKvAddedMatchTokens());
            taskInfo.setFirstPrefillStepId(taskInfoPB.getFirstPrefillStepId());
            taskInfo.setLastPrefillStepId(taskInfoPB.getLastPrefillStepId());
            taskInfo.setPrefillStepCount(taskInfoPB.getPrefillStepCount());
            taskInfo.setPrefillNonfinalChunkTokensMin(taskInfoPB.getPrefillNonfinalChunkTokensMin());
            taskInfo.setPrefillNonfinalChunkTokensMax(taskInfoPB.getPrefillNonfinalChunkTokensMax());
            if (taskInfoPB.hasCompletedPrefillTokens()) {
                taskInfo.setCompletedPrefillTokens(taskInfoPB.getCompletedPrefillTokens());
            }
            if (taskInfoPB.hasRemainingPrefillTokens()) {
                taskInfo.setRemainingPrefillTokens(taskInfoPB.getRemainingPrefillTokens());
            }
            if (taskInfoPB.hasLastCompletedPrefillStepId()) {
                taskInfo.setLastCompletedPrefillStepId(taskInfoPB.getLastCompletedPrefillStepId());
            }

            taskInfoMap.put(String.valueOf(requestId), taskInfo);
        }

        return taskInfoMap;
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
