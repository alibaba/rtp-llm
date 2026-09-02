package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.util.Logger;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Converter between gRPC protobuf messages and Java objects for engine status
 */
public class EngineStatusConverter {

    private static final AtomicBoolean KVCM_COMPATIBILITY_LAYOUT_REPORTED = new AtomicBoolean();

    /**
     * Convert WorkerStatusPB to WorkerStatusResponse
     */
    public static WorkerStatusResponse convertToWorkerStatusResponse(EngineRpcService.WorkerStatusPB workerStatusPB) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        WorkerStatusFields fields = WorkerStatusFields.decode(workerStatusPB);

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
        response.setDpRank(fields.dpRank());
        response.setBlockHashLookaheadTokens(fields.blockHashLookaheadTokens());
        response.setCacheMatchRollbackBlocks(fields.cacheMatchRollbackBlocks());
        response.setKvCacheGroupMode(fields.kvCacheGroupMode());
        response.setStatusVersion(workerStatusPB.getStatusVersion());
        response.setLatestFinishedVersion(workerStatusPB.getLatestFinishedVersion());
        response.setAlive(workerStatusPB.getAlive());
        response.setAvailableKvCacheTokens(fields.availableKvCache());
        response.setTotalKvCacheTokens(fields.totalKvCache());
        response.setMaxSeqLen(fields.maxSeqLen());
        response.setMaxBatchTokensSize(fields.maxBatchTokensSize());
        if (fields.blockSize() > 0) {
            response.setCacheStatus(CacheStatus.builder()
                    .availableKvCache(fields.availableKvCache())
                    .totalKvCache(fields.totalKvCache())
                    .blockSize(fields.blockSize())
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

    private static KvCacheGroupMode convertKvCacheGroupMode(long encodedMode) {
        if (encodedMode == EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY_VALUE) {
            return KvCacheGroupMode.FULL_ATTENTION_ONLY;
        }
        if (encodedMode == EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA_VALUE) {
            return KvCacheGroupMode.WITH_MAMBA;
        }
        return KvCacheGroupMode.UNSPECIFIED;
    }

    /**
     * Resource fields from the two WorkerStatusPB layouts supported during the
     * FlexLB/KVCM rolling upgrade.
     *
     * <p>The KVCM layout used fields 16-18 and 21-23 before the dsv4 layout
     * assigned different meanings to those numbers. Its decoded shape is
     * unambiguous when the value read as dp_rank is outside dp_size or the
     * current cache tuple violates available &lt;= total, while the shifted tuple
     * remains valid. Keep this repair at the protocol boundary so scheduling
     * and cache matching consume one coherent domain object.</p>
     */
    private record WorkerStatusFields(
            long dpRank,
            long availableKvCache,
            long totalKvCache,
            long maxSeqLen,
            long maxBatchTokensSize,
            long blockSize,
            int blockHashLookaheadTokens,
            int cacheMatchRollbackBlocks,
            KvCacheGroupMode kvCacheGroupMode) {

        private static WorkerStatusFields decode(EngineRpcService.WorkerStatusPB status) {
            if (!usesKvcmCompatibilityLayout(status)) {
                return new WorkerStatusFields(
                        status.getDpRank(), status.getAvailableKvCache(), status.getTotalKvCache(),
                        status.getMaxSeqLen(), status.getMaxBatchTokensSize(), status.getBlockSize(),
                        status.getBlockHashLookaheadTokens(), status.getCacheMatchRollbackBlocks(),
                        convertKvCacheGroupMode(status.getKvCacheGroupMode()));
            }
            if (KVCM_COMPATIBILITY_LAYOUT_REPORTED.compareAndSet(false, true)) {
                Logger.warn("Detected KVCM WorkerStatusPB compatibility layout; remapping resource fields 16-23");
            }
            return new WorkerStatusFields(
                    0L, status.getDpRank(), status.getAvailableKvCache(), 0L, 0L,
                    status.getTotalKvCache(), Math.toIntExact(status.getMaxSeqLen()),
                    Math.toIntExact(status.getBlockSize()),
                    convertKvCacheGroupMode(status.getMaxBatchTokensSize()));
        }

        private static boolean usesKvcmCompatibilityLayout(EngineRpcService.WorkerStatusPB status) {
            long possibleAvailableKv = status.getDpRank();
            long possibleTotalKv = status.getAvailableKvCache();
            long possibleBlockSize = status.getTotalKvCache();
            boolean invalidCurrentLayout = possibleAvailableKv >= status.getDpSize()
                    || status.getAvailableKvCache() > status.getTotalKvCache();
            return status.getDpSize() > 0
                    && invalidCurrentLayout
                    && possibleAvailableKv >= 0
                    && possibleAvailableKv <= possibleTotalKv
                    && possibleBlockSize > 0
                    && possibleBlockSize < possibleTotalKv
                    && status.getMaxBatchTokensSize() >= 0
                    && status.getMaxBatchTokensSize()
                            <= EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_WITH_MAMBA_VALUE
                    && status.getBlockHashLookaheadTokens() == 0
                    && status.getCacheMatchRollbackBlocks() == 0
                    && status.getKvCacheGroupMode()
                            == EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_UNSPECIFIED;
        }
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
