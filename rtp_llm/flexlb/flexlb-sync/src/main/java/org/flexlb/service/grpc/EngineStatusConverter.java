package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.TaskPhase;

import java.util.HashMap;
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

        // Convert proto enum to role name string (e.g., "DECODE")
        response.setRole(workerStatusPB.getRole().name());
        response.setAvailableConcurrency(workerStatusPB.getAvailableConcurrency());
        response.setRunningQueryLen(workerStatusPB.getRunningQueryLen());
        response.setWaitingQueryLen(workerStatusPB.getWaitingQueryLen());
        response.setStepLatencyMs(workerStatusPB.getStepLatencyMs());
        response.setIterateCount(workerStatusPB.getIterateCount());
        response.setDpSize(workerStatusPB.getDpSize());
        response.setTpSize(workerStatusPB.getTpSize());
        response.setDpRank(workerStatusPB.getDpRank());
        response.setStatusVersion(workerStatusPB.getStatusVersion());
        response.setLatestFinishedVersion(workerStatusPB.getLatestFinishedVersion());
        response.setAlive(workerStatusPB.getAlive());
        response.setAvailableKvCacheTokens(workerStatusPB.getAvailableKvCache());

        response.setRunningTaskInfo(convertToTaskInfoList(workerStatusPB.getRunningTaskInfoList()));

        // Convert finished task list
        response.setFinishedTaskInfo(convertToTaskInfoList(workerStatusPB.getFinishedTaskListList()));

        return response;
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
        Set<Long> cachedKeysSet = cacheKeysMap.keySet();
        cacheStatus.setCachedKeys(cachedKeysSet);
        cacheStatus.setCacheKeySize(cacheKeysMap.size());
        return cacheStatus;
    }

    /**
     * Convert list of TaskInfoPB to list of TaskInfo
     */
    private static Map<String, TaskInfo> convertToTaskInfoList(List<EngineRpcService.TaskInfoPB> taskInfoPBList) {
        if (taskInfoPBList == null) {
            return null;
        }
        Map<String, TaskInfo> taskInfoMap = new HashMap<>(taskInfoPBList.size());

        for (EngineRpcService.TaskInfoPB taskInfoPB : taskInfoPBList) {
            TaskInfo taskInfo = new TaskInfo();
            taskInfo.setRequestId(taskInfoPB.getRequestId());
            taskInfo.setPrefixLength(taskInfoPB.getPrefixLength());
            taskInfo.setInputLength(taskInfoPB.getInputLength());
            taskInfo.setWaitingTime(taskInfoPB.getWaitingTimeMs());
            taskInfo.setIterateCount(taskInfoPB.getIterateCount());
            taskInfo.setEndTimeMs(taskInfoPB.getEndTimeMs());
            taskInfo.setDpRank(taskInfoPB.getDpRank());
            taskInfo.setBatchId(taskInfoPB.getBatchId());
            taskInfo.setPhase(convertPhase(taskInfoPB.getPhase()));
            if (taskInfoPB.hasErrorInfo() && taskInfoPB.getErrorInfo().getErrorCode() != 0L) {
                taskInfo.setErrorCode(taskInfoPB.getErrorInfo().getErrorCode());
                taskInfo.setErrorMessage(taskInfoPB.getErrorInfo().getErrorMessage());
            }

            taskInfoMap.put(String.valueOf(taskInfoPB.getRequestId()), taskInfo);
        }

        return taskInfoMap;
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
