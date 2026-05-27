package org.flexlb.service.grpc;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.DpRankCacheStatus;
import org.flexlb.dao.master.DpRankStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;

import java.util.ArrayList;
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

        // Set role directly as string
        response.setRole(workerStatusPB.getRole());
        response.setAvailableConcurrency(workerStatusPB.getAvailableConcurrency());
        response.setRunningQueryLen(workerStatusPB.getRunningQueryLen());
        response.setWaitingQueryLen(workerStatusPB.getWaitingQueryLen());
        response.setStepLatencyMs(workerStatusPB.getStepLatencyMs());
        response.setIterateCount(workerStatusPB.getIterateCount());
        response.setDpSize(workerStatusPB.getDpSize());
        response.setTpSize(workerStatusPB.getTpSize());
        response.setStatusVersion(workerStatusPB.getStatusVersion());
        response.setLatestFinishedVersion(workerStatusPB.getLatestFinishedVersion());
        response.setAlive(workerStatusPB.getAlive());

        List<EngineRpcService.TaskInfoPB> srcRunningTaskInfoList = workerStatusPB.getRunningTaskInfoList();
        List<EngineRpcService.TaskInfoPB> waitingTaskInfoList = srcRunningTaskInfoList.stream().filter(taskInfoPB -> taskInfoPB.getIsWaiting()).toList();
        List<EngineRpcService.TaskInfoPB> runningTaskInfoList = srcRunningTaskInfoList.stream().filter(taskInfoPB -> !taskInfoPB.getIsWaiting()).toList();

        // Convert waiting task info
        response.setWaitingTaskInfo(convertToTaskInfoList(waitingTaskInfoList));

        // Convert running task info
        response.setRunningTaskInfo(convertToTaskInfoList(runningTaskInfoList));

        // Convert finished task list
        response.setFinishedTaskInfo(convertToTaskInfoList(workerStatusPB.getFinishedTaskListList()));

        if (workerStatusPB.getDpSize() > 1 && workerStatusPB.getDpStatusCount() > 0) {
            response.setDpStatuses(convertDpStatusList(workerStatusPB.getDpStatusList()));
        }

        return response;
    }

    private static List<DpRankStatus> convertDpStatusList(List<EngineRpcService.WorkerStatusPB> dpStatusList) {
        List<DpRankStatus> out = new ArrayList<>(dpStatusList.size());
        for (int rank = 0; rank < dpStatusList.size(); rank++) {
            EngineRpcService.WorkerStatusPB pb = dpStatusList.get(rank);
            out.add(new DpRankStatus(
                    rank,
                    pb.getIp(),
                    pb.getGrpcPort(),
                    pb.getAvailableConcurrency(),
                    pb.getRunningQueryLen(),
                    pb.getWaitingQueryLen(),
                    pb.getIterateCount(),
                    pb.getAlive()));
        }
        return out;
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

        if (cacheStatusPB.getDpCacheCount() > 0) {
            cacheStatus.setDpCaches(convertDpCacheList(cacheStatusPB.getDpCacheList()));
        }

        return cacheStatus;
    }

    private static List<DpRankCacheStatus> convertDpCacheList(List<EngineRpcService.CacheStatusPB> dpCacheList) {
        List<DpRankCacheStatus> out = new ArrayList<>(dpCacheList.size());
        for (int rank = 0; rank < dpCacheList.size(); rank++) {
            EngineRpcService.CacheStatusPB pb = dpCacheList.get(rank);
            out.add(new DpRankCacheStatus(
                    rank,
                    pb.getIp(),
                    pb.getGrpcPort(),
                    pb.getAvailableKvCache(),
                    pb.getTotalKvCache(),
                    pb.getBlockSize(),
                    new HashSet<>(pb.getCacheKeysMap().keySet())));
        }
        return out;
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
            taskInfo.setPrefillTime(taskInfoPB.getPrefillTimeUs());

            taskInfoMap.put(String.valueOf(taskInfoPB.getRequestId()), taskInfo);
        }

        return taskInfoMap;
    }
}