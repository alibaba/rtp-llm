package org.flexlb.sync.schedule;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
@Component
public class ExpirationCleaner {

    private static final long TASK_TIME_OUT_US = 1000 * 60 * 1000L;

    @Scheduled(fixedRate = 3000)
    public void cleanExpiredWorkers() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.get("engine_service");
        if (modelWorkerStatus == null) {
            log.error("modelWorkerStatus is null, modelName: engine_service");
            return;
        }
        doClean(modelWorkerStatus.getPrefillStatusMap());
        doClean(modelWorkerStatus.getDecodeStatusMap());
        doClean(modelWorkerStatus.getPdFusionStatusMap());
        doClean(modelWorkerStatus.getVitStatusMap());
    }

    public static void doClean(Map<String, WorkerStatus> workerStatusMap) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return;
        }
        long curTimeMillis = System.nanoTime() / 1000;
        for (Iterator<Map.Entry<String, WorkerStatus>> it = workerStatusMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, WorkerStatus> item = it.next();
            WorkerStatus workerStatus = item.getValue();
            long expirationTime = workerStatus.getStatusLastUpdateTime().get() + 3 * 1000 * 1000L; // 3秒
            if (curTimeMillis > expirationTime) {
                it.remove();
            }

            // 删除运行队列中的超时任务
            ConcurrentHashMap<Long, TaskInfo> localTaskMap = workerStatus.getLocalTaskMap();
            long currentTime = System.nanoTime() / 1000;
            localTaskMap.forEach((requestId, task) -> {
                if (isTaskTimeout(task, currentTime)) {
                    localTaskMap.computeIfPresent(requestId, (k, existing) -> {
                        log.warn("Removing timeout task: {}", requestId);
                        return decrementQueueTime(workerStatus.getRunningQueueTime(), existing, workerStatus.getRole());
                    });
                }
            });
        }
    }

    private static boolean isTaskTimeout(TaskInfo task, long currentTime) {
        // 使用任务开始时间或创建时间判断超时
        long lastAccessTime = task.getLastActiveTimeUs();
        return (currentTime - lastAccessTime) > TASK_TIME_OUT_US;
    }

    private static TaskInfo decrementQueueTime(AtomicLong runningQueueTime, TaskInfo task, String role) {
        if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
            long delta = task.estimatePrefillTime();
            WorkerStatus.safeDecrementQueueTime(runningQueueTime, delta);
        }
        return null; // 返回null表示删除条目
    }
}