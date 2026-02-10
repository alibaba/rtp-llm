package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class ExpirationCleaner {

    private static final String TASK_REMOVED = "task.removed";
    
    private final long taskTimeoutUs;
    private final long workerTimeoutUs;
    private final FlexMonitor monitor;

    public ExpirationCleaner(FlexMonitor monitor) {
        this.monitor = monitor;
        this.taskTimeoutUs = Long.parseLong(System.getenv().getOrDefault("TASK_TIMEOUT_US", "3000000"));  // 默认3s
        this.workerTimeoutUs = Long.parseLong(System.getenv().getOrDefault("WORKER_TIMEOUT_US", "3000000")); // 默认3s
    }

    @PostConstruct
    public void init() {
        this.monitor.register(TASK_REMOVED, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    @Scheduled(fixedRate = 3000)
    public void cleanExpiredWorkers() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        this.doClean(modelWorkerStatus.getPrefillStatusMap(), RoleType.PREFILL);
        this.doClean(modelWorkerStatus.getDecodeStatusMap(), RoleType.DECODE);
        this.doClean(modelWorkerStatus.getPdFusionStatusMap(), RoleType.PDFUSION);
        this.doClean(modelWorkerStatus.getVitStatusMap(), RoleType.VIT);
    }

    public void doClean(Map<String, WorkerStatus> workerStatusMap, RoleType role) {
        if (MapUtils.isEmpty(workerStatusMap)) {
            return;
        }

        for (Iterator<Map.Entry<String, WorkerStatus>> it = workerStatusMap.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<String, WorkerStatus> item = it.next();
            WorkerStatus workerStatus = item.getValue();

            // 1. 判断Worker是否需要清理
            long expirationTime = workerStatus.getStatusLastUpdateTime().get() + workerTimeoutUs;
            long currentTime = System.nanoTime() / 1000;
            if (currentTime > expirationTime) {
                it.remove();
                continue;
            }

            // 2. 判断Worker内的Task是否需要清理：丢失的任务和长时间超时的任务
            ConcurrentHashMap<String, TaskInfo> localTaskMap = workerStatus.getLocalTaskMap();
            Iterator<Map.Entry<String, TaskInfo>> taskIterator = localTaskMap.entrySet().iterator();
            while (taskIterator.hasNext()) {
                Map.Entry<String, TaskInfo> entry = taskIterator.next();
                String requestId = entry.getKey();
                TaskInfo task = entry.getValue();
                
                boolean shouldRemove = false;
                
                // 检查是否是丢失的任务
                if (task.isLost()) {
                    Logger.warn("Cleaning lost task: {}, state: {}, role: {}, worker: {}", requestId, task.getTaskState(), role, workerStatus.getIp());
                    reportTaskRemoved(workerStatus.getRole(), workerStatus.getIp(), "lost");
                    task.updateTaskState(TaskStateEnum.CLEANED);
                    shouldRemove = true;
                }
                // 检查是否是超时的任务
                else if (task.isTimeout(currentTime, taskTimeoutUs)) {
                    Logger.warn("Removing timeout task: {}, state: {}, age: {}ms, role: {}, worker: {}", requestId, task.getTaskState(),
                            (currentTime - task.getLastActiveTimeUs()) / 1000, role, workerStatus.getIp());
                    reportTaskRemoved(workerStatus.getRole(), workerStatus.getIp(), "timeout");
                    task.updateTaskState(TaskStateEnum.CLEANED);
                    shouldRemove = true;
                }
                
                if (shouldRemove) {
                    decrementQueueTime(workerStatus.getRunningQueueTime(), task, workerStatus.getRole());
                    taskIterator.remove();
                }
            }
        }
    }

    private void reportTaskRemoved(String role, String ip, String type) {
        FlexMetricTags tags = FlexMetricTags.of(
            "role", role, 
            "ip", ip,
            "type", type
        );
        monitor.report(TASK_REMOVED, tags, 1);
    }

    private static void decrementQueueTime(AtomicLong runningQueueTime, TaskInfo task, String role) {
        if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
            long delta = task.estimatePrefillTime();
            WorkerStatus.safeDecrementQueueTime(runningQueueTime, delta);
        }
    }
}