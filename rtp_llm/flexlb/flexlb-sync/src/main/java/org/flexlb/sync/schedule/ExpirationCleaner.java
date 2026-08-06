package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.TaskConfirmationTimeoutPvLog;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class ExpirationCleaner {

    private static final String TASK_REMOVED = "task.removed";
    private static final org.slf4j.Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final long taskConfirmTimeoutUs;
    private final long workerTimeoutUs;
    private final FlexMonitor monitor;

    public ExpirationCleaner(FlexMonitor monitor, ConfigService configService) {
        this.monitor = monitor;
        this.taskConfirmTimeoutUs = TimeUnit.MILLISECONDS.toMicros(configService.loadBalanceConfig().getTaskConfirmTimeoutMs());
        this.workerTimeoutUs = Long.parseLong(System.getenv().getOrDefault("WORKER_TIMEOUT_US", "3000000"));
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

            // 1. Check if worker needs cleanup
            long expirationTime = workerStatus.getStatusLastUpdateTime().get() + workerTimeoutUs;
            long currentTime = System.nanoTime() / 1000;
            if (currentTime > expirationTime) {
                it.remove();
                continue;
            }

            // 2. Check if tasks within worker need cleanup: lost tasks and long-timeout tasks
            ConcurrentHashMap<String, TaskInfo> localTaskMap = workerStatus.getLocalTaskMap();
            Iterator<Map.Entry<String, TaskInfo>> taskIterator = localTaskMap.entrySet().iterator();
            boolean pendingQueueChanged = false;
            while (taskIterator.hasNext()) {
                Map.Entry<String, TaskInfo> entry = taskIterator.next();
                String requestId = entry.getKey();
                TaskInfo task = entry.getValue();

                boolean shouldRemove = false;

                // Check if task is lost
                if (task.isLost()) {
                    Logger.warn("Cleaning lost task: {}, state: {}, role: {}, worker: {}", requestId, task.getTaskState(), role, workerStatus.getIp());
                    reportTaskRemoved(workerStatus.getRole(), workerStatus.getIp(), "lost");
                    task.updateTaskState(TaskStateEnum.CLEANED);
                    shouldRemove = true;
                }
                // Keep the local prediction until WorkerStatus confirms the task or this window expires.
                else if (task.getTaskState() == TaskStateEnum.IN_TRANSIT && task.isTimeout(currentTime, taskConfirmTimeoutUs)) {
                    reportTaskConfirmationTimeout(requestId, task, workerStatus, role, currentTime);
                    reportTaskRemoved(workerStatus.getRole(), workerStatus.getIp(), "timeout");
                    task.updateTaskState(TaskStateEnum.CLEANED);
                    shouldRemove = true;
                }

                if (shouldRemove) {
                    decrementQueueTime(workerStatus.getRunningQueueTime(), task, workerStatus.getRole());
                    taskIterator.remove();
                    pendingQueueChanged = true;
                }
            }
            if (pendingQueueChanged) {
                workerStatus.refreshInTransitAndWaitingStats();
            }
        }
    }

    private void reportTaskConfirmationTimeout(String requestId, TaskInfo task, WorkerStatus workerStatus,
                                               RoleType role, long currentTimeUs) {
        TaskConfirmationTimeoutPvLog event = new TaskConfirmationTimeoutPvLog(
                TaskConfirmationTimeoutPvLog.EVENT_TYPE,
                requestId,
                role.getCode(),
                workerStatus.getIp(),
                workerStatus.getPort(),
                task.getTaskState().getValue(),
                TimeUnit.MICROSECONDS.toMillis(currentTimeUs - task.getLastActiveTimeUs()),
                TimeUnit.MICROSECONDS.toMillis(taskConfirmTimeoutUs),
                task.getInputLength(),
                task.getPredictedPrefixLength(),
                task.getCacheMatchSource(),
                task.estimatePrefillTime());
        String eventJson = JsonUtils.toStringOrEmpty(event);
        Logger.warn("Task confirmation timed out: {}", eventJson);
        pvLogger.info(eventJson);
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
