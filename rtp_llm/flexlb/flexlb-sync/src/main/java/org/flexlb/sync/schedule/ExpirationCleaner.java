package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;

@Component
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final long workerTimeoutUs;
    private final EndpointRegistry endpointRegistry;

    @Autowired
    public ExpirationCleaner(EndpointRegistry endpointRegistry) {
        this(endpointRegistry,
                Long.parseLong(System.getenv().getOrDefault("WORKER_TIMEOUT_US", "3000000")));
    }

    ExpirationCleaner(EndpointRegistry endpointRegistry, long workerTimeoutUs) {
        this.endpointRegistry = endpointRegistry;
        this.workerTimeoutUs = workerTimeoutUs;
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

            long expirationTime = workerStatus.getStatusLastUpdateTime().get() + workerTimeoutUs;
            long currentTime = System.nanoTime() / 1000;
            if (currentTime > expirationTime) {
                logger.info("Removing expired worker: {}, role: {}", item.getKey(), role);
                workerStatus.setAlive(false);
                workerStatusMap.remove(item.getKey(), workerStatus);
                endpointRegistry.remove(role, item.getKey(), workerStatus);
            }

            // 2. Check if tasks within worker need cleanup: lost tasks and long-timeout tasks
            ConcurrentHashMap<Long, TaskInfo> localTaskMap = workerStatus.getLocalTaskMap();
            Iterator<Map.Entry<Long, TaskInfo>> taskIterator = localTaskMap.entrySet().iterator();
            while (taskIterator.hasNext()) {
                Map.Entry<Long, TaskInfo> entry = taskIterator.next();
                Long requestId = entry.getKey();
                TaskInfo task = entry.getValue();

                boolean shouldRemove = false;

                // Check if task is lost
                if (task.isLost()) {
                    Logger.warn("Cleaning lost task: {}, state: {}, role: {}, worker: {}", requestId, task.getTaskState(), role, workerStatus.getIp());
                    reportTaskRemoved(workerStatus.getRole(), "lost");
                    task.updateTaskState(TaskStateEnum.CLEANED);
                    shouldRemove = true;
                }
                // Check if task is timed out
                else if (task.isTimeout(currentTime, taskTimeoutUs)) {
                    Logger.warn("Removing timeout task: {}, state: {}, age: {}ms, role: {}, worker: {}", requestId, task.getTaskState(),
                            (currentTime - task.getLastActiveTimeUs()) / 1000, role, workerStatus.getIp());
                    reportTaskRemoved(workerStatus.getRole(), "timeout");
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

    private void reportTaskRemoved(String role, String type) {
        FlexMetricTags tags = FlexMetricTags.of(
            "role", role,
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
