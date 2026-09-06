package org.flexlb.sync.schedule;

import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.TaskConfirmationTimeoutPvLog;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Periodically evicts workers that have stopped sending WorkerStatus reports
 * (crash, network partition, OOM kill, etc.) from the routing tables.
 *
 * <p>Without this cleaner, the Master would keep routing requests to dead
 * workers, producing a flood of 8400 / 8513 errors (the "decode death spiral"):
 * every dispatched request times out on a dead endpoint, the request is
 * retried onto another stale entry, and the cycle amplifies until the entire
 * decode fleet appears saturated. This component is the backstop that breaks
 * the cycle — once a worker's last report is older than
 * {@code workerRegistry.health.statusStaleAfterMs}, the entry is removed from both
 * {@link EngineWorkerStatus} and {@link EndpointRegistry}, forcing the
 * scheduler to rediscover live workers on the next sync round.
 *
 * <p>Runs every {@code WORKER_CLEAN_INTERVAL_MS} (default 3 s) via Spring
 * {@link Scheduled}. The timeout is intentionally generous (3× gRPC timeout)
 * to avoid racing a transient gRPC delay and evicting a still-alive endpoint.
 */
@Component
public class ExpirationCleaner {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final long workerTimeoutUs;
    private final long taskConfirmationTimeoutUs;
    private final EndpointRegistry endpointRegistry;

    @Autowired
    public ExpirationCleaner(EndpointRegistry endpointRegistry, ConfigService configService) {
        this(endpointRegistry, resolveWorkerTimeoutUs(configService),
                TimeUnit.MILLISECONDS.toMicros(configService.loadBalanceConfig()
                        .getWorkerRegistry().getHealth().getTaskConfirmationTimeoutMs()));
    }

    /**
     * Resolve the worker expiration timeout in microseconds.
     *
     * <p>The worker registry owns this timeout. It is configured in milliseconds
     * and converted to the monotonic microsecond clock used by WorkerStatus.
     * The default 10 s is 2× the gRPC sync timeout (5 s), eliminating the race
     * where a transient gRPC delay causes the cleaner to evict a still-alive endpoint.
     */
    private static long resolveWorkerTimeoutUs(ConfigService configService) {
        long configMs = configService.loadBalanceConfig().getWorkerRegistry()
                .getHealth().getStatusStaleAfterMs();
        return configMs * 1000L;
    }

    ExpirationCleaner(EndpointRegistry endpointRegistry, long workerTimeoutUs) {
        this(endpointRegistry, workerTimeoutUs, TimeUnit.MILLISECONDS.toMicros(300_000));
    }

    ExpirationCleaner(EndpointRegistry endpointRegistry, long workerTimeoutUs,
                      long taskConfirmationTimeoutUs) {
        this.endpointRegistry = endpointRegistry;
        this.workerTimeoutUs = workerTimeoutUs;
        this.taskConfirmationTimeoutUs = taskConfirmationTimeoutUs;
    }

    @Scheduled(fixedRateString = "${WORKER_CLEAN_INTERVAL_MS:3000}")
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
                workerStatus.setAlive(false);
                boolean statusRemoved = workerStatusMap.remove(item.getKey(), workerStatus);
                boolean endpointRemoved = endpointRegistry.remove(role, item.getKey(), workerStatus);
                if (statusRemoved || endpointRemoved) {
                    logger.warn("Removed expired worker: {}, role: {}, statusRemoved={}, endpointRemoved={}",
                            item.getKey(), role, statusRemoved, endpointRemoved);
                }
                continue;
            }

            ConcurrentHashMap<String, TaskInfo> localTasks = workerStatus.getLocalTaskMap();
            Iterator<Map.Entry<String, TaskInfo>> taskIterator = localTasks.entrySet().iterator();
            boolean taskSetChanged = false;
            while (taskIterator.hasNext()) {
                Map.Entry<String, TaskInfo> taskEntry = taskIterator.next();
                TaskInfo task = taskEntry.getValue();
                boolean confirmationTimedOut = task.getTaskState() == TaskStateEnum.IN_TRANSIT
                        && task.isTimeout(currentTime, taskConfirmationTimeoutUs);
                if (!task.isLost() && !confirmationTimedOut) {
                    continue;
                }

                if (confirmationTimedOut) {
                    reportTaskConfirmationTimeout(task, workerStatus, role, currentTime);
                }
                task.updateTaskState(TaskStateEnum.CLEANED);
                decrementQueueTime(workerStatus.getRunningQueueTime(), task, workerStatus.getRole());
                taskIterator.remove();
                taskSetChanged = true;
            }
            if (taskSetChanged) {
                workerStatus.refreshInTransitAndWaitingStats();
                workerStatus.refreshRunningRemainingPrefillTokens();
            }
        }
    }

    private void reportTaskConfirmationTimeout(TaskInfo task, WorkerStatus workerStatus,
                                               RoleType role, long currentTimeUs) {
        TaskConfirmationTimeoutPvLog event = new TaskConfirmationTimeoutPvLog(
                TaskConfirmationTimeoutPvLog.EVENT_TYPE,
                task.getRequestId(),
                role.getCode(),
                workerStatus.getIp(),
                workerStatus.getPort(),
                task.getTaskState().getValue(),
                TimeUnit.MICROSECONDS.toMillis(currentTimeUs - task.getLastActiveTimeUs()),
                TimeUnit.MICROSECONDS.toMillis(taskConfirmationTimeoutUs),
                task.getInputLength(),
                task.getPredictedPrefixLength(),
                task.getCacheMatchSource(),
                task.estimatePrefillTime());
        String eventJson = JsonUtils.toStringOrEmpty(event);
        logger.warn("Task confirmation timed out: {}", eventJson);
        pvLogger.info(eventJson);
    }

    private static void decrementQueueTime(
            AtomicLong runningQueueTime, TaskInfo task, RoleType role) {
        if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
            WorkerStatus.safeDecrementQueueTime(
                    runningQueueTime, task.estimatePrefillTime());
        }
    }
}
