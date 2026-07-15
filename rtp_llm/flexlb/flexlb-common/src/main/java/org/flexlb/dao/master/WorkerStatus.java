package org.flexlb.dao.master;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.pv.CacheHitComparisonPvLog;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

@Data
@Slf4j
public class WorkerStatus {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger("syncLogger");
    public final transient ReentrantLock lock = new ReentrantLock();
    private String role;
    private String group;
    private String ip;
    private int port;
    private String site;
    private Long availableConcurrency;
    private boolean alive;
    private AtomicLong availableKvCacheTokens = new AtomicLong();
    private AtomicLong usedKvCacheTokens = new AtomicLong();
    private CacheStatus cacheStatus;
    private AtomicLong runningQueueTime = new AtomicLong();
    private Map<String, TaskInfo> waitingTaskList;
    private Map<String, TaskInfo> runningTaskList;
    private AtomicLong latestFinishedTaskVersion = new AtomicLong(-1L);

    private ConcurrentHashMap<String/*requestId*/, TaskInfo> localTaskMap = new ConcurrentHashMap<>();
    private double stepLatencyMs;
    private long iterateCount;
    private long dpSize;
    private long tpSize;
    private int blockHashLookaheadTokens;

    private AtomicLong statusLastUpdateTime = new AtomicLong(-1); // Last status update time (microseconds)
    private AtomicLong statusUpdateIntervalUs = new AtomicLong(0); // Actual interval between last two status updates (microseconds)
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1); // Last cache status update time
    private AtomicLong lastSelectedTime = new AtomicLong(-1); // Last selection time
    private AtomicBoolean resourceAvailable = new AtomicBoolean(true); // Resource availability state
    private AtomicBoolean statusCheckInProgress = new AtomicBoolean(false); // Status check in progress flag
    private AtomicBoolean cacheCheckInProgress = new AtomicBoolean(false); // Cache check in progress flag
    private AtomicLong statusVersion = new AtomicLong(-1L);

    /**
     * Add task to local running queue
     * @param requestId Request ID
     * @param taskInfo Task information
     */
    public void putLocalTask(String requestId, TaskInfo taskInfo) {
        localTaskMap.put(requestId, taskInfo);
        taskInfo.updateTaskState(TaskStateEnum.IN_TRANSIT);

        // Local incremental queue time update
        this.addRunningQueueTime(taskInfo.estimatePrefillTime());
        // Local incremental KV cache tokens update
        long needNewKvCacheLen = taskInfo.getInputLength() - taskInfo.getPrefixLength();
        this.decKvCacheFree(needNewKvCacheLen);
        this.addKvCacheUsed(needNewKvCacheLen);

        lastSelectedTime.set(System.nanoTime() / 1000);
        Logger.debug("Task {} added to local queue with state: {}", requestId, TaskStateEnum.IN_TRANSIT);
    }

    /**
     * Remove task from local running queue
     * @param requestId Request ID
     */
    public void removeLocalTask(String requestId) {
        TaskInfo taskInfo = localTaskMap.get(requestId);
        if (taskInfo != null) {
            safeDecrementQueueTime(runningQueueTime, taskInfo.estimatePrefillTime());
            long needNewKvCacheLen = taskInfo.getInputLength() - taskInfo.getPrefixLength();
            decKvCacheFree(-needNewKvCacheLen);
            addKvCacheUsed(-needNewKvCacheLen);
            localTaskMap.remove(requestId);
        }
    }

    /**
     * Add estimated execution time to running queue
     * @param len Estimated execution time to add
     */
    public void addRunningQueueTime(long len) {
        runningQueueTime.addAndGet(len);
    }

    public void addKvCacheUsed(long len) {
        usedKvCacheTokens.addAndGet(len);
    }

    public void decKvCacheFree(long len) {
        availableKvCacheTokens.accumulateAndGet(len, (current, decrement) ->
                Math.max(0, current - decrement));
    }

    /**
     * Update task states
     * Check for lost tasks, update running/waiting tasks, and clean up finished tasks
     */
    public List<CacheHitComparisonPvLog> updateTaskStates(
            Map<String, TaskInfo> waitingTaskInfo,
            Map<String, TaskInfo> runningTaskInfo,
            Map<String, TaskInfo> finishedTaskInfo) {
        List<CacheHitComparisonPvLog> cacheHitComparisons = Collections.emptyList();
        Iterator<Map.Entry<String, TaskInfo>> iterator = localTaskMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, TaskInfo> entry = iterator.next();
            String requestId = entry.getKey();
            TaskInfo localTask = entry.getValue();

            TaskInfo finishedTask = finishedTaskInfo != null ? finishedTaskInfo.get(requestId) : null;
            if (finishedTask != null) {
                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker", requestId);
                }
                localTask.updateTaskState(TaskStateEnum.FINISHED);
                updateTaskInputLength(localTask, finishedTask);
                cacheHitComparisons = appendCacheHitComparison(
                        cacheHitComparisons,
                        applyActualCacheHit(localTask, finishedTask, "finished"));

                if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
                    long delta = localTask.estimatePrefillTime();
                    safeDecrementQueueTime(runningQueueTime, delta);
                }
                Logger.debug("Task {} finished and removed", requestId);
                iterator.remove();
                continue;
            }

            TaskInfo runningTask = runningTaskInfo != null ? runningTaskInfo.get(requestId) : null;
            if (runningTask != null) {
                localTask.setLastActiveTimeUs(System.nanoTime() / 1000);

                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker", requestId);
                }
                if (localTask.getTaskState() != TaskStateEnum.RUNNING) {
                    localTask.updateTaskState(TaskStateEnum.RUNNING);
                }

                updateTaskInputLength(localTask, runningTask);
                cacheHitComparisons = appendCacheHitComparison(
                        cacheHitComparisons,
                        applyActualCacheHit(localTask, runningTask, "running"));
                localTask.setPrefillTime(runningTask.getPrefillTime());
                localTask.setWaitingTime(runningTask.getWaitingTime());
                localTask.setIterateCount(runningTask.getIterateCount());
                localTask.setEndTimeMs(runningTask.getEndTimeMs());
                localTask.setDpRank(runningTask.getDpRank());

                continue;
            }

            TaskInfo waitingTask = waitingTaskInfo != null ? waitingTaskInfo.get(requestId) : null;
            if (waitingTask != null) {
                localTask.setLastActiveTimeUs(System.nanoTime() / 1000);

                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker (waiting)", requestId);
                }

                updateTaskInputLength(localTask, waitingTask);
                cacheHitComparisons = appendCacheHitComparison(
                        cacheHitComparisons,
                        applyActualCacheHit(localTask, waitingTask, "waiting"));
                localTask.setWaitingTime(waitingTask.getWaitingTime());
                localTask.setDpRank(waitingTask.getDpRank());

                continue;
            }

            if (localTask.getTaskState() == TaskStateEnum.CONFIRMED || localTask.getTaskState() == TaskStateEnum.RUNNING) {
                localTask.updateTaskState(TaskStateEnum.LOST);
                logger.warn("Task {} marked as LOST - not in waiting, running or finished list", requestId);
            }
        }
        return cacheHitComparisons;
    }

    private void updateTaskInputLength(TaskInfo localTask, TaskInfo engineTask) {
        if (engineTask.getInputLength() > 0) {
            localTask.setInputLength(engineTask.getInputLength());
        }
    }

    private List<CacheHitComparisonPvLog> appendCacheHitComparison(
            List<CacheHitComparisonPvLog> comparisons,
            CacheHitComparisonPvLog comparison) {
        if (comparison == null) {
            return comparisons;
        }
        if (comparisons.isEmpty()) {
            comparisons = new ArrayList<>();
        }
        comparisons.add(comparison);
        return comparisons;
    }

    private CacheHitComparisonPvLog applyActualCacheHit(
            TaskInfo localTask,
            TaskInfo engineTask,
            String taskState) {
        if (!engineTask.isPrefixLengthValid()) {
            if (localTask.isPrefixLengthValid()) {
                long previousPrefillTime = localTask.estimatePrefillTime();
                localTask.setPrefixLength(localTask.getPredictedPrefixLength());
                localTask.setPrefixLengthValid(false);
                correctRunningQueueTime(localTask.estimatePrefillTime() - previousPrefillTime);
            }
            return null;
        }

        boolean cacheHitBecameValid = !localTask.isPrefixLengthValid();
        long previousPrefillTime = localTask.estimatePrefillTime();
        localTask.setPrefixLength(engineTask.getPrefixLength());
        localTask.setPrefixLengthValid(true);
        correctRunningQueueTime(localTask.estimatePrefillTime() - previousPrefillTime);

        if (!cacheHitBecameValid) {
            return null;
        }

        long predictedHitTokens = localTask.getPredictedPrefixLength();
        long actualHitTokens = localTask.getPrefixLength();
        long inputTokens = localTask.getInputLength();
        long blockSize = cacheStatus == null ? 0 : cacheStatus.getBlockSize();
        return new CacheHitComparisonPvLog(
                "cache_hit_comparison",
                localTask.getRequestId(),
                localTask.getCacheMatchSource(),
                role,
                group,
                ip,
                port,
                taskState,
                inputTokens,
                blockSize,
                predictedHitTokens,
                actualHitTokens,
                actualHitTokens - predictedHitTokens);
    }

    private void correctRunningQueueTime(long correction) {
        if (correction == 0
                || (!RoleType.PREFILL.matches(role) && !RoleType.PDFUSION.matches(role))) {
            return;
        }
        runningQueueTime.accumulateAndGet(
                correction,
                (current, change) -> Math.max(0, current + change));
    }

    /**
     * Update total queue time for running queue
     */
    public void updateRunningQueueTime() {
        int localTaskMapSize = localTaskMap.size();
        if (localTaskMapSize == 0) {
            runningQueueTime.getAndSet(0);
            return;
        }
        long rectifiedEstimateRunningTime = 0;
        for (Entry<String, TaskInfo> entry : localTaskMap.entrySet()) {
            TaskInfo taskInfo = entry.getValue();
            // Recalculate based on accurate cache hit count, rectify local task running queue time
            rectifiedEstimateRunningTime += taskInfo.estimatePrefillTime();
        }
        if (RoleType.PREFILL.matches(role) || RoleType.PDFUSION.matches(role)) {
            // Actual cache-hit corrections are applied incrementally in both directions.
            // This reconciliation only repairs an overestimated aggregate.
            if (runningQueueTime.get() > rectifiedEstimateRunningTime) {
                runningQueueTime.getAndSet(rectifiedEstimateRunningTime);
            }
        }
    }

    public void updateKvCacheTokens(long latestUsedKvCacheTokens, long latestAvailableKvCacheTokens) {

        int localTaskMapSize = localTaskMap.size();
        if (localTaskMapSize == 0) {
            usedKvCacheTokens.getAndSet(latestUsedKvCacheTokens);
            availableKvCacheTokens.getAndSet(latestAvailableKvCacheTokens);
            return;
        }

        long inTransitTaskCacheUsed = 0;
        for (Map.Entry<String, TaskInfo> entry : localTaskMap.entrySet()) {
            TaskInfo taskInfo = entry.getValue();
            // Calculate tokens occupied by in-transit task cache miss portion
            if (taskInfo.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                inTransitTaskCacheUsed = inTransitTaskCacheUsed + taskInfo.getInputLength() - taskInfo.getPrefixLength();
            }
        }
        // Rectify KV cache tokens affected by in-transit tasks
        latestUsedKvCacheTokens += inTransitTaskCacheUsed;
        latestAvailableKvCacheTokens -= inTransitTaskCacheUsed;

        usedKvCacheTokens.getAndSet(latestUsedKvCacheTokens);
        availableKvCacheTokens.getAndSet(latestAvailableKvCacheTokens);

    }

    /**
     * Safely decrement total queue time for running queue, ensuring it never becomes negative
     *
     * @param runningQueueTime Total queue time for running queue
     * @param timeToReduce Time to reduce
     */
    public static void safeDecrementQueueTime(AtomicLong runningQueueTime, long timeToReduce) {
        if (timeToReduce <= 0) {
            logger.warn("Invalid tokens to reduce: {}", timeToReduce);
            return;
        }
        runningQueueTime.accumulateAndGet(timeToReduce, (currentRunningQueueTime, reductionAmount) -> {
            // Ensure reduction amount is positive, calculate new value, but not less than 0
            long newRunningQueueTime = currentRunningQueueTime - reductionAmount;

            // If result is negative, set to 0, ensuring token count never goes below 0
            return Math.max(newRunningQueueTime, 0L);
        });
    }

    /**
     * Update resource availability with hysteresis to prevent state oscillation.
     * <p>
     * Hysteresis uses two thresholds: upper and lower (calculated as upper - hysteresisBias%).
     * This creates a band where state doesn't change, preventing rapid toggling.
     * <p>
     * State transitions:
     * - AVAILABLE → UNAVAILABLE: when current metric EXCEEDS upper threshold
     * - UNAVAILABLE → AVAILABLE: when current metric FALLS BELOW lower threshold
     *
     * @param currentMetric current resource metric value
     * @param upperThreshold upper threshold for disabling availability
     * @param hysteresisBias bias percentage for calculating lower threshold (lower = upper - upper * bias / 100)
     * @return the new resource availability state
     */
    public boolean updateResourceAvailabilityWithHysteresis(long currentMetric, long upperThreshold, long hysteresisBias) {
        long lowerThreshold = Math.max(0, upperThreshold - (long)(upperThreshold * hysteresisBias / 100.0));

        if (currentMetric >= upperThreshold) {
            resourceAvailable.compareAndSet(true, false);
        } else if (currentMetric <= lowerThreshold) {
            resourceAvailable.compareAndSet(false, true);
        }
        return resourceAvailable.get();
    }

    /**
     * Get IP:PORT format address
     *
     * @return IP:PORT string
     */
    public String getIpPort() {
        if (ip == null) {
            return null;
        }
        return ip + ":" + port;
    }
}
