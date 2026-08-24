package org.flexlb.dao.master;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.constant.CommonConstants.LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR;

@Data
@Slf4j
public class WorkerStatus {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger("syncLogger");
    public final transient ReentrantLock lock = new ReentrantLock();
    private String role;
    private String group;
    private String deploymentName;
    private String endpointAddress = "";
    private String ip;
    private int port;
    private int engineIndex;
    private int multiEngineNum = 1;
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
    private volatile long inTransitAndWaitingTaskCount;
    private volatile long inTransitAndWaitingUncachedTokens;
    // Reported separately from IN_TRANSIT + WAITING so existing admission
    // accounting retains its established semantics.
    private volatile long runningRemainingPrefillTokens;
    private double stepLatencyMs;
    private long iterateCount;
    private long dpSize;
    private long tpSize;
    private int blockHashLookaheadTokens;
    private int cacheMatchRollbackBlocks;
    private KvCacheGroupMode kvCacheGroupMode = KvCacheGroupMode.UNSPECIFIED;

    private AtomicLong statusLastUpdateTime = new AtomicLong(-1); // Last status update time (microseconds)
    private AtomicLong statusUpdateIntervalUs = new AtomicLong(0); // Actual interval between last two status updates (microseconds)
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1); // Last cache status update time
    private AtomicLong lastSelectedTime = new AtomicLong(-1); // Last selection time
    private AtomicBoolean resourceAvailable = new AtomicBoolean(true); // Resource availability state
    private AtomicBoolean statusCheckInProgress = new AtomicBoolean(false); // Status check in progress flag
    private AtomicBoolean cacheCheckInProgress = new AtomicBoolean(false); // Cache check in progress flag
    private AtomicLong statusVersion = new AtomicLong(-1L);

    /** Returns the physical frontend address in {@code ip:port} format. */
    public String getPhysicalIpPort() {
        return ip == null ? null : ip + ":" + port;
    }

    /**
     * Returns the logical worker identity in {@code ip:port@engineIndex} format. The index
     * identifies one independently routable engine behind the physical frontend.
     */
    public String getLogicalIpPort() {
        String physicalIpPort = getPhysicalIpPort();
        return physicalIpPort == null
                ? null
                : physicalIpPort + LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR + engineIndex;
    }

    public String getPhysicalGroupKey() {
        return endpointAddress + "|" + group + "|" + getPhysicalIpPort();
    }

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
        refreshInTransitAndWaitingStats();
        refreshRunningRemainingPrefillTokens();
        Logger.debug("Task {} added to local queue with state: {}", requestId, TaskStateEnum.IN_TRANSIT);
    }

    public long getOutstandingUncachedTokens() {
        return Math.max(0, inTransitAndWaitingUncachedTokens) + Math.max(0, runningRemainingPrefillTokens);
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
            refreshInTransitAndWaitingStats();
            refreshRunningRemainingPrefillTokens();
        }
    }

    public long getInTransitTaskCount() {
        return localTaskMap.values().stream()
                .filter(taskInfo -> taskInfo.getTaskState() == TaskStateEnum.IN_TRANSIT)
                .count();
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
     *
     * @return outcomes produced by this task-state update
     */
    public TaskStateUpdateResult updateTaskStates(Map<String, TaskInfo> waitingTaskInfo,
                                                  Map<String, TaskInfo> runningTaskInfo,
                                                  Map<String, TaskInfo> finishedTaskInfo) {

        addObservedTasks(runningTaskInfo, TaskStateEnum.RUNNING);
        addObservedTasks(waitingTaskInfo, TaskStateEnum.CONFIRMED);

        List<CacheHitFeedback> cacheHitFeedbacks = new ArrayList<>();
        List<Long> decisionToWaitingObservedLatenciesMs = new ArrayList<>();
        List<Long> waitingToRunningObservedLatenciesMs = new ArrayList<>();
        List<Long> engineWaitingToRunningLatenciesMs = new ArrayList<>();
        List<Long> engineReceivedToWaitingLatenciesMs = new ArrayList<>();
        Iterator<Map.Entry<String, TaskInfo>> iterator = localTaskMap.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, TaskInfo> entry = iterator.next();
            String requestId = entry.getKey();
            TaskInfo localTask = entry.getValue();

            TaskInfo finishedTask = finishedTaskInfo != null ? finishedTaskInfo.get(requestId) : null;
            if (finishedTask != null) {
                boolean runningWasObserved = localTask.getTaskState() == TaskStateEnum.RUNNING;
                boolean receivedToWaitingWasObserved = localTask.getRequestReceivedTimeMs() > 0;
                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker", requestId);
                }
                localTask.updateTaskState(TaskStateEnum.FINISHED);
                updateTaskInputLength(localTask, finishedTask);
                updateCacheHitFromEngine(localTask, finishedTask, "finished", cacheHitFeedbacks);
                localTask.setRequestReceivedTimeMs(finishedTask.getRequestReceivedTimeMs());
                localTask.setWaitingEnteredTimeMs(finishedTask.getWaitingEnteredTimeMs());
                localTask.setRunningEnteredTimeMs(finishedTask.getRunningEnteredTimeMs());
                if (!receivedToWaitingWasObserved && finishedTask.getRequestReceivedTimeMs() > 0
                        && finishedTask.getWaitingEnteredTimeMs() > 0) {
                    engineReceivedToWaitingLatenciesMs.add(Math.max(0,
                            finishedTask.getWaitingEnteredTimeMs() - finishedTask.getRequestReceivedTimeMs()));
                }
                if (!runningWasObserved && finishedTask.getWaitingEnteredTimeMs() > 0
                        && finishedTask.getRunningEnteredTimeMs() > 0) {
                    engineWaitingToRunningLatenciesMs.add(Math.max(0,
                            finishedTask.getRunningEnteredTimeMs() - finishedTask.getWaitingEnteredTimeMs()));
                }

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
                long runningObservationTimeUs = System.nanoTime() / 1000;
                boolean firstRunningObservation = localTask.getTaskState() != TaskStateEnum.RUNNING;
                boolean receivedToWaitingWasObserved = localTask.getRequestReceivedTimeMs() > 0;

                if (localTask.getWaitingConfirmTimeUs() > 0) {
                    waitingToRunningObservedLatenciesMs.add(
                            Math.max(0, runningObservationTimeUs - localTask.getWaitingConfirmTimeUs()) / 1000);
                    localTask.setWaitingConfirmTimeUs(-1);
                }
                localTask.setLastActiveTimeUs(runningObservationTimeUs);

                if (localTask.getTaskState() == TaskStateEnum.IN_TRANSIT) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker", requestId);
                }
                if (localTask.getTaskState() != TaskStateEnum.RUNNING) {
                    localTask.updateTaskState(TaskStateEnum.RUNNING);
                }

                updateTaskInputLength(localTask, runningTask);
                updateCacheHitFromEngine(localTask, runningTask, "running", cacheHitFeedbacks);
                updatePrefillRunningProgressFromEngine(localTask, runningTask);
                localTask.setPrefillTime(runningTask.getPrefillTime());
                localTask.setWaitingTime(runningTask.getWaitingTime());
                localTask.setIterateCount(runningTask.getIterateCount());
                localTask.setEndTimeMs(runningTask.getEndTimeMs());
                localTask.setDpRank(runningTask.getDpRank());
                localTask.setRequestReceivedTimeMs(runningTask.getRequestReceivedTimeMs());
                localTask.setWaitingEnteredTimeMs(runningTask.getWaitingEnteredTimeMs());
                localTask.setRunningEnteredTimeMs(runningTask.getRunningEnteredTimeMs());

                if (!receivedToWaitingWasObserved && runningTask.getRequestReceivedTimeMs() > 0
                        && runningTask.getWaitingEnteredTimeMs() > 0) {
                    engineReceivedToWaitingLatenciesMs.add(Math.max(0,
                            runningTask.getWaitingEnteredTimeMs() - runningTask.getRequestReceivedTimeMs()));
                }
                if (firstRunningObservation && runningTask.getWaitingEnteredTimeMs() > 0 && runningTask.getRunningEnteredTimeMs() > 0) {
                    engineWaitingToRunningLatenciesMs.add(Math.max(0,
                            runningTask.getRunningEnteredTimeMs() - runningTask.getWaitingEnteredTimeMs()));
                }

                continue;
            }

            TaskInfo waitingTask = waitingTaskInfo != null ? waitingTaskInfo.get(requestId) : null;
            if (waitingTask != null) {
                boolean firstWaitingConfirmation = localTask.getTaskState() == TaskStateEnum.IN_TRANSIT;
                boolean receivedToWaitingWasObserved = localTask.getRequestReceivedTimeMs() > 0;
                long confirmationTimeUs = System.nanoTime() / 1000;
                long decisionToWaitingObservedMs = firstWaitingConfirmation
                        ? Math.max(0, confirmationTimeUs - localTask.getLastActiveTimeUs()) / 1000
                        : 0;
                localTask.setLastActiveTimeUs(confirmationTimeUs);
                if (firstWaitingConfirmation) {
                    localTask.setWaitingConfirmTimeUs(confirmationTimeUs);
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                    Logger.debug("Task {} first confirmed by worker (waiting)", requestId);
                }

                updateTaskInputLength(localTask, waitingTask);
                updateCacheHitFromEngine(localTask, waitingTask, "waiting", cacheHitFeedbacks);
                if (localTask.getTaskState() == TaskStateEnum.RUNNING) {
                    localTask.updateTaskState(TaskStateEnum.CONFIRMED);
                }
                localTask.setWaitingTime(waitingTask.getWaitingTime());
                localTask.setDpRank(waitingTask.getDpRank());
                localTask.setWaitingEnteredTimeMs(waitingTask.getWaitingEnteredTimeMs());
                localTask.setRequestReceivedTimeMs(waitingTask.getRequestReceivedTimeMs());
                if (firstWaitingConfirmation) {
                    decisionToWaitingObservedLatenciesMs.add(decisionToWaitingObservedMs);
                }
                if (!receivedToWaitingWasObserved && waitingTask.getRequestReceivedTimeMs() > 0
                        && waitingTask.getWaitingEnteredTimeMs() > 0) {
                    engineReceivedToWaitingLatenciesMs.add(
                            Math.max(0, waitingTask.getWaitingEnteredTimeMs() - waitingTask.getRequestReceivedTimeMs()));
                }

                continue;
            }

            if (localTask.getTaskState() == TaskStateEnum.CONFIRMED || localTask.getTaskState() == TaskStateEnum.RUNNING) {
                localTask.updateTaskState(TaskStateEnum.LOST);
                logger.warn("Task {} marked as LOST - not in waiting, running or finished list", requestId);
            }
        }
        refreshInTransitAndWaitingStats();
        refreshRunningRemainingPrefillTokens();
        return TaskStateUpdateResult.from(
                cacheHitFeedbacks,
                decisionToWaitingObservedLatenciesMs,
                waitingToRunningObservedLatenciesMs,
                engineWaitingToRunningLatenciesMs,
                engineReceivedToWaitingLatenciesMs
        );
    }

    private void addObservedTasks(Map<String, TaskInfo> tasks, TaskStateEnum taskState) {
        if (tasks == null) {
            return;
        }
        tasks.forEach((requestId, taskInfo) -> {
            if (requestId == null || taskInfo == null) {
                return;
            }
            // Atomically add tasks first observed in worker status. A null return value means
            // this request was absent and prevents repeated status updates from counting it twice.
            if (localTaskMap.putIfAbsent(requestId, taskInfo) == null) {
                taskInfo.updateTaskState(taskState);
                addRunningQueueTime(taskInfo.estimatePrefillTime());
                logger.info("Task {} added from worker status with state: {}", requestId, taskState);
            }
        });
    }

    private void updateTaskInputLength(TaskInfo localTask, TaskInfo engineTask) {
        if (engineTask.getInputLength() > 0) {
            localTask.setInputLength(engineTask.getInputLength());
        }
    }

    private void updatePrefillRunningProgressFromEngine(TaskInfo localTask, TaskInfo engineTask) {
        localTask.setCompletedPrefillTokens(Math.max(0, engineTask.getCompletedPrefillTokens()));
        localTask.setRemainingPrefillTokens(engineTask.getRemainingPrefillTokens());
        localTask.setLastCompletedPrefillStepId(
                Math.max(0, engineTask.getLastCompletedPrefillStepId()));
    }

    private void updateCacheHitFromEngine(TaskInfo localTask, TaskInfo engineTask, String taskState,
                                          List<CacheHitFeedback> cacheHitFeedbacks) {
        if (!engineTask.isPrefixLengthValid()) {
            if (localTask.isPrefixLengthValid()) {
                long previousPrefillTime = localTask.estimatePrefillTime();
                localTask.setPrefixLength(localTask.getPredictedPrefixLength());
                localTask.setPrefixLengthValid(false);
                correctRunningQueueTime(localTask.estimatePrefillTime() - previousPrefillTime);
            }
            return;
        }

        boolean cacheHitBecameValid = !localTask.isPrefixLengthValid();
        long previousPrefillTime = localTask.estimatePrefillTime();
        localTask.setPrefixLength(engineTask.getPrefixLength());
        localTask.setPrefixLengthValid(true);
        correctRunningQueueTime(localTask.estimatePrefillTime() - previousPrefillTime);

        if (!cacheHitBecameValid) {
            return;
        }

        long predictedHitTokens = localTask.getPredictedPrefixLength();
        long actualHitTokens = localTask.getPrefixLength();
        long inputTokens = localTask.getInputLength();
        long blockSize = cacheStatus == null ? 0 : cacheStatus.getBlockSize();
        cacheHitFeedbacks.add(new CacheHitFeedback(
                "cache_hit_comparison",
                localTask.getRequestId(),
                localTask.getCacheMatchSource(),
                role,
                group,
                ip,
                port,
                engineIndex,
                taskState,
                inputTokens,
                blockSize,
                predictedHitTokens,
                localTask.isKvcmMatchAvailable(),
                localTask.getKvcmLocalMatchTokens(),
                localTask.getKvcmP2pFetchTokens(),
                localTask.getKvcmP2pTotalMatchTokens(),
                actualHitTokens,
                actualHitTokens - predictedHitTokens));
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

    public void refreshInTransitAndWaitingStats() {
        long inTransitAndWaitingTaskCount = 0;
        long inTransitAndWaitingTokens = 0;

        // Local tasks keep the routing prediction and are corrected with the actual prefix on status update.
        for (TaskInfo task : localTaskMap.values()) {
            if (!isInTransitOrWaiting(task)) {
                continue;
            }
            inTransitAndWaitingTaskCount++;
            inTransitAndWaitingTokens += uncachedTokens(task);
        }

        this.inTransitAndWaitingTaskCount = inTransitAndWaitingTaskCount;
        this.inTransitAndWaitingUncachedTokens = inTransitAndWaitingTokens;
    }

    /**
     * Sum only authoritative post-forward remaining work for active RUNNING
     * tasks. This intentionally does not alter the existing pending-task
     * aggregates or runningQueueTime estimate.
     */
    public void refreshRunningRemainingPrefillTokens() {
        long runningRemainingTokens = 0;
        for (TaskInfo task : localTaskMap.values()) {
            if (task == null || task.getTaskState() != TaskStateEnum.RUNNING) {
                continue;
            }
            long remainingPrefillTokens = task.getRemainingPrefillTokens();
            runningRemainingTokens += remainingPrefillTokens >= 0
                    ? remainingPrefillTokens
                    : uncachedTokens(task);
        }
        this.runningRemainingPrefillTokens = runningRemainingTokens;
    }

    private boolean isInTransitOrWaiting(TaskInfo task) {
        // CONFIRMED is the local state after the task appears in the engine waiting queue.
        return task != null && (task.getTaskState() == TaskStateEnum.IN_TRANSIT
                || task.getTaskState() == TaskStateEnum.CONFIRMED);
    }

    private long uncachedTokens(TaskInfo task) {
        long inputTokens = task.getInputLength();
        if (inputTokens <= 0) {
            return 0;
        }

        long cacheHitTokens = task.isPrefixLengthValid()
                ? task.getPrefixLength()
                : task.getPredictedPrefixLength();
        cacheHitTokens = Math.max(0, Math.min(inputTokens, cacheHitTokens));
        return inputTokens - cacheHitTokens;
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
     * Returns the logical worker identity used by routing and cache ownership.
     */
    public String getIpPort() {
        return getLogicalIpPort();
    }
}
