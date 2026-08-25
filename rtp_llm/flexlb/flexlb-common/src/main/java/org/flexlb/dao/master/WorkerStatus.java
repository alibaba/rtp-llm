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

@Data
@Slf4j
public class WorkerStatus {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger("syncLogger");
    public final transient ReentrantLock lock = new ReentrantLock();
    private RoleType role;
    private String group;
    private String deploymentName;
    private String ip;
    private int port;
    private int grpcPort;
    private String site;
    /**
     * Compatibility-only mirror of WorkerStatusPB.available_concurrency.
     * LocalRpcServer currently leaves that protobuf field unset, so this value
     * must not participate in routing, admission control, or batch sizing.
     */
    private Long availableConcurrency;
    private volatile boolean alive;
    private AtomicLong availableKvCacheTokens = new AtomicLong();
    private AtomicLong usedKvCacheTokens = new AtomicLong();
    private AtomicLong totalKvCacheTokens = new AtomicLong();
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
    private long dpRank;
    private int blockHashLookaheadTokens;
    /** Model-level maximum sequence length reported by the Engine. */
    private long maxSeqLen;
    /** Strict aggregate context-token limit for an Engine batch/group. */
    private long maxBatchTokensSize;
    private int cacheMatchRollbackBlocks;
    private KvCacheGroupMode kvCacheGroupMode = KvCacheGroupMode.UNSPECIFIED;

    private AtomicLong statusLastUpdateTime = new AtomicLong(-1);
    private AtomicLong statusUpdateIntervalUs = new AtomicLong(0);
    private AtomicLong cacheLastUpdateTime = new AtomicLong(-1);
    private AtomicLong lastSelectedTime = new AtomicLong(-1);
    private AtomicBoolean resourceAvailable = new AtomicBoolean(true);
    private AtomicBoolean statusCheckInProgress = new AtomicBoolean(false);
    private AtomicBoolean cacheCheckInProgress = new AtomicBoolean(false);
    private AtomicLong statusVersion = new AtomicLong(-1L);
    private AtomicLong consecutiveFailures = new AtomicLong(0);

    /**
     * Absorb all dynamic engine fields from a gRPC status response.
     * Topology labels ({@code site}, {@code group}) are NOT set here —
     * they are managed externally by the sync runner.
     */
    public void updateFromResponse(WorkerStatusResponse resp) {
        if (resp == null) {
            return;
        }
        this.role = resp.getRole();
        this.alive = resp.isAlive();
        this.availableConcurrency = resp.getAvailableConcurrency();
        this.stepLatencyMs = resp.getStepLatencyMs();
        this.iterateCount = resp.getIterateCount();
        this.dpSize = resp.getDpSize();
        this.tpSize = resp.getTpSize();
        this.dpRank = resp.getDpRank();
        this.blockHashLookaheadTokens = resp.getBlockHashLookaheadTokens();
        this.cacheMatchRollbackBlocks = resp.getCacheMatchRollbackBlocks();
        this.kvCacheGroupMode = resp.getKvCacheGroupMode();
        this.maxSeqLen = resp.getMaxSeqLen();
        this.maxBatchTokensSize = resp.getMaxBatchTokensSize();
        this.availableKvCacheTokens.set(resp.getAvailableKvCacheTokens());
        this.totalKvCacheTokens.set(resp.getTotalKvCacheTokens());
        this.usedKvCacheTokens.set(Math.max(0,
                resp.getTotalKvCacheTokens() - resp.getAvailableKvCacheTokens()));
        // GetWorkerStatus response does not include cache status; preserve the one
        // set by GrpcCacheStatusCheckRunner to avoid nullifying it on every status sync.
        if (resp.getCacheStatus() != null) {
            this.cacheStatus = resp.getCacheStatus();
        }
        this.runningTaskList = resp.getRunningTaskInfo();
        this.statusVersion.set(resp.getStatusVersion());
        // NOTE: latestFinishedTaskVersion is NOT set here. It is advanced only after
        // calibrate has processed finished tasks, in GrpcWorkerStatusRunner.handleStatusResponse().
        // Setting it here would advance the version before calibrate runs, causing the engine
        // to filter out unprocessed finished tasks on the next poll — leaking inflight entries.
        updateStatusHeartbeatTime();
    }

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

                if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
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

    private void updateCacheHitFromEngine(
            TaskInfo localTask,
            TaskInfo engineTask,
            String taskState,
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
        long blockSize = cacheStatus == null ? 0 : cacheStatus.getBlockSize();
        cacheHitFeedbacks.add(new CacheHitFeedback(
                "cache_hit_comparison",
                String.valueOf(localTask.getRequestId()),
                localTask.getCacheMatchSource(),
                role == null ? "" : role.name(),
                group,
                ip,
                port,
                taskState,
                localTask.getInputLength(),
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
                || (role != RoleType.PREFILL && role != RoleType.PDFUSION)) {
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
        if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
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

        updateStatusHeartbeatTime();
    }

    /**
     * Record a successful status heartbeat without replacing versioned task data.
     */
    public void refreshStatusHeartbeat(boolean alive) {
        this.alive = alive;
        updateStatusHeartbeatTime();
    }

    private void updateStatusHeartbeatTime() {
        long nowUs = System.nanoTime() / 1000;
        long prev = this.statusLastUpdateTime.get();
        if (prev > 0) {
            this.statusUpdateIntervalUs.set(nowUs - prev);
        }
        this.statusLastUpdateTime.set(nowUs);
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
     * Decrement an aggregate queue estimate without allowing it to become negative.
     */
    public static void safeDecrementQueueTime(
            AtomicLong runningQueueTime, long timeToReduce) {
        if (timeToReduce <= 0) {
            return;
        }
        runningQueueTime.accumulateAndGet(
                timeToReduce,
                (current, reduction) -> Math.max(0, current - reduction));
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
