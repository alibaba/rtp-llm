package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.CacheMatchResult;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.ShortestTtftDecision;
import org.flexlb.dao.pv.ShortestTtftDecision.QueueTask;
import org.flexlb.dao.pv.ShortestTtftDecision.WorkerDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Load balancing strategy based on shortest Time-To-First-Token (TTFT)
 *
 * <p>This strategy selects the optimal worker by considering the following factors:
 * 1. KV-Cache hit rate: Prioritize workers with higher cache hit rates
 * 2. Queue time: Consider the current task queue status of workers
 * 3. Cache preference: Among workers with similar TTFT, prefer a meaningful cache lead
 *
 * @author saichen.sm
 * @since 2025/3/10
 */
@Component("shortestTTFTStrategy")
public class ShortestTTFTStrategy implements LoadBalancer {

    private final EngineWorkerStatus engineWorkerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final LoadBalanceStrategyEnum strategy;

    private static final int SMALL_CLUSTER_SIZE = 3;
    private static final int MIN_CANDIDATE_COUNT = 2;
    private static final double CANDIDATE_PERCENTAGE = 0.3;
    private static final double STDDEV_THRESHOLD_FACTOR = 0.5;

    @Autowired
    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                EngineHealthReporter engineHealthReporter,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory) {
        this(
                engineWorkerStatus,
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                LoadBalanceStrategyEnum.SHORTEST_TTFT);
    }

    protected ShortestTTFTStrategy(
            EngineWorkerStatus engineWorkerStatus,
            EngineHealthReporter engineHealthReporter,
            CacheAwareService cacheAwareService,
            ResourceMeasureFactory resourceMeasureFactory,
            LoadBalanceStrategyEnum strategy) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.strategy = strategy;
        LoadBalanceStrategyFactory.register(strategy, this);
    }

    /**
     * Select optimal worker to execute task
     *
     * @param balanceContext Load balancing context
     * @param roleType Worker role type
     * @param group Worker group
     * @return Selected server status
     */
    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("Failed to select worker", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    /**
     * Release local cached tasks on the specified worker
     *
     * @param ipPort Worker IP address
     * @param requestId Request ID
     */
    @Override
    public void rollBack(String ipPort, String requestId) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        Logger.debug("Prefill rollBack - ipPort: {}, requestId: {}", ipPort, requestId);

        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(requestId);
        }
    }

    /**
     * Core logic for worker selection
     *
     * @param balanceContext Load balancing context
     * @param roleType Worker role type
     * @param group Worker group
     * @return Selected server status
     */
    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        String requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();

        Logger.debug("Starting {} selection for role: {}", strategy.getName(), roleType);

        // Get available worker list
        FlexlbConfig config = balanceContext.getConfig();
        List<WorkerStatus> availableWorkers = getAvailableWorkers(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(availableWorkers)) {
            Logger.warn("No available workers for role: {}", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Calculate cache match results for each engine
        CacheMatchResult cacheMatchResult = cacheAwareService.findMatchingEngines(
                requestId, balanceContext.getRequest().getBlockCacheKeys(), roleType, group);

        List<ScoredWorker> scoredWorkers = scoreWorkers(
                availableWorkers,
                cacheMatchResult.matches(),
                seqLen,
                config.getPrefillCacheHitDiscount());

        ScoredWorker bestWorker = selectBestWorker(
                scoredWorkers, balanceContext, roleType, group, seqLen, config);
        if (bestWorker == null) {
            Logger.warn("Failed to find best worker for role: {}", roleType);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        balanceContext.recordCacheMatch(
                cacheMatchResult.source().name(),
                cacheMatchResult.queryTimeUs(),
                roleType,
                bestWorker.worker().getIp(),
                bestWorker.hitCacheTokens());

        return finalizeWorkerSelection(
                bestWorker,
                balanceContext,
                roleType,
                requestId,
                seqLen,
                config.getPrefillCacheHitDiscount());
    }

    /**
     * Get available worker list
     *
     * @param roleType Worker role type
     * @param group Worker group
     * @param indicator ResourceMeasureIndicatorEnum
     * @return Available worker list
     */
    private List<WorkerStatus> getAvailableWorkers(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerStatusMap)) {
            return new ArrayList<>();
        }

        ResourceMeasure resourceMeasure = resourceMeasureFactory.getMeasure(indicator);
        if (resourceMeasure == null) {
            Logger.warn("No ResourceMeasure registered for indicator: {}", indicator);
            return new ArrayList<>();
        }

        return new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)
                .filter(resourceMeasure::isResourceAvailable)
                .toList();
    }

    /**
     * Calculate TTFT scores for all active workers
     *
     * @param workers Worker list
     * @param cacheMatchResults Cache match results
     * @param seqLen Sequence length
     * @return List of scored workers
     */
    private List<ScoredWorker> scoreWorkers(
            List<WorkerStatus> workers,
            Map<String, Integer> cacheMatchResults,
            long seqLen,
            double cacheHitDiscount) {
        return workers.stream()
                .filter(WorkerStatus::isAlive)
                .map(workerStatus -> {
                    long hitCacheTokens = calculatePrefixMatchLength(workerStatus, cacheMatchResults);
                    long prefillTime = TaskInfo.estimatePrefillTimeMs(
                            seqLen, hitCacheTokens, cacheHitDiscount);
                    long queueTime = workerStatus.getRunningQueueTime().get();
                    long newTTFT = prefillTime + queueTime;
                    long lastSelectedTime = workerStatus.getLastSelectedTime().get();
                    Logger.debug("Calculate TTFT for worker - ip: {}, port: {}, hitCacheTokens: {}, prefillTime: {}, queueTime: {}, newTTFT: {}",
                            workerStatus.getIp(),
                            workerStatus.getPort(),
                            hitCacheTokens,
                            prefillTime,
                            queueTime,
                            newTTFT);
                    return new ScoredWorker(workerStatus, newTTFT, hitCacheTokens, lastSelectedTime);
                })
                .collect(Collectors.toList());
    }

    /**
     * Finalize worker selection and update status
     *
     * @param selectedWorker Selected worker
     * @param balanceContext Load balancing context
     * @param roleType Worker role type
     * @param requestId Request ID
     * @param seqLen Sequence length
     * @return Server status
     */
    private ServerStatus finalizeWorkerSelection(ScoredWorker selectedWorker,
                                                 BalanceContext balanceContext,
                                                 RoleType roleType,
                                                 String requestId,
                                                 long seqLen,
                                                 double cacheHitDiscount) {
        WorkerStatus workerStatus = selectedWorker.worker();

        logWorkerSelection(selectedWorker, roleType);
        reportCacheHitMetrics(roleType, workerStatus.getIp(), selectedWorker.hitCacheTokens(), seqLen);

        TaskInfo task = createTaskInfo(
                requestId,
                balanceContext.getRequest().getSeqLen(),
                selectedWorker.hitCacheTokens(),
                balanceContext.getCacheMatchSource(),
                cacheHitDiscount);
        workerStatus.putLocalTask(requestId, task);

        return buildServerStatus(selectedWorker, roleType, requestId);
    }

    /**
     * Log worker selection
     *
     * @param selectedWorker Selected worker
     * @param roleType Worker role type
     */
    private void logWorkerSelection(ScoredWorker selectedWorker, RoleType roleType) {
        WorkerStatus workerStatus = selectedWorker.worker();
        Logger.debug("Selected {} worker - ip: {}, port: {}, hitCacheTokens: {}, ttft: {}",
                roleType,
                workerStatus.getIp(),
                workerStatus.getPort(),
                selectedWorker.hitCacheTokens(),
                selectedWorker.ttft());
    }

    /**
     * Report cache hit metrics
     *
     * @param roleType Worker role type
     * @param ip Worker IP address
     * @param hitCacheTokens Number of cached tokens hit
     * @param seqLen Sequence length
     */
    private void reportCacheHitMetrics(RoleType roleType, String ip, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ip, hitCacheTokens, hitRate);
    }

    /**
     * Create task information
     *
     * @param requestId Request ID
     * @param inputLength Input length
     * @param prefixLength Prefix length
     * @return Task information
     */
    private TaskInfo createTaskInfo(
            String requestId,
            long inputLength,
            long prefixLength,
            String cacheMatchSource,
            double cacheHitDiscount) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setPredictedPrefixLength(prefixLength);
        task.setCacheMatchSource(cacheMatchSource);
        task.setCacheHitDiscount(cacheHitDiscount);
        return task;
    }

    /**
     * Select best worker considering TTFT and cache preference
     *
     * <p>Algorithm: 1. Sort workers by TTFT. 2. Consider all workers in a small cluster,
     * otherwise the top 30%. 3. Among workers with similar TTFT, prefer the worker whose
     * cache lead over the shortest-TTFT worker reaches the configured block threshold.
     *
     * @param scoredWorkers List of scored workers
     * @return Best worker
     */
    protected ScoredWorker selectBestWorker(
            List<ScoredWorker> scoredWorkers,
            BalanceContext balanceContext,
            RoleType roleType,
            String group,
            long seqLen,
            FlexlbConfig config) {
        if (scoredWorkers.isEmpty()) {
            return null;
        }

        List<ScoredWorker> sortedWorkers = sortByTTFT(scoredWorkers);
        List<ScoredWorker> candidates = selectTopCandidates(sortedWorkers);
        Logger.debug("Select best worker, sortedWorkers size: {}, candidates size: {}", sortedWorkers.size(), candidates.size());

        if (candidates.isEmpty()) {
            return null;
        }

        long minTTFT = candidates.getFirst().ttft();
        double threshold = calculateTTFTThreshold(
                candidates,
                minTTFT,
                config.getShortestTtftSimilarityThresholdRatio());

        List<ScoredWorker> similarWorkers = filterSimilarWorkers(candidates, minTTFT, threshold);

        ScoredWorker selectedWorker = selectWorkerByCachePreference(
                similarWorkers,
                candidates,
                config.getPrefillCachePreferenceMinBlockGap());
        if (Logger.isDebugEnabled()) {
            balanceContext.recordShortestTtftDecision(buildDecisionSnapshot(
                    selectedWorker,
                    sortedWorkers,
                    candidates,
                    similarWorkers,
                    minTTFT,
                    threshold,
                    roleType,
                    group,
                    seqLen,
                    config.getPrefillCacheHitDiscount()));
        }
        return selectedWorker;
    }

    private ShortestTtftDecision buildDecisionSnapshot(
            ScoredWorker selectedWorker,
            List<ScoredWorker> sortedWorkers,
            List<ScoredWorker> topCandidates,
            List<ScoredWorker> similarWorkers,
            long minimumTtft,
            double similarTtftThreshold,
            RoleType roleType,
            String group,
            long seqLen,
            double cacheHitDiscount) {
        List<WorkerDecision> workers = sortedWorkers.stream()
                .map(scoredWorker -> buildWorkerDecision(
                        scoredWorker,
                        selectedWorker,
                        topCandidates,
                        similarWorkers,
                        seqLen,
                        cacheHitDiscount))
                .toList();
        return new ShortestTtftDecision(
                roleType,
                group,
                seqLen,
                minimumTtft,
                similarTtftThreshold,
                workers);
    }

    private WorkerDecision buildWorkerDecision(
            ScoredWorker scoredWorker,
            ScoredWorker selectedWorker,
            List<ScoredWorker> topCandidates,
            List<ScoredWorker> similarWorkers,
            long seqLen,
            double cacheHitDiscount) {
        WorkerStatus worker = scoredWorker.worker();
        long requestPrefillTime = TaskInfo.estimatePrefillTimeMs(
                seqLen, scoredWorker.hitCacheTokens(), cacheHitDiscount);
        List<QueueTask> trackedTasks = snapshotTrackedTasks(worker.getLocalTaskMap());
        List<QueueTask> waitingTasks = snapshotWorkerTasks(
                worker.getWaitingTaskList(), "waiting", cacheHitDiscount);
        List<QueueTask> runningTasks = snapshotWorkerTasks(
                worker.getRunningTaskList(), "running", cacheHitDiscount);
        long blockSize = worker.getCacheStatus() == null ? 0 : worker.getCacheStatus().getBlockSize();

        return new WorkerDecision(
                worker.getIp(),
                worker.getPort(),
                topCandidates.contains(scoredWorker),
                similarWorkers.contains(scoredWorker),
                selectedWorker.equals(scoredWorker),
                blockSize,
                scoredWorker.hitCacheTokens(),
                requestPrefillTime,
                scoredWorker.ttft() - requestPrefillTime,
                scoredWorker.ttft(),
                scoredWorker.lastSelectedTime(),
                trackedTasks.size(),
                waitingTasks.size(),
                runningTasks.size(),
                trackedTasks,
                waitingTasks,
                runningTasks);
    }

    private List<QueueTask> snapshotTrackedTasks(Map<String, TaskInfo> tasks) {
        if (MapUtils.isEmpty(tasks)) {
            return List.of();
        }
        return tasks.entrySet().stream()
                .filter(entry -> entry.getValue() != null)
                .map(entry -> toQueueTask(
                        entry.getKey(), entry.getValue(), entry.getValue().getTaskState().getValue()))
                .toList();
    }

    private List<QueueTask> snapshotWorkerTasks(
            Map<String, TaskInfo> tasks, String state, double cacheHitDiscount) {
        if (MapUtils.isEmpty(tasks)) {
            return List.of();
        }
        return tasks.entrySet().stream()
                .filter(entry -> entry.getValue() != null)
                .map(entry -> toQueueTask(
                        entry.getKey(), entry.getValue(), state, cacheHitDiscount))
                .toList();
    }

    private QueueTask toQueueTask(String requestId, TaskInfo task, String state) {
        return toQueueTask(requestId, task, state, task.getCacheHitDiscount());
    }

    private QueueTask toQueueTask(
            String requestId, TaskInfo task, String state, double cacheHitDiscount) {
        return new QueueTask(
                requestId,
                state,
                task.getInputLength(),
                task.getPrefixLength(),
                TaskInfo.estimatePrefillTimeMs(
                        task.getInputLength(), task.getPrefixLength(), cacheHitDiscount),
                task.getWaitingTime());
    }

    /**
     * Sort workers by TTFT
     *
     * @param workers Worker list
     * @return Sorted worker list in ascending order
     */
    protected List<ScoredWorker> sortByTTFT(List<ScoredWorker> workers) {
        // Two-level sorting
        // 1. Primary sort: by TTFT (Time-To-First-Token) in ascending order
        // 2. Secondary sort: when TTFT is equal, by lastSelectedTime in ascending order
        return workers.stream()
                .sorted(Comparator.comparingLong(ScoredWorker::ttft)
                        .thenComparingLong(ScoredWorker::lastSelectedTime))
                .toList();
    }

    /**
     * Select top N candidate workers
     *
     * @param sortedWorkers Sorted worker list
     * @return Candidate worker list
     */
    private List<ScoredWorker> selectTopCandidates(List<ScoredWorker> sortedWorkers) {
        int workerCount = sortedWorkers.size();
        int candidateCount = workerCount <= SMALL_CLUSTER_SIZE
                ? workerCount
                : Math.max(
                        MIN_CANDIDATE_COUNT,
                        (int) Math.ceil(workerCount * CANDIDATE_PERCENTAGE));
        return sortedWorkers.stream().limit(candidateCount).toList();
    }

    /**
     * Calculate TTFT similarity threshold
     *
     * @param candidates Candidate worker list
     * @return TTFT threshold
     */
    private double calculateTTFTThreshold(
            List<ScoredWorker> candidates,
            long minTTFT,
            double similarityThresholdRatio) {
        double avgTTFT = candidates.stream().mapToLong(ScoredWorker::ttft).average().orElse(0.0);

        double stdDev = Math.sqrt(
                candidates.stream()
                        .mapToLong(ScoredWorker::ttft)
                        .mapToDouble(v -> Math.pow(v - avgTTFT, 2))
                        .average()
                        .orElse(0.0));
        double percentageMinTTFT = minTTFT * similarityThresholdRatio;
        double factoredStdDev = stdDev * STDDEV_THRESHOLD_FACTOR;
        Logger.debug("Calculate TTFT threshold, minTTFT: {}, avgTTFT: {}, stdDev: {}, percentageMinTTFT: {}, factoredStdDev: {}",
                minTTFT, avgTTFT, stdDev, percentageMinTTFT, factoredStdDev);
        return Math.max(percentageMinTTFT, factoredStdDev);
    }

    /**
     * Filter workers with similar TTFT
     *
     * @param candidates Candidate worker list
     * @param minTTFT Minimum TTFT value
     * @param threshold Threshold
     * @return List of workers with similar TTFT
     */
    private List<ScoredWorker> filterSimilarWorkers(List<ScoredWorker> candidates, long minTTFT, double threshold) {
        List<ScoredWorker> scoredWorkers = candidates.stream()
                .filter(worker -> Math.abs(worker.ttft() - minTTFT) <= threshold)
                .toList();
        Logger.debug("Filter similar workers, minTTFT: {}, threshold: {}, candidates size: {}", minTTFT, threshold, scoredWorkers.size());
        return scoredWorkers;
    }

    /**
     * Among workers with similar TTFT, prefer the cache leader only when its lead over the
     * shortest-TTFT worker reaches the configured number of blocks. Otherwise preserve the
     * shortest-TTFT choice, which selects the shortest queue when cache hits are equal.
     *
     * @param similarWorkers workers whose TTFT is close to the minimum
     * @param fallbackCandidates candidates sorted by TTFT
     * @param minimumCacheLeadBlocks minimum cache lead required for cache preference
     * @return selected worker
     */
    private ScoredWorker selectWorkerByCachePreference(
            List<ScoredWorker> similarWorkers,
            List<ScoredWorker> fallbackCandidates,
            int minimumCacheLeadBlocks) {
        ScoredWorker shortestTtftWorker = fallbackCandidates.getFirst();
        if (similarWorkers.isEmpty()) {
            return shortestTtftWorker;
        }

        ScoredWorker cacheLeader = similarWorkers.stream()
                .min(Comparator.comparingLong(ScoredWorker::hitCacheTokens)
                        .reversed()
                        .thenComparingLong(ScoredWorker::ttft))
                .orElse(shortestTtftWorker);
        long blockSize = cacheLeader.worker().getCacheStatus() == null
                ? 0
                : cacheLeader.worker().getCacheStatus().getBlockSize();
        long cacheLeadTokens = cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens();
        long minimumCacheLeadTokens = blockSize * Math.max(0, minimumCacheLeadBlocks);

        Logger.debug(
                "Cache preference - shortest: {}, cacheLeader: {}, cacheLeadTokens: {}, minimumCacheLeadTokens: {}, shortestTtft: {}, cacheLeaderTtft: {}",
                shortestTtftWorker.worker().getIpPort(),
                cacheLeader.worker().getIpPort(),
                cacheLeadTokens,
                minimumCacheLeadTokens,
                shortestTtftWorker.ttft(),
                cacheLeader.ttft());
        ScoredWorker preferredWorker = blockSize > 0 && cacheLeadTokens >= minimumCacheLeadTokens
                ? cacheLeader
                : shortestTtftWorker;
        return claimPreferredWorker(preferredWorker, similarWorkers, shortestTtftWorker);
    }

    /**
     * Prevent concurrent scheduler threads that observed the same queue snapshot from all
     * selecting one worker. The algorithm's preferred worker is tried first; only a concurrent
     * claim causes another eligible worker to be considered.
     */
    protected ScoredWorker claimPreferredWorker(
            ScoredWorker preferredWorker,
            List<ScoredWorker> candidateWorkers,
            ScoredWorker fallbackWorker) {
        List<ScoredWorker> claimOrder = new ArrayList<>(candidateWorkers.size());
        claimOrder.add(preferredWorker);
        candidateWorkers.stream()
                .filter(worker -> !worker.equals(preferredWorker))
                .sorted(Comparator.comparingLong(ScoredWorker::ttft))
                .forEach(claimOrder::add);

        long now = System.nanoTime() / 1000;
        for (ScoredWorker candidate : claimOrder) {
            if (candidate.worker().getLastSelectedTime().compareAndSet(
                    candidate.lastSelectedTime(), now)) {
                return candidate;
            }
        }
        return fallbackWorker;
    }

    /**
     * Build server status response
     *
     * @param selectedWorker Selected worker
     * @param roleType Worker role type
     * @param requestId Request ID
     * @return Server status
     */
    private ServerStatus buildServerStatus(ScoredWorker selectedWorker, RoleType roleType, String requestId) {
        WorkerStatus workerStatus = selectedWorker.worker();
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setRole(roleType);
            result.setRequestId(requestId);
            result.setPrefillTime(selectedWorker.ttft());
            result.setGroup(workerStatus.getGroup());
            result.setServerIp(workerStatus.getIp());
            result.setHttpPort(workerStatus.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(workerStatus.getPort()));
        } catch (Exception e) {
            Logger.error("Failed to build server status for requestId: {}", requestId, e);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
            result.setSuccess(false);
        }
        return result;
    }

    /**
     * Calculate prefix match length (number of cached tokens hit)
     *
     * @param workerStatus Worker status
     * @param cacheMatchResults Cache match results
     * @return Number of tokens hit
     */
    private long calculatePrefixMatchLength(WorkerStatus workerStatus, Map<String, Integer> cacheMatchResults) {
        if (workerStatus.getCacheStatus() == null || cacheMatchResults == null) {
            return 0L;
        }

        Integer prefixMatchLength = cacheMatchResults.get(workerStatus.getIpPort());
        if (prefixMatchLength == null) {
            return 0L;
        }

        long blockSize = workerStatus.getCacheStatus().getBlockSize();
        return blockSize * prefixMatchLength;
    }

}
