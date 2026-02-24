package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
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
 * 3. Scheduling fairness: Achieve load balancing among workers with similar performance
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

    private static final int MIN_CANDIDATE_COUNT = 1;
    private static final double CANDIDATE_PERCENTAGE = 0.3;
    private static final double TTFT_THRESHOLD_PERCENTAGE = 0.1;
    private static final double STDDEV_THRESHOLD_FACTOR = 0.5;

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                EngineHealthReporter engineHealthReporter,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, this);
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
     * @param interRequestId Internal request ID
     */
    @Override
    public void rollBack(String ipPort, String interRequestId) {

        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        Logger.debug("Prefill rollBack - ipPort: {}, interRequestId: {}", ipPort, interRequestId);

        WorkerStatus workerStatus = workerStatusMap.get(ipPort);
        if (workerStatus != null) {
            workerStatus.removeLocalTask(interRequestId);
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
        String interRequestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();

        Logger.debug("Starting shortest TTFT selection for role: {}", roleType);

        // Get available worker list
        FlexlbConfig config = balanceContext.getConfig();
        List<WorkerStatus> availableWorkers = getAvailableWorkers(roleType, group, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(availableWorkers)) {
            Logger.warn("No available workers for role: {}", roleType.getCode());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Calculate cache match results for each engine
        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        List<ScoredWorker> scoredWorkers = scoreWorkers(availableWorkers, cacheMatchResults, seqLen);

        ScoredWorker bestWorker = selectBestWorker(scoredWorkers);
        if (bestWorker == null) {
            Logger.warn("Failed to find best worker for role: {}", roleType);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        return finalizeWorkerSelection(bestWorker, balanceContext, roleType, interRequestId, seqLen);
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

        return new ArrayList<>(workerStatusMap.values()).stream()
                .filter(WorkerStatus::isAlive)                   // Check if resource is available
                .filter(resourceMeasure::isResourceAvailable)    // Check if worker has available resources
                .toList();
    }

    /**
     * Get cache match results
     *
     * @param balanceContext Load balancing context
     * @param roleType Worker role type
     * @param group Worker group
     * @return Cache match results: key: engineIpPort, value: prefixMatchLength
     */
    private Map<String /*engineIpPort*/, Integer /*prefixMatchLength*/> getCacheMatchResults(BalanceContext balanceContext,
                                                                                             RoleType roleType,
                                                                                             String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(blockCacheKeys, roleType, group);
    }

    /**
     * Calculate TTFT scores for all active workers
     *
     * @param workers Worker list
     * @param cacheMatchResults Cache match results
     * @param seqLen Sequence length
     * @return List of scored workers
     */
    private List<ScoredWorker> scoreWorkers(List<WorkerStatus> workers, Map<String, Integer> cacheMatchResults, long seqLen) {
        return workers.stream()
                .filter(WorkerStatus::isAlive)
                .map(workerStatus -> {
                    long hitCacheTokens = calculatePrefixMatchLength(workerStatus, cacheMatchResults);
                    long prefillTime = TaskInfo.estimatePrefillTimeMs(seqLen, hitCacheTokens);
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
     * @param interRequestId Internal request ID
     * @param seqLen Sequence length
     * @return Server status
     */
    private ServerStatus finalizeWorkerSelection(ScoredWorker selectedWorker,
                                                 BalanceContext balanceContext,
                                                 RoleType roleType,
                                                 String interRequestId,
                                                 long seqLen) {
        WorkerStatus workerStatus = selectedWorker.worker();

        logWorkerSelection(selectedWorker, roleType);
        reportCacheHitMetrics(roleType, workerStatus.getIp(), selectedWorker.hitCacheTokens(), seqLen);

        TaskInfo task = createTaskInfo(interRequestId, balanceContext.getRequest().getSeqLen(), selectedWorker.hitCacheTokens());
        workerStatus.putLocalTask(interRequestId, task);

        return buildServerStatus(selectedWorker, roleType, interRequestId);
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
     * @param interRequestId Internal request ID
     * @param inputLength Input length
     * @param prefixLength Prefix length
     * @return Task information
     */
    private TaskInfo createTaskInfo(String interRequestId, long inputLength, long prefixLength) {
        TaskInfo task = new TaskInfo();
        task.setInterRequestId(interRequestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        return task;
    }

    /**
     * Select best worker considering TTFT and scheduling fairness
     *
     * <p>Algorithm: 1. Sort workers by TTFT 2. Select top 30% as candidates (at least 1) 3. Among candidates with similar TTFT, prioritize recently unscheduled workers
     *
     * @param scoredWorkers List of scored workers
     * @return Best worker
     */
    private ScoredWorker selectBestWorker(List<ScoredWorker> scoredWorkers) {
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
        double threshold = calculateTTFTThreshold(candidates);

        List<ScoredWorker> similarWorkers = filterSimilarWorkers(candidates, minTTFT, threshold);

        return selectWorkerByScheduleFairness(similarWorkers, candidates);
    }

    /**
     * Sort workers by TTFT
     *
     * @param workers Worker list
     * @return Sorted worker list in ascending order
     */
    private List<ScoredWorker> sortByTTFT(List<ScoredWorker> workers) {
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
        int candidateCount = Math.max(MIN_CANDIDATE_COUNT, (int) (sortedWorkers.size() * CANDIDATE_PERCENTAGE));
        return sortedWorkers.stream().limit(candidateCount).toList();
    }

    /**
     * Calculate TTFT similarity threshold
     *
     * @param candidates Candidate worker list
     * @return TTFT threshold
     */
    private double calculateTTFTThreshold(List<ScoredWorker> candidates) {
        double avgTTFT = candidates.stream().mapToLong(ScoredWorker::ttft).average().orElse(0.0);

        double stdDev = Math.sqrt(
                candidates.stream()
                        .mapToLong(ScoredWorker::ttft)
                        .mapToDouble(v -> Math.pow(v - avgTTFT, 2))
                        .average()
                        .orElse(0.0));
        double percentageAvgTTFT = avgTTFT * TTFT_THRESHOLD_PERCENTAGE;
        double factoredStdDev = stdDev * STDDEV_THRESHOLD_FACTOR;
        Logger.debug("Calculate TTFT threshold, avgTTFT: {}, stdDev: {}, percentageAvgTTFT: {}, factoredStdDev: {}", avgTTFT, stdDev, percentageAvgTTFT, factoredStdDev);
        return Math.max(percentageAvgTTFT, factoredStdDev);
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
     * Select worker based on scheduling fairness
     * Among workers with similar TTFT, prioritize recently unscheduled workers
     *
     * @param similarWorkers List of workers with similar TTFT
     * @param fallbackCandidates Fallback candidate worker list
     * @return Finally selected worker
     */
    private ScoredWorker selectWorkerByScheduleFairness(List<ScoredWorker> similarWorkers, List<ScoredWorker> fallbackCandidates) {
        if (similarWorkers.isEmpty()) {
            return fallbackCandidates.getFirst();
        }

        return similarWorkers.stream()
                // Prioritize recently unscheduled worker
                .min(Comparator.comparingLong(ScoredWorker::lastSelectedTime))
                .orElse(fallbackCandidates.getFirst());
    }

    /**
     * Build server status response
     *
     * @param selectedWorker Selected worker
     * @param roleType Worker role type
     * @param interRequestId Internal request ID
     * @return Server status
     */
    private ServerStatus buildServerStatus(ScoredWorker selectedWorker, RoleType roleType, String interRequestId) {
        WorkerStatus workerStatus = selectedWorker.worker();
        ServerStatus result = new ServerStatus();
        try {
            result.setSuccess(true);
            result.setRole(roleType);
            result.setInterRequestId(interRequestId);
            result.setPrefillTime(selectedWorker.ttft());
            result.setGroup(workerStatus.getGroup());
            result.setServerIp(workerStatus.getIp());
            result.setHttpPort(workerStatus.getPort());
            result.setGrpcPort(CommonUtils.toGrpcPort(workerStatus.getPort()));
        } catch (Exception e) {
            Logger.error("Failed to build server status for requestId: {}", interRequestId, e);
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
