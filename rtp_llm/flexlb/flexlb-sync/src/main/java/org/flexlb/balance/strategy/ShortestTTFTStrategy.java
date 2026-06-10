package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
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
    private final EndpointRegistry endpointRegistry;

    private static final int MIN_CANDIDATE_COUNT = 1;
    private static final double CANDIDATE_PERCENTAGE = 0.3;
    private static final double TTFT_THRESHOLD_PERCENTAGE = 0.1;
    private static final double STDDEV_THRESHOLD_FACTOR = 0.5;

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                EngineHealthReporter engineHealthReporter,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory,
                                EndpointRegistry endpointRegistry) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.engineHealthReporter = engineHealthReporter;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.endpointRegistry = endpointRegistry;
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
     * @param requestId Request ID
     */
    @Override
    public void rollBack(String ipPort, long requestId) {

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
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();

        Logger.debug("Starting shortest TTFT selection for role: {}", roleType);

        // Get available worker list (alive + resource check)
        FlexlbConfig config = balanceContext.getConfig();
        Map<String, WorkerStatus> workerStatusMap = engineWorkerStatus.selectModelWorkerStatus(roleType, group);
        List<WorkerStatus> availableWorkers = getAvailableWorkers(workerStatusMap, roleType, config.getResourceMeasureIndicator(roleType));
        if (CollectionUtils.isEmpty(availableWorkers)) {
            int total = workerStatusMap.size();
            long aliveCount = workerStatusMap.values().stream().filter(WorkerStatus::isAlive).count();
            if (total == 0) {
                Logger.warn("No workers discovered for role: {}", roleType.getCode());
            } else if (aliveCount == 0) {
                Logger.warn("All {} workers for role: {} are not alive", total, roleType.getCode());
            } else {
                Logger.warn("{}/{} workers alive for role: {} but none have available resources",
                        aliveCount, total, roleType.getCode());
            }
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // Calculate cache match results for each engine
        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        // SLO filter: reject workers whose estimated TTFT exceeds SLO
        long sloMs = config.resolveSloMs(seqLen);
        PrefillTimePredictor predictor = createPredictor(config);
        availableWorkers = filterBySlo(availableWorkers, cacheMatchResults, seqLen, sloMs, predictor);
        if (CollectionUtils.isEmpty(availableWorkers)) {
            Logger.warn("No workers within SLO for role: {}, sloMs: {}", roleType.getCode(), sloMs);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        List<ScoredWorker> scoredWorkers = scoreWorkers(availableWorkers, cacheMatchResults, seqLen);

        ScoredWorker bestWorker = selectBestWorker(scoredWorkers);
        if (bestWorker == null) {
            Logger.warn("Failed to find best worker for role: {}", roleType);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        return finalizeWorkerSelection(bestWorker, balanceContext, roleType, requestId, seqLen, predictor);
    }

    /**
     * Get available worker list
     *
     * @param roleType Worker role type
     * @param group Worker group
     * @param indicator ResourceMeasureIndicatorEnum
     * @return Available worker list
     */
    private List<WorkerStatus> getAvailableWorkers(Map<String, WorkerStatus> workerStatusMap,
                                                    RoleType roleType,
                                                    ResourceMeasureIndicatorEnum indicator) {
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

    private List<WorkerStatus> filterBySlo(List<WorkerStatus> workers, Map<String, Integer> cacheMatchResults,
                                            long seqLen, long sloMs, PrefillTimePredictor predictor) {
        if (sloMs <= 0) {
            return workers;
        }
        return workers.stream()
                .filter(w -> {
                    long hitCacheTokens = calculatePrefixMatchLength(w, cacheMatchResults);
                    long prefillMs = predictor.estimateMs(seqLen, hitCacheTokens);
                    long queueMs = w.getPredictedQueueTimeMs().get();
                    PrefillEndpoint ep = endpointRegistry.getPrefill(w.getIpPort());
                    long inflightWaitMs = ep != null ? ep.getEstimatedWaitingTimeMs() : 0;
                    return queueMs + inflightWaitMs + prefillMs <= sloMs;
                })
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
                    PrefillEndpoint ep = endpointRegistry.getPrefill(workerStatus.getIpPort());
                    long inflightWaitMs = ep != null ? ep.getEstimatedWaitingTimeMs() : 0;
                    long newTTFT = prefillTime + queueTime + inflightWaitMs;
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
                                                 long requestId,
                                                 long seqLen,
                                                 PrefillTimePredictor predictor) {
        WorkerStatus workerStatus = selectedWorker.worker();

        logWorkerSelection(selectedWorker, roleType);
        reportCacheHitMetrics(roleType, workerStatus.getIp(), selectedWorker.hitCacheTokens(), seqLen);

        TaskInfo task = createTaskInfo(requestId, balanceContext.getRequest().getSeqLen(), selectedWorker.hitCacheTokens());
        task.setPredictedMs(predictor.estimateMs(seqLen, selectedWorker.hitCacheTokens()));
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
    private TaskInfo createTaskInfo(long requestId, long inputLength, long prefixLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
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
        double threshold = calculateTTFTThreshold(candidates, minTTFT);

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
    private double calculateTTFTThreshold(List<ScoredWorker> candidates, long minTTFT) {
        double avgTTFT = candidates.stream().mapToLong(ScoredWorker::ttft).average().orElse(0.0);

        double stdDev = Math.sqrt(
                candidates.stream()
                        .mapToLong(ScoredWorker::ttft)
                        .mapToDouble(v -> Math.pow(v - avgTTFT, 2))
                        .average()
                        .orElse(0.0));
        double percentageMinTTFT = minTTFT * TTFT_THRESHOLD_PERCENTAGE;
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
     * Select worker based on scheduling fairness.
     * Among workers with similar TTFT, prefer the least recently scheduled one.
     * CAS on lastSelectedTime ensures concurrent requests are spread across different workers
     * rather than all landing on the same one.
     *
     * @param similarWorkers workers with similar TTFT
     * @param fallbackCandidates fallback candidate list
     * @return selected worker
     */
    private ScoredWorker selectWorkerByScheduleFairness(List<ScoredWorker> similarWorkers, List<ScoredWorker> fallbackCandidates) {
        if (similarWorkers.isEmpty()) {
            return fallbackCandidates.getFirst();
        }

        // Sort ascending by lastSelectedTime so the least recently used worker is tried first
        List<ScoredWorker> sorted = similarWorkers.stream()
                .sorted(Comparator.comparingLong(ScoredWorker::lastSelectedTime))
                .toList();

        long now = System.nanoTime() / 1000;
        for (ScoredWorker candidate : sorted) {
            long expected = candidate.lastSelectedTime();
            // CAS: claim this worker only if lastSelectedTime hasn't changed since we read it.
            // A failed CAS means another concurrent request already claimed this worker.
            if (candidate.worker().getLastSelectedTime().compareAndSet(expected, now)) {
                return candidate;
            }
            // Another request claimed this worker; try the next candidate
        }

        // All candidates were claimed concurrently; fall back to the first candidate
        return fallbackCandidates.getFirst();
    }

    /**
     * Build server status response
     *
     * @param selectedWorker Selected worker
     * @param roleType Worker role type
     * @param requestId Request ID
     * @return Server status
     */
    private ServerStatus buildServerStatus(ScoredWorker selectedWorker, RoleType roleType, long requestId) {
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
            result.setDpRank(workerStatus.getDpRank());
        } catch (Exception e) {
            Logger.error("Failed to build server status for requestId: {}", requestId, e);
            result.setCode(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
            result.setMessage(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorMsg());
            result.setSuccess(false);
        }
        return result;
    }

    private PrefillTimePredictor createPredictor(FlexlbConfig config) {
        return new PrefillTimePredictor(
                config.getCostAlpha0(), config.getCostAlpha1(), config.getCostAlpha2(),
                config.getCostAlpha3(), config.getCostAlpha4(), config.getCostAlpha5());
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
