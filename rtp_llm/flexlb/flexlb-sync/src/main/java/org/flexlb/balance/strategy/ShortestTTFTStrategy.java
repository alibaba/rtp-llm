package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.ShortestTtftDecision;
import org.flexlb.dao.pv.ShortestTtftDecision.CacheAffinityDecision;
import org.flexlb.dao.pv.ShortestTtftDecision.WorkerDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.StrategySelectionReason;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
    private static final int DECISION_SNAPSHOT_WORKER_LIMIT = 5;

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
     * @param ipPort logical worker identity in {@code ip:port@engineIndex} format; the index
     *               identifies one independently routable engine behind the physical frontend
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
                cacheMatchQuery(
                        balanceContext,
                        balanceContext.getRequest().getBlockSize(),
                        roleType,
                        group));

        List<ScoredWorker> scoredWorkers = scoreWorkers(
                availableWorkers,
                cacheMatchResult,
                seqLen,
                config.getP2pHitDiscount());

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
                cacheMatchResult);
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

        Map<String, WorkerStatus> workerStatusMap =
                engineWorkerStatus.selectRoutableModelWorkerStatus(roleType, group);
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
     * @param cacheMatchResult Cache match result
     * @param seqLen Sequence length
     * @return List of scored workers
     */
    private List<ScoredWorker> scoreWorkers(List<WorkerStatus> workers,
                                            CacheMatchResult cacheMatchResult,
                                            long seqLen,
                                            double p2pHitDiscount) {
        return workers.stream()
                .filter(WorkerStatus::isAlive)
                .map(workerStatus -> {
                    HostCacheMatch hostCacheMatch = cacheMatchResult.hostMatch(workerStatus.getLogicalIpPort());
                    long hitCacheTokens = calculatePrefixMatchLength(
                            workerStatus, cacheMatchResult, p2pHitDiscount, seqLen);
                    long prefillTime = TaskInfo.estimatePrefillTimeMs(seqLen, hitCacheTokens);
                    long queueTime = workerStatus.getRunningQueueTime().get();
                    long newTTFT = prefillTime + queueTime;
                    long lastSelectedTime = workerStatus.getLastSelectedTime().get();
                    long localMatchTokens = matchTokens(
                            hostCacheMatch == null ? 0 : hostCacheMatch.localMatchBlocks(),
                            cacheMatchResult.blockSize(),
                            seqLen);
                    long p2pFetchTokens = matchTokens(
                            hostCacheMatch == null ? 0 : hostCacheMatch.p2pFetchBlocks(),
                            cacheMatchResult.blockSize(),
                            seqLen);
                    long p2pTotalMatchTokens = matchTokens(
                            hostCacheMatch == null ? 0 : hostCacheMatch.p2pTotalMatchBlocks(),
                            cacheMatchResult.blockSize(),
                            seqLen);
                    Logger.debug("Calculate TTFT for worker - ip: {}, port: {}, hitCacheTokens: {}, prefillTime: {}, queueTime: {}, newTTFT: {}",
                            workerStatus.getIp(),
                            workerStatus.getPort(),
                            hitCacheTokens,
                            prefillTime,
                            queueTime,
                            newTTFT);
                    return new ScoredWorker(
                            workerStatus,
                            newTTFT,
                            hitCacheTokens,
                            lastSelectedTime,
                            localMatchTokens,
                            p2pFetchTokens,
                            p2pTotalMatchTokens);
                })
                .collect(Collectors.toList());
    }

    private long matchTokens(long matchBlocks, long blockSize, long inputTokens) {
        return CacheMatchResult.matchedTokens(matchBlocks, blockSize, inputTokens);
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
                                                 CacheMatchResult cacheMatchResult) {
        WorkerStatus workerStatus = selectedWorker.worker();

        logWorkerSelection(selectedWorker, roleType);
        reportCacheHitMetrics(roleType, workerStatus.getIp(), selectedWorker.hitCacheTokens(), seqLen);

        TaskInfo task = createTaskInfo(
                requestId,
                balanceContext.getRequest().getSeqLen(),
                selectedWorker.hitCacheTokens(),
                balanceContext.getCacheMatchSource());
        recordKvcmMatch(
                task,
                cacheMatchResult,
                cacheMatchResult.hostMatch(workerStatus.getLogicalIpPort()),
                seqLen);
        engineHealthReporter.reportKvcmSelectedMatch(
                roleType,
                workerStatus.getIp(),
                task.getKvcmLocalMatchTokens(),
                task.getKvcmP2pFetchTokens(),
                task.getKvcmP2pTotalMatchTokens(),
                task.isKvcmMatchAvailable());
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
            String cacheMatchSource) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setPredictedPrefixLength(prefixLength);
        task.setCacheMatchSource(cacheMatchSource);
        return task;
    }

    /**
     * Select best worker considering TTFT and cache preference
     *
     * <p>Algorithm: 1. Sort workers by TTFT. 2. Consider all workers in a small cluster,
     * otherwise the top 30%. 3. Among workers with similar TTFT, prefer cache.
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
        List<ScoredWorker> eligibleWorkers = filterByOutstandingUncachedTokens(
                sortedWorkers, roleType, seqLen, config);
        boolean outstandingGuardFallback = eligibleWorkers.isEmpty();
        List<ScoredWorker> selectionWorkers = outstandingGuardFallback ? sortedWorkers : eligibleWorkers;
        List<ScoredWorker> candidates = selectTopCandidates(selectionWorkers);
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
        ScoredWorker selectedWorker = selectWorkerByCachePreference(similarWorkers, candidates);
        recordDecisionSnapshot(
                balanceContext,
                selectedWorker,
                sortedWorkers,
                candidates,
                similarWorkers,
                minTTFT,
                threshold,
                roleType,
                group,
                seqLen,
                (outstandingGuardFallback
                                ? StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK
                                : StrategySelectionReason.SHORTEST_TTFT)
                        .name(),
                null,
                outstandingUncachedTokensThresholdForSnapshot(roleType, config));
        return selectedWorker;
    }

    protected List<ScoredWorker> filterByOutstandingUncachedTokens(List<ScoredWorker> workers,
                                                                   RoleType roleType,
                                                                   long seqLen,
                                                                   FlexlbConfig config) {
        long threshold = configuredOutstandingUncachedTokensThreshold(config);
        if (!outstandingUncachedTokensGuardEnabled(roleType, threshold)) {
            return workers;
        }
        return workers.stream()
                .filter(worker -> worker.worker().getOutstandingUncachedTokens()
                        + Math.max(0, seqLen - worker.hitCacheTokens()) <= threshold)
                .toList();
    }

    private boolean outstandingUncachedTokensGuardEnabled(RoleType roleType, long threshold) {
        return (roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION)
                && threshold > 0;
    }

    protected long configuredOutstandingUncachedTokensThreshold(FlexlbConfig config) {
        return config.getEffectiveOutstandingUncachedTokensThreshold(strategy);
    }

    protected long outstandingUncachedTokensThresholdForSnapshot(RoleType roleType, FlexlbConfig config) {
        long threshold = configuredOutstandingUncachedTokensThreshold(config);
        return outstandingUncachedTokensGuardEnabled(roleType, threshold) ? threshold : 0;
    }

    protected void recordDecisionSnapshot(BalanceContext balanceContext,
                                          ScoredWorker selectedWorker,
                                          List<ScoredWorker> sortedWorkers,
                                          List<ScoredWorker> topCandidates,
                                          List<ScoredWorker> similarWorkers,
                                          long minimumTtft,
                                          double similarTtftThreshold,
                                          RoleType roleType,
                                          String group,
                                          long seqLen,
                                          String selectionReason,
                                          CacheAffinityDecision cacheAffinityDecision,
                                          long outstandingUncachedTokensThreshold) {
        balanceContext.recordSelectionReason(roleType, selectionReason);
        balanceContext.recordShortestTtftDecision(buildDecisionSnapshot(
                balanceContext,
                selectedWorker,
                sortedWorkers,
                topCandidates,
                similarWorkers,
                minimumTtft,
                similarTtftThreshold,
                roleType,
                group,
                seqLen,
                selectionReason,
                cacheAffinityDecision,
                outstandingUncachedTokensThreshold));
    }

    protected void reportCacheAffinityDecision(RoleType roleType, String engineIp, String decision) {
        engineHealthReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    private ShortestTtftDecision buildDecisionSnapshot(BalanceContext balanceContext,
                                                       ScoredWorker selectedWorker,
                                                       List<ScoredWorker> sortedWorkers,
                                                       List<ScoredWorker> topCandidates,
                                                       List<ScoredWorker> similarWorkers,
                                                       long minimumTtft,
                                                       double similarTtftThreshold,
                                                       RoleType roleType,
                                                       String group,
                                                       long seqLen,
                                                       String selectionReason,
                                                       CacheAffinityDecision cacheAffinityDecision,
                                                       long outstandingUncachedTokensThreshold) {
        long decisionTimeMs = System.currentTimeMillis();
        long decisionTimeUs = System.nanoTime() / 1000;
        List<ScoredWorker> snapshotWorkers = selectSnapshotWorkers(
                selectedWorker, sortedWorkers, cacheAffinityDecision);
        List<WorkerDecision> workers = snapshotWorkers.stream()
                .map(scoredWorker -> buildWorkerDecision(
                        scoredWorker,
                        sortedWorkers.indexOf(scoredWorker) + 1,
                        selectedWorker,
                        topCandidates,
                        similarWorkers,
                        seqLen,
                        decisionTimeUs,
                        cacheAffinityDecision,
                        outstandingUncachedTokensThreshold))
                .toList();
        return new ShortestTtftDecision(
                roleType,
                group,
                strategy.getName(),
                selectionReason,
                decisionTimeMs,
                balanceContext.getRetryCount() + 1,
                balanceContext.getConfig() == null ? 0 : balanceContext.getConfig().getP2pHitDiscount(),
                seqLen,
                minimumTtft,
                similarTtftThreshold,
                sortedWorkers.size(),
                topCandidates.size(),
                similarWorkers.size(),
                DECISION_SNAPSHOT_WORKER_LIMIT,
                workers.size() < sortedWorkers.size(),
                workers.stream().mapToLong(WorkerDecision::outstandingUncachedTokens).sum(),
                workers,
                cacheAffinityDecision);
    }

    private List<ScoredWorker> selectSnapshotWorkers(ScoredWorker selectedWorker,
                                                     List<ScoredWorker> sortedWorkers,
                                                     CacheAffinityDecision cacheAffinityDecision) {
        LinkedHashMap<String, ScoredWorker> prioritizedWorkers = new LinkedHashMap<>();
        prioritizedWorkers.put(selectedWorker.worker().getLogicalIpPort(), selectedWorker);
        if (cacheAffinityDecision != null) {
            addSnapshotWorker(
                    prioritizedWorkers,
                    sortedWorkers,
                    cacheAffinityDecision.shortestTtftWorkerIpPort());
            addSnapshotWorker(
                    prioritizedWorkers,
                    sortedWorkers,
                    cacheAffinityDecision.cacheLeaderIpPort());
        }
        sortedWorkers.forEach(worker ->
                prioritizedWorkers.putIfAbsent(worker.worker().getLogicalIpPort(), worker));
        return prioritizedWorkers.values().stream()
                .limit(DECISION_SNAPSHOT_WORKER_LIMIT)
                .sorted(Comparator.comparingInt(sortedWorkers::indexOf))
                .toList();
    }

    /**
     * Adds a decision-snapshot worker addressed by logical {@code ip:port@engineIndex}.
     *
     * @param ipPort logical worker identity; the index identifies one independently routable
     *               engine behind the physical frontend
     */
    private void addSnapshotWorker(Map<String, ScoredWorker> prioritizedWorkers,
                                   List<ScoredWorker> sortedWorkers,
                                   String ipPort) {
        sortedWorkers.stream()
                .filter(worker -> worker.worker().getLogicalIpPort().equals(ipPort))
                .findFirst()
                .ifPresent(worker -> prioritizedWorkers.putIfAbsent(ipPort, worker));
    }

    private WorkerDecision buildWorkerDecision(ScoredWorker scoredWorker,
                                               int estimatedTtftRank,
                                               ScoredWorker selectedWorker,
                                               List<ScoredWorker> topCandidates,
                                               List<ScoredWorker> similarWorkers,
                                               long seqLen,
                                               long decisionTimeUs,
                                               CacheAffinityDecision cacheAffinityDecision,
                                               long outstandingUncachedTokensThreshold) {
        WorkerStatus worker = scoredWorker.worker();
        long requestPrefillTime = TaskInfo.estimatePrefillTimeMs(seqLen, scoredWorker.hitCacheTokens());
        long requestUncachedTokens = Math.max(0, seqLen - scoredWorker.hitCacheTokens());
        double requestHitRatePct = seqLen > 0
                ? scoredWorker.hitCacheTokens() * 100.0 / seqLen
                : 0.0;
        long outstandingUncachedTokens = worker.getOutstandingUncachedTokens();
        Map<String, TaskInfo> trackedTasks = worker.getLocalTaskMap();
        Map<String, TaskInfo> waitingTasks = worker.getWaitingTaskList();
        Map<String, TaskInfo> runningTasks = worker.getRunningTaskList();
        long blockSize = worker.getCacheStatus() == null ? 0 : worker.getCacheStatus().getBlockSize();

        return new WorkerDecision(
                estimatedTtftRank,
                worker.getIp(),
                worker.getPort(),
                topCandidates.contains(scoredWorker),
                similarWorkers.contains(scoredWorker),
                selectedWorker.equals(scoredWorker),
                isDecisionWorker(worker, cacheAffinityDecision == null
                        ? null
                        : cacheAffinityDecision.cacheLeaderIpPort()),
                isDecisionWorker(worker, cacheAffinityDecision == null
                        ? sortedWorkerIpPort(topCandidates)
                        : cacheAffinityDecision.shortestTtftWorkerIpPort()),
                outstandingUncachedTokensThreshold <= 0
                        || outstandingUncachedTokens + requestUncachedTokens
                                <= outstandingUncachedTokensThreshold,
                blockSize,
                scoredWorker.hitCacheTokens(),
                requestHitRatePct,
                requestUncachedTokens,
                scoredWorker.localMatchTokens(),
                scoredWorker.p2pFetchTokens(),
                scoredWorker.p2pTotalMatchTokens(),
                Math.max(0, scoredWorker.p2pTotalMatchTokens() - scoredWorker.localMatchTokens()),
                requestPrefillTime,
                scoredWorker.ttft() - requestPrefillTime,
                scoredWorker.ttft(),
                outstandingUncachedTokens,
                outstandingUncachedTokens + requestUncachedTokens,
                scoredWorker.lastSelectedTime(),
                countTasks(trackedTasks),
                worker.getInTransitAndWaitingTaskCount(),
                worker.getInTransitAndWaitingUncachedTokens(),
                countTrackedRunningTasks(trackedTasks),
                worker.getRunningRemainingPrefillTokens(),
                countTasks(waitingTasks),
                sumUncachedTokens(waitingTasks),
                countTasks(runningTasks),
                sumRunningRemainingPrefillTokens(runningTasks),
                worker.isAlive(),
                worker.getResourceAvailable().get(),
                worker.getAvailableConcurrency(),
                worker.getAvailableKvCacheTokens().get(),
                worker.getUsedKvCacheTokens().get(),
                worker.getStatusVersion().get(),
                elapsedUs(decisionTimeUs, worker.getStatusLastUpdateTime().get()),
                worker.getStatusUpdateIntervalUs().get(),
                elapsedUs(decisionTimeUs, worker.getCacheLastUpdateTime().get()));
    }

    private String sortedWorkerIpPort(List<ScoredWorker> sortedWorkers) {
        return sortedWorkers.isEmpty() ? null : sortedWorkers.getFirst().worker().getLogicalIpPort();
    }

    /**
     * Checks whether a worker has the supplied logical identity.
     *
     * @param ipPort logical worker identity in {@code ip:port@engineIndex} format
     */
    private boolean isDecisionWorker(WorkerStatus worker, String ipPort) {
        return ipPort != null && ipPort.equals(worker.getLogicalIpPort());
    }

    private int countTasks(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0
                : (int) tasks.values().stream().filter(Objects::nonNull).count();
    }

    private int countTrackedRunningTasks(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0
                : (int) tasks.values().stream()
                        .filter(task -> task != null && task.getTaskState() == TaskStateEnum.RUNNING)
                        .count();
    }

    private long sumUncachedTokens(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0
                : tasks.values().stream()
                        .filter(Objects::nonNull)
                        .mapToLong(this::uncachedTokens)
                        .sum();
    }

    private long sumRunningRemainingPrefillTokens(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0
                : tasks.values().stream()
                        .filter(Objects::nonNull)
                        .mapToLong(task -> task.getRemainingPrefillTokens() >= 0
                                ? task.getRemainingPrefillTokens()
                                : uncachedTokens(task))
                        .sum();
    }

    private long uncachedTokens(TaskInfo task) {
        long inputTokens = Math.max(0, task.getInputLength());
        long hitTokens = task.isPrefixLengthValid()
                ? task.getPrefixLength()
                : task.getPredictedPrefixLength();
        return Math.max(0, inputTokens - Math.max(0, Math.min(inputTokens, hitTokens)));
    }

    private long elapsedUs(long nowUs, long timestampUs) {
        return timestampUs > 0 ? Math.max(0, nowUs - timestampUs) : -1;
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
     * Among workers with similar TTFT, prefer any positive cache lead. When cache hits are equal,
     * preserve the shortest-TTFT choice, which selects the shortest queue.
     *
     * @param similarWorkers workers whose TTFT is close to the minimum
     * @param fallbackCandidates candidates sorted by TTFT
     * @return selected worker
     */
    private ScoredWorker selectWorkerByCachePreference(
            List<ScoredWorker> similarWorkers,
            List<ScoredWorker> fallbackCandidates) {
        ScoredWorker shortestTtftWorker = fallbackCandidates.getFirst();
        if (similarWorkers.isEmpty()) {
            return shortestTtftWorker;
        }

        ScoredWorker cacheLeader = similarWorkers.stream()
                .min(Comparator.comparingLong(ScoredWorker::hitCacheTokens)
                        .reversed()
                        .thenComparingLong(ScoredWorker::ttft))
                .orElse(shortestTtftWorker);
        long cacheLeadTokens = Math.max(0, cacheLeader.hitCacheTokens() - shortestTtftWorker.hitCacheTokens());

        Logger.debug(
                "Cache preference - shortest: {}, cacheLeader: {}, cacheLeadTokens: {}, shortestTtft: {}, "
                        + "cacheLeaderTtft: {}",
                shortestTtftWorker.worker().getLogicalIpPort(),
                cacheLeader.worker().getLogicalIpPort(),
                cacheLeadTokens,
                shortestTtftWorker.ttft(),
                cacheLeader.ttft());
        ScoredWorker preferredWorker = cacheLeadTokens > 0 ? cacheLeader : shortestTtftWorker;
        return selectWorkerByScheduleFairness(preferredWorker, similarWorkers, shortestTtftWorker);
    }

    /**
     * Prevent concurrent scheduler threads that observed the same queue snapshot from all
     * selecting one worker. The algorithm's preferred worker is tried first; only a concurrent
     * claim causes another eligible worker to be considered.
     */
    protected ScoredWorker selectWorkerByScheduleFairness(ScoredWorker preferredWorker,
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
            result.setSelectedEngineIndex(
                    workerStatus.getEngineIndex(), workerStatus.getMultiEngineNum());
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
     * @param cacheMatchResult Cache match result
     * @return Number of tokens hit
     */
    private long calculatePrefixMatchLength(
            WorkerStatus workerStatus,
            CacheMatchResult cacheMatchResult,
            double p2pHitDiscount,
            long inputTokens) {
        HostCacheMatch match = cacheMatchResult.hostMatch(workerStatus.getLogicalIpPort());
        if (match == null) {
            return 0L;
        }
        long p2pAddedMatchBlocks = Math.max(0L, match.p2pTotalMatchBlocks() - match.localMatchBlocks());
        double effectiveMatchBlocks = match.localMatchBlocks() + p2pAddedMatchBlocks * Math.max(0.0, p2pHitDiscount);
        return CacheMatchResult.matchedTokens(
                effectiveMatchBlocks, cacheMatchResult.blockSize(), inputTokens);
    }

    private void recordKvcmMatch(
            TaskInfo task, CacheMatchResult cacheMatchResult, HostCacheMatch match, long inputTokens) {
        if (cacheMatchResult.source() != CacheMatchSource.KVCM
                || match == null
                || cacheMatchResult.blockSize() <= 0) {
            return;
        }
        task.setKvcmMatchAvailable(true);
        task.setKvcmLocalMatchTokens(CacheMatchResult.matchedTokens(
                match.localMatchBlocks(), cacheMatchResult.blockSize(), inputTokens));
        task.setKvcmP2pFetchTokens(CacheMatchResult.matchedTokens(
                match.p2pFetchBlocks(), cacheMatchResult.blockSize(), inputTokens));
        task.setKvcmP2pTotalMatchTokens(CacheMatchResult.matchedTokens(
                match.p2pTotalMatchBlocks(), cacheMatchResult.blockSize(), inputTokens));
    }

    private CacheMatchQuery cacheMatchQuery(BalanceContext balanceContext, long blockSize,
                                            RoleType roleType, String group) {
        return new CacheMatchQuery(
                balanceContext.getRequestId(),
                balanceContext.getRequest().getBlockCacheKeys(),
                blockSize,
                balanceContext.getRequest().getLocalStandbyBlockCacheKeys(),
                balanceContext.getRequest().getLocalStandbyBlockSize(),
                roleType,
                group);
    }

}
