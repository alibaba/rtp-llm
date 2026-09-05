package org.flexlb.balance.strategy;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.ShortestTtftDecision;
import org.flexlb.dao.pv.ShortestTtftDecision.CacheAffinityDecision;
import org.flexlb.dao.pv.ShortestTtftDecision.WorkerDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.StrategySelectionReason;
import org.flexlb.enums.TaskStateEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Load balancing strategy based on shortest Time-To-First-Token (TTFT).
 *
 * <p>This strategy selects the optimal worker by considering the following factors:
 * 1. KV-Cache hit rate: Prioritize workers with higher cache hit rates
 * 2. Queue time: Consider the current task queue status of workers
 * 3. Scheduling fairness: Achieve load balancing among workers with similar performance
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Score all eligible endpoints by TTFT = prefillTime + queueTime</li>
 *   <li>Sort by TTFT ascending, take top-N as the candidate pool</li>
 *   <li>Within the pool, use CAS on {@code lastSelectedTime} to pick the
 *       least-recently-selected worker, ensuring concurrent requests spread
 *       across different workers</li>
 *   <li>If all CAS attempts fail, fall back to the lowest-TTFT candidate</li>
 * </ol>
 *
 * <p>Supports DIRECT and QUEUE scheduling with either dispatcher. Common
 * scheduler paths include the worker-batcher wait while their inflight
 * lifecycle remains owned by {@code PriorityScheduler}; DIRECT and FIFO
 * non-batch queue routing reserve locally.
 */
@Component("shortestTtftStrategy")
public class ShortestTTFTStrategy implements LoadBalanceStrategy {

    private static final int DECISION_SNAPSHOT_WORKER_LIMIT = 5;

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;

    public ShortestTTFTStrategy(EngineWorkerStatus engineWorkerStatus,
                                CacheAwareService cacheAwareService,
                                ResourceMeasureFactory resourceMeasureFactory,
                                EngineHealthReporter engineHealthReporter) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, this);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("{} select failed", LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(), e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(WorkerEndpoint ep, String requestId) {
        // Release non-batch prefill inflight reservation on routing failure.
        // Batch path inflight is managed by PriorityScheduler — no-op here.
        if (ep instanceof PrefillEndpoint pe) {
            pe.releaseBatch(requestId);
            pe.getStatus().removeLocalTask(String.valueOf(requestId));
        }
    }

    /** Internal record holding TTFT score and cache hit for a single endpoint. */
    protected record ScoredEndpoint(PrefillEndpoint ep,
                                    long ttft,
                                    long hitCache,
                                    long prefillMs,
                                    long lastSelectedTime,
                                    long localMatchTokens,
                                    long p2pFetchTokens,
                                    long p2pTotalMatchTokens) {

        protected ScoredEndpoint(PrefillEndpoint ep,
                                 long ttft,
                                 long hitCache,
                                 long prefillMs,
                                 long lastSelectedTime) {
            this(ep, ttft, hitCache, prefillMs, lastSelectedTime, 0, 0, 0);
        }
    }

    // ==================== Core Selection ====================

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        String requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        List<PrefillEndpoint> eligible = getAvailableEndpoints(
                roleType,
                group,
                config.resourceMeasureFor(roleType),
                balanceContext.getExcludedPrefillIpPort());
        if (CollectionUtils.isEmpty(eligible)) {
            Logger.debug("ShortestTTFT select failed: no available endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        CacheMatchResult cacheMatchResult = getCacheMatchResult(balanceContext, roleType, group);

        // Score all eligible endpoints by TTFT
        List<ScoredEndpoint> scoredEndpoints = scoreEndpoints(
                eligible, cacheMatchResult, balanceContext);
        if (scoredEndpoints.isEmpty()) {
            Logger.debug("ShortestTTFT select failed: no scored endpoints, request_id={}", requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
        long candidateMaxHitTokens = scoredEndpoints.stream()
                .mapToLong(ScoredEndpoint::hitCache)
                .max()
                .orElse(0L);

        // Sort by TTFT ascending; secondary sort by the selection-time snapshot for determinism.
        scoredEndpoints.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(ScoredEndpoint::lastSelectedTime));

        ScoredEndpoint selected = selectBestEndpoint(
                scoredEndpoints, balanceContext, roleType, group, seqLen, config);
        if (selected == null) {
            Logger.debug("{} select failed: no selectable endpoint, request_id={}",
                    LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(), requestId);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Logger.debug("{} selected endpoint - ip: {}, port: {}, ttft: {}, hitCache: {}",
                LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(),
                selected.ep().getIp(), selected.ep().getHttpPort(),
                selected.ttft(), selected.hitCache());

        balanceContext.recordCacheMatch(
                cacheMatchResult.source().name(),
                cacheMatchResult.queryTimeUs(),
                roleType,
                selected.ep().getIp(),
                selected.hitCache());

        reportCacheHitMetrics(roleType, selected.ep().getStatus().getIpIndex(), selected.hitCache(), seqLen);
        reportRoutingCacheMatchMetrics(
                roleType,
                selected.hitCache(),
                candidateMaxHitTokens,
                seqLen);

        return buildServerStatus(
                selected, roleType, requestId, balanceContext, cacheMatchResult);
    }

    /**
     * Select the final endpoint from a list already sorted by TTFT.
     * Subclasses may override this decision while reusing filtering, cache lookup,
     * scoring, metrics, and response construction.
     */
    protected ScoredEndpoint selectBestEndpoint(List<ScoredEndpoint> scoredEndpoints,
                                                BalanceContext balanceContext,
                                                RoleType roleType,
                                                String group,
                                                long seqLen,
                                                FlexlbConfig config) {
        long outstandingThreshold = configuredOutstandingUncachedTokensThreshold(config);
        List<ScoredEndpoint> outstandingEligible = filterByOutstandingUncachedTokens(
                scoredEndpoints, roleType, seqLen, outstandingThreshold);
        boolean outstandingGuardFallback = outstandingEligible.isEmpty()
                && outstandingGuardEnabled(roleType, outstandingThreshold);
        List<ScoredEndpoint> selectionPool = outstandingGuardFallback
                ? scoredEndpoints
                : outstandingEligible;

        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        ScoredEndpoint selected;
        String reason;
        CacheAffinityDecision cacheAffinityDecision = null;
        if (cacheAffinity == null) {
            selected = selectBaselineEndpoint(selectionPool, config);
            reason = outstandingGuardFallback
                    ? StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK.name()
                    : StrategySelectionReason.SHORTEST_TTFT.name();
        } else {
            long minTtftMs = selectionPool.getFirst().ttft();
            long referenceHitTokens = selectionPool.stream()
                    .filter(candidate -> candidate.ttft() == minTtftMs)
                    .mapToLong(ScoredEndpoint::hitCache)
                    .max()
                    .orElse(0L);
            CacheAffinityPolicy.Decision affinity = CacheAffinityPolicy.evaluate(
                    selectionPool.size(),
                    index -> selectionPool.get(index).ttft(),
                    index -> selectionPool.get(index).hitCache(),
                    minTtftMs,
                    referenceHitTokens,
                    seqLen,
                    cacheAffinity.getMaxExtraTtftMs(),
                    cacheAffinity.getMinPrefixHitPercent());

            if (affinity.hasPreference()) {
                List<ScoredEndpoint> preferenceOrder = new ArrayList<>(affinity.preferredCount());
                for (int i = 0; i < affinity.preferredCount(); i++) {
                    preferenceOrder.add(selectionPool.get(affinity.preferredIndex(i)));
                }
                selected = selectFirstWithoutConcurrentConflict(preferenceOrder);
                if (selected == null) {
                    selected = selectBaselineEndpoint(
                            refreshSelectionSnapshots(selectionPool), config);
                    reason = StrategySelectionReason.CACHE_AFFINITY_FALLBACK.name();
                } else {
                    reason = selected.equals(preferenceOrder.getFirst())
                            ? StrategySelectionReason.CACHE_LEADER.name()
                            : StrategySelectionReason.CACHE_AFFINITY_FALLBACK.name();
                }
            } else {
                selected = selectBaselineEndpoint(selectionPool, config);
                reason = switch (affinity.reason()) {
                    case LOW_CACHE_HIT -> StrategySelectionReason.SHORTEST_TTFT_LOW_CACHE_HIT.name();
                    case CACHE_LEADER -> StrategySelectionReason.CACHE_LEADER.name();
                    default -> StrategySelectionReason.SHORTEST_TTFT.name();
                };
            }

            ScoredEndpoint shortest = scoredEndpoints.getFirst();
            ScoredEndpoint cacheLeader = scoredEndpoints.stream()
                    .min(Comparator.comparingLong(ScoredEndpoint::hitCache).reversed()
                            .thenComparingLong(ScoredEndpoint::ttft)
                            .thenComparingLong(ScoredEndpoint::lastSelectedTime))
                    .orElse(shortest);
            if (!outstandingGuardFallback && !selectionPool.contains(cacheLeader)) {
                reason = StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD.name();
            } else if (outstandingGuardFallback) {
                reason = StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK.name();
            }
            cacheAffinityDecision = new CacheAffinityDecision(
                    cacheLeader.ep().ipPort(),
                    shortest.ep().ipPort(),
                    Math.max(0L, cacheLeader.hitCache() - shortest.hitCache()),
                    cacheLeader.ttft() - shortest.ttft(),
                    cacheAffinity.getMaxExtraTtftMs(),
                    outstandingThreshold,
                    selectionPool.contains(cacheLeader));
        }

        if (selected != null) {
            balanceContext.recordSelectionReason(roleType, reason);
            recordDecisionSnapshot(
                    balanceContext,
                    selected,
                    scoredEndpoints,
                    selectionPool,
                    roleType,
                    group,
                    seqLen,
                    reason,
                    cacheAffinityDecision,
                    outstandingThreshold,
                    config);
            reportCacheAffinityDecision(roleType, selected.ep().getStatus().getIpIndex(), reason);
            Logger.debug(
                    "ShortestTtft decision - role: {}, group: {}, selected: {}, "
                            + "selectedTtftMs: {}, hitTokens: {}, reason: {}",
                    roleType,
                    group,
                    selected.ep().ipPort(),
                    selected.ttft(),
                    selected.hitCache(),
                    reason);
        }
        return selected;
    }

    private List<ScoredEndpoint> filterByOutstandingUncachedTokens(
            List<ScoredEndpoint> scoredEndpoints,
            RoleType roleType,
            long seqLen,
            long threshold) {
        if (!outstandingGuardEnabled(roleType, threshold)) {
            return scoredEndpoints;
        }
        return scoredEndpoints.stream()
                .filter(scored -> scored.ep().getStatus().getOutstandingUncachedTokens()
                        + Math.max(0L, seqLen - scored.hitCache()) <= threshold)
                .toList();
    }

    private boolean outstandingGuardEnabled(RoleType roleType, long threshold) {
        return (roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION)
                && threshold > 0L;
    }

    private long configuredOutstandingUncachedTokensThreshold(FlexlbConfig config) {
        RoutingConfig.CacheAffinityConfig cacheAffinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        return cacheAffinity == null
                ? 0L
                : Math.max(0L, cacheAffinity.getMaxOutstandingUncachedTokens());
    }

    /** Refresh CAS snapshots after an affinity claim loses to concurrent schedulers. */
    private List<ScoredEndpoint> refreshSelectionSnapshots(
            List<ScoredEndpoint> scoredEndpoints) {
        List<ScoredEndpoint> refreshed = new ArrayList<>(scoredEndpoints.size());
        for (ScoredEndpoint scored : scoredEndpoints) {
            refreshed.add(new ScoredEndpoint(
                    scored.ep(),
                    scored.ttft(),
                    scored.hitCache(),
                    scored.prefillMs(),
                    scored.ep().getLastSelectedTime().get(),
                    scored.localMatchTokens(),
                    scored.p2pFetchTokens(),
                    scored.p2pTotalMatchTokens()));
        }
        refreshed.sort(Comparator.comparingLong(ScoredEndpoint::ttft)
                .thenComparingLong(ScoredEndpoint::lastSelectedTime));
        return refreshed;
    }

    /** Preserve the original candidate-pool and CAS fairness behavior. */
    private ScoredEndpoint selectBaselineEndpoint(
            List<ScoredEndpoint> scoredEndpoints, FlexlbConfig config) {
        int candidateCount = config.shortestTtftCandidateCount(scoredEndpoints.size());
        List<ScoredEndpoint> candidates = scoredEndpoints.subList(
                0, Math.min(candidateCount, scoredEndpoints.size()));

        ScoredEndpoint selected = selectByFairness(candidates);
        if (selected != null) {
            return selected;
        }

        ScoredEndpoint fallback = candidates.getFirst();
        Logger.debug("ShortestTtft: all CAS claims failed, falling back to lowest-TTFT endpoint, ip={}",
                fallback.ep().getIp());
        return fallback;
    }

    /**
     * Select worker based on scheduling fairness.
     *
     * <p>Among the candidate pool, prefer the least-recently-selected worker.
     * CAS on {@code lastSelectedTime} ensures concurrent requests are spread
     * across different workers rather than all landing on the same one.
     *
     * @param candidates candidate pool (already sorted by TTFT ascending)
     * @return selected endpoint, or {@code null} if all CAS attempts failed
     */
    protected ScoredEndpoint selectByFairness(List<ScoredEndpoint> candidates) {
        if (candidates.isEmpty()) {
            return null;
        }

        // Sort ascending by lastSelectedTime so the least recently used worker is tried first
        List<ScoredEndpoint> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingLong(ScoredEndpoint::lastSelectedTime));
        return selectFirstWithoutConcurrentConflict(sorted);
    }

    /**
     * Claim the first endpoint whose selection timestamp still matches the scoring snapshot.
     * A failed claim means another scheduler made a decision from the same snapshot.
     */
    protected ScoredEndpoint selectFirstWithoutConcurrentConflict(List<ScoredEndpoint> selectionOrder) {
        long now = System.nanoTime() / 1000;
        for (ScoredEndpoint candidate : selectionOrder) {
            long expected = candidate.lastSelectedTime();
            if (expected == Long.MAX_VALUE) {
                continue;
            }
            long claimedAt = Math.max(now, expected + 1L);
            if (candidate.ep().getLastSelectedTime().compareAndSet(expected, claimedAt)) {
                return candidate;
            }
        }
        return null;
    }

    protected void reportCacheAffinityDecision(RoleType roleType,
                                               String engineIp,
                                               String decision) {
        engineHealthReporter.reportCacheAffinityDecision(roleType, engineIp, decision);
    }

    private void recordDecisionSnapshot(
            BalanceContext balanceContext,
            ScoredEndpoint selected,
            List<ScoredEndpoint> sortedEndpoints,
            List<ScoredEndpoint> selectionPool,
            RoleType roleType,
            String group,
            long seqLen,
            String selectionReason,
            CacheAffinityDecision cacheAffinityDecision,
            long outstandingThreshold,
            FlexlbConfig config) {
        int candidateCount = config.shortestTtftCandidateCount(selectionPool.size());
        List<ScoredEndpoint> candidates = selectionPool.subList(
                0, Math.min(candidateCount, selectionPool.size()));
        long nowUs = System.nanoTime() / 1_000;
        List<ScoredEndpoint> snapshotEndpoints = selectSnapshotEndpoints(
                selected, sortedEndpoints, cacheAffinityDecision);
        List<WorkerDecision> workers = snapshotEndpoints.stream()
                .map(scored -> buildWorkerDecision(
                        scored,
                        sortedEndpoints.indexOf(scored) + 1,
                        selected,
                        candidates,
                        seqLen,
                        nowUs,
                        cacheAffinityDecision,
                        outstandingThreshold))
                .toList();
        RoutingConfig.CacheAffinityConfig affinity = config.getRouter().getRoles()
                .getPrefill().getCacheAffinity();
        double p2pDiscount = affinity == null
                ? 0.2
                : Math.max(0.0, affinity.getP2pHitDiscount());
        double toleranceMs = affinity == null
                ? 0.0
                : Math.max(0L, affinity.getMaxExtraTtftMs());
        balanceContext.recordShortestTtftDecision(new ShortestTtftDecision(
                roleType,
                group,
                LoadBalanceStrategyEnum.SHORTEST_TTFT.getName(),
                selectionReason,
                System.currentTimeMillis(),
                Math.max(1, balanceContext.getScheduleAttempt()),
                p2pDiscount,
                seqLen,
                sortedEndpoints.getFirst().ttft(),
                toleranceMs,
                sortedEndpoints.size(),
                candidates.size(),
                candidates.size(),
                DECISION_SNAPSHOT_WORKER_LIMIT,
                workers.size() < sortedEndpoints.size(),
                workers.stream().mapToLong(WorkerDecision::outstandingUncachedTokens).sum(),
                workers,
                cacheAffinityDecision));
    }

    private List<ScoredEndpoint> selectSnapshotEndpoints(
            ScoredEndpoint selected,
            List<ScoredEndpoint> sortedEndpoints,
            CacheAffinityDecision cacheAffinityDecision) {
        LinkedHashMap<String, ScoredEndpoint> prioritized = new LinkedHashMap<>();
        prioritized.put(selected.ep().ipPort(), selected);
        if (cacheAffinityDecision != null) {
            addSnapshotEndpoint(
                    prioritized,
                    sortedEndpoints,
                    cacheAffinityDecision.shortestTtftWorkerIpPort());
            addSnapshotEndpoint(
                    prioritized,
                    sortedEndpoints,
                    cacheAffinityDecision.cacheLeaderIpPort());
        }
        sortedEndpoints.forEach(scored -> prioritized.putIfAbsent(
                scored.ep().ipPort(), scored));
        return prioritized.values().stream()
                .limit(DECISION_SNAPSHOT_WORKER_LIMIT)
                .sorted(Comparator.comparingInt(sortedEndpoints::indexOf))
                .toList();
    }

    private void addSnapshotEndpoint(
            Map<String, ScoredEndpoint> prioritized,
            List<ScoredEndpoint> sortedEndpoints,
            String ipPort) {
        sortedEndpoints.stream()
                .filter(scored -> scored.ep().ipPort().equals(ipPort))
                .findFirst()
                .ifPresent(scored -> prioritized.putIfAbsent(ipPort, scored));
    }

    private WorkerDecision buildWorkerDecision(
            ScoredEndpoint scored,
            int estimatedTtftRank,
            ScoredEndpoint selected,
            List<ScoredEndpoint> candidates,
            long seqLen,
            long decisionTimeUs,
            CacheAffinityDecision cacheAffinityDecision,
            long outstandingThreshold) {
        WorkerStatus worker = scored.ep().getStatus();
        long requestUncachedTokens = Math.max(0L, seqLen - scored.hitCache());
        double requestHitRatePct = seqLen > 0L
                ? scored.hitCache() * 100.0 / seqLen
                : 0.0;
        long outstandingUncachedTokens = worker.getOutstandingUncachedTokens();
        Map<String, TaskInfo> trackedTasks = worker.getLocalTaskMap();
        Map<String, TaskInfo> waitingTasks = worker.getWaitingTaskList();
        Map<String, TaskInfo> runningTasks = worker.getRunningTaskList();
        long blockSize = worker.getCacheStatus() == null
                ? 0L
                : worker.getCacheStatus().getBlockSize();
        String cacheLeader = cacheAffinityDecision == null
                ? null
                : cacheAffinityDecision.cacheLeaderIpPort();
        String shortest = cacheAffinityDecision == null
                ? candidates.isEmpty() ? null : candidates.getFirst().ep().ipPort()
                : cacheAffinityDecision.shortestTtftWorkerIpPort();

        return new WorkerDecision(
                estimatedTtftRank,
                worker.getIp(),
                worker.getPort(),
                candidates.contains(scored),
                candidates.contains(scored),
                selected.equals(scored),
                worker.getLogicalIpPort().equals(cacheLeader),
                worker.getLogicalIpPort().equals(shortest),
                outstandingThreshold <= 0L
                        || outstandingUncachedTokens + requestUncachedTokens
                                <= outstandingThreshold,
                blockSize,
                scored.hitCache(),
                requestHitRatePct,
                requestUncachedTokens,
                scored.localMatchTokens(),
                scored.p2pFetchTokens(),
                scored.p2pTotalMatchTokens(),
                Math.max(0L, scored.p2pTotalMatchTokens() - scored.localMatchTokens()),
                scored.prefillMs(),
                Math.max(0L, scored.ttft() - scored.prefillMs()),
                scored.ttft(),
                outstandingUncachedTokens,
                outstandingUncachedTokens + requestUncachedTokens,
                scored.lastSelectedTime(),
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

    private int countTasks(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0
                : (int) tasks.values().stream().filter(Objects::nonNull).count();
    }

    private int countTrackedRunningTasks(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0
                : (int) tasks.values().stream()
                        .filter(task -> task != null
                                && task.getTaskState() == TaskStateEnum.RUNNING)
                        .count();
    }

    private long sumUncachedTokens(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0L
                : tasks.values().stream()
                        .filter(Objects::nonNull)
                        .mapToLong(this::uncachedTokens)
                        .sum();
    }

    private long sumRunningRemainingPrefillTokens(Map<String, TaskInfo> tasks) {
        return MapUtils.isEmpty(tasks)
                ? 0L
                : tasks.values().stream()
                        .filter(Objects::nonNull)
                        .mapToLong(task -> task.getRemainingPrefillTokens() >= 0L
                                ? task.getRemainingPrefillTokens()
                                : uncachedTokens(task))
                        .sum();
    }

    private long uncachedTokens(TaskInfo task) {
        long inputTokens = Math.max(0L, task.getInputLength());
        long hitTokens = task.isPrefixLengthValid()
                ? task.getPrefixLength()
                : task.getPredictedPrefixLength();
        return Math.max(0L,
                inputTokens - Math.max(0L, Math.min(inputTokens, hitTokens)));
    }

    private long elapsedUs(long nowUs, long timestampUs) {
        return timestampUs > 0L ? Math.max(0L, nowUs - timestampUs) : -1L;
    }

    // ==================== Scoring ====================

    /**
     * Calculate TTFT scores for all eligible endpoints.
     *
     * <p>TTFT = predicted prefill time + estimated queue wait time.
     * Endpoints without a predictor are skipped.
     *
     * @param endpoints eligible endpoint list
     * @param cacheMatchResult cache match results from {@link CacheAwareService}
     * @param balanceContext request and scheduling context
     * @return list of scored endpoints
     */
    private List<ScoredEndpoint> scoreEndpoints(List<PrefillEndpoint> endpoints,
                                                CacheMatchResult cacheMatchResult,
                                                BalanceContext balanceContext) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        RoutingConfig.CacheAffinityConfig affinityConfig = balanceContext.getConfig()
                .getRouter().getRoles().getPrefill().getCacheAffinity();
        double p2pHitDiscount = affinityConfig == null
                ? 0.2
                : Math.max(0.0, affinityConfig.getP2pHitDiscount());
        List<ScoredEndpoint> result = new ArrayList<>(endpoints.size());
        for (PrefillEndpoint ep : endpoints) {
            PrefillTimePredictor predictor = ep.getPredictor();
            if (predictor == null) {
                Logger.debug("ShortestTTFT: skipping endpoint without predictor, ip={}", ep.getIp());
                continue;
            }
            HostCacheMatch match = cacheMatchResult.hostMatch(ep.getStatus());
            long localMatchBlocks = match == null ? 0L : match.localMatchBlocks();
            long p2pFetchBlocks = match == null ? 0L : match.p2pFetchBlocks();
            long p2pTotalMatchBlocks = match == null ? 0L : match.p2pTotalMatchBlocks();
            long p2pAddedMatchBlocks = Math.max(
                    0L, p2pTotalMatchBlocks - localMatchBlocks);
            double effectiveMatchBlocks = localMatchBlocks
                    + p2pAddedMatchBlocks * p2pHitDiscount;
            long cacheHit = CacheMatchResult.matchedTokens(
                    effectiveMatchBlocks, cacheMatchResult.blockSize(), seqLen);
            long localMatchTokens = CacheMatchResult.matchedTokens(
                    localMatchBlocks, cacheMatchResult.blockSize(), seqLen);
            long p2pFetchTokens = CacheMatchResult.matchedTokens(
                    p2pFetchBlocks, cacheMatchResult.blockSize(), seqLen);
            long p2pTotalMatchTokens = CacheMatchResult.matchedTokens(
                    p2pTotalMatchBlocks, cacheMatchResult.blockSize(), seqLen);
            long prefillMs = Math.max(0L, predictor.estimateMs(seqLen, cacheHit));
            long queueMs = estimatedQueueWaitMs(ep, balanceContext);
            if (queueMs == Long.MAX_VALUE) {
                Logger.debug("ShortestTTFT: skipping endpoint with unavailable wait estimate, ip={}",
                        ep.getIp());
                continue;
            }
            long ttft = saturatingAdd(prefillMs, queueMs);
            Logger.debug("ShortestTTFT score - ip: {}, hitCache: {}, prefillMs: {}, queueMs: {}, ttft: {}",
                    ep.getIp(), cacheHit, prefillMs, queueMs, ttft);
            result.add(new ScoredEndpoint(
                    ep,
                    ttft,
                    cacheHit,
                    prefillMs,
                    ep.getLastSelectedTime().get(),
                    localMatchTokens,
                    p2pFetchTokens,
                    p2pTotalMatchTokens));
        }
        return result;
    }

    /**
     * Estimate all work already ahead of a request. Queue-scheduler-owned paths include
     * both dispatched inflight work and the per-worker batcher queue; otherwise only the
     * inflight ledger is relevant. Long.MAX_VALUE remains an unavailable sentinel.
     */
    protected long estimatedQueueWaitMs(PrefillEndpoint ep, BalanceContext balanceContext) {
        long inflightWaitMs = ep.realWaitTimeMs();
        if (inflightWaitMs == Long.MAX_VALUE) {
            return Long.MAX_VALUE;
        }
        inflightWaitMs = Math.max(0L, inflightWaitMs);
        if (!queueSchedulerOwnsRequest(balanceContext)) {
            return inflightWaitMs;
        }

        FlexlbConfig config = balanceContext.getConfig();
        long batcherWaitMs = config.isPriorityOrdering()
                ? ep.batcherEstimatedWaitMs(
                        balanceContext.getPriority(),
                        balanceContext.getRequestId())
                : ep.batcherWaitMs();
        batcherWaitMs = Math.max(0L, batcherWaitMs);
        return saturatingAdd(inflightWaitMs, batcherWaitMs);
    }

    private static long saturatingAdd(long left, long right) {
        if (right > 0L && left > Long.MAX_VALUE - right) {
            return Long.MAX_VALUE;
        }
        if (right < 0L && left < Long.MIN_VALUE - right) {
            return Long.MIN_VALUE;
        }
        return left + right;
    }

    // ==================== Endpoint Filtering (mirrors CostBasedPrefillStrategy) ====================

    private List<PrefillEndpoint> getAvailableEndpoints(RoleType roleType,
                                                        String group,
                                                        ResourceMeasureIndicatorEnum indicator,
                                                        String excludedIpPort) {
        Map<String, WorkerEndpoint> workerEndpointMap = engineWorkerStatus.selectRoutableModelWorkerStatus(roleType, group);
        if (MapUtils.isEmpty(workerEndpointMap)) {
            return new ArrayList<>();
        }
        PrefillResourceMeasure measure = (PrefillResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new ArrayList<>();
        }
        List<PrefillEndpoint> result = new ArrayList<>();
        PrefillEndpoint excludedEligible = null;
        for (WorkerEndpoint ep : workerEndpointMap.values()) {
            if (!(ep instanceof PrefillEndpoint pe)) {
                continue;
            }
            if (!engineWorkerStatus.isPhysicalGroupHealthy(pe)) {
                continue;
            }
            if (!measure.isResourceAvailable(pe)) {
                continue;
            }
            if (excludedIpPort != null && excludedIpPort.equals(pe.ipPort())) {
                excludedEligible = pe;
                continue;
            }
            result.add(pe);
        }
        if (result.isEmpty() && excludedEligible != null) {
            result.add(excludedEligible);
        }
        return result;
    }

    private CacheMatchResult getCacheMatchResult(
            BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long blockSize = request.getBlockSize() > 0L
                ? request.getBlockSize()
                : request.getCacheKeyBlockSize();
        return cacheAwareService.findMatchingEngines(new CacheMatchQuery(
                String.valueOf(balanceContext.getRequestId()),
                request.getBlockCacheKeys(),
                blockSize,
                request.getLocalStandbyBlockCacheKeys(),
                request.getLocalStandbyBlockSize(),
                roleType,
                group));
    }

    // ==================== Metrics & ServerStatus (mirrors CostBasedPrefillStrategy) ====================

    private void reportCacheHitMetrics(RoleType roleType, String ipIndex, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ipIndex, hitCacheTokens, hitRate);
    }

    private void reportRoutingCacheMatchMetrics(RoleType roleType,
                                                long selectedHitTokens,
                                                long candidateMaxHitTokens,
                                                long totalTokens) {
        engineHealthReporter.reportRoutingSelectedCacheMatchMetrics(
                roleType, selectedHitTokens, totalTokens);
        engineHealthReporter.reportRoutingCandidateMaxCacheMatchMetrics(
                roleType, candidateMaxHitTokens);
    }

    private ServerStatus buildServerStatus(ScoredEndpoint selected,
                                           RoleType roleType,
                                           String requestId,
                                           BalanceContext balanceContext,
                                           CacheMatchResult cacheMatchResult) {
        PrefillEndpoint ep = selected.ep();
        long ttft = selected.ttft();
        long bestCacheHit = selected.hitCache();

        if (!engineWorkerStatus.isPhysicalGroupHealthy(ep)) {
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        // DIRECT owns its reservation here; QUEUE owns reservations in the scheduler.
        if (strategyOwnsInflightTracking(balanceContext)) {
            ep.commitBatch(requestId, selected.prefillMs(), Collections.emptyList());
        }

        String lifecycleRequestId = String.valueOf(requestId);
        WorkerStatus workerStatus = ep.getStatus();
        if (!workerStatus.getLocalTaskMap().containsKey(lifecycleRequestId)) {
            TaskInfo task = createTaskInfo(
                    requestId,
                    balanceContext.getRequest().getSeqLen(),
                    bestCacheHit,
                    cacheMatchResult.source().name());
            recordKvcmMatch(
                    task,
                    cacheMatchResult,
                    cacheMatchResult.hostMatch(ep.getStatus()),
                    balanceContext.getRequest().getSeqLen());
            workerStatus.putLocalTask(lifecycleRequestId, task);
        }

        // Populate DebugInfo so BatchItem.hitCache() can read hitCacheLen for batch metrics
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(bestCacheHit);

        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setRequestId(requestId);
        result.setPrefillTime(ttft);
        result.setGroup(ep.getStatus().getGroup());
        result.setServerIp(ep.getIp());
        result.setHttpPort(ep.getHttpPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(ep.getHttpPort()));
        result.setDpRank(ep.getStatus().getDpRank());
        result.setSelectedEngineIndex(ep.getStatus().getEngineIndex(),
                ep.getStatus().getMultiEngineNum());
        result.setDebugInfo(debugInfo);
        return result;
    }

    private TaskInfo createTaskInfo(
            String requestId, long inputLength, long prefixLength, String cacheMatchSource) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setPredictedPrefixLength(prefixLength);
        task.setCacheMatchSource(cacheMatchSource);
        return task;
    }

    private void recordKvcmMatch(
            TaskInfo task,
            CacheMatchResult cacheMatchResult,
            HostCacheMatch match,
            long inputTokens) {
        if (cacheMatchResult.source() != CacheMatchSource.KVCM
                || match == null
                || cacheMatchResult.blockSize() <= 0L) {
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

    /** Whether this request is owned by the common queue/batch scheduler. */
    private static boolean queueSchedulerOwnsRequest(BalanceContext balanceContext) {
        if (balanceContext == null || balanceContext.getConfig() == null) {
            return false;
        }
        FlexlbConfig config = balanceContext.getConfig();
        return config.isQueue();
    }

    /** Whether this strategy, rather than PriorityScheduler, owns Prefill accounting. */
    private static boolean strategyOwnsInflightTracking(BalanceContext balanceContext) {
        return !queueSchedulerOwnsRequest(balanceContext);
    }
}
