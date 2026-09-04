package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

@Component("costBasedPrefillStrategy")
public class CostBasedPrefillStrategy implements LoadBalanceStrategy {

    private final EngineWorkerStatus engineWorkerStatus;
    private final CacheAwareService cacheAwareService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final EngineHealthReporter engineHealthReporter;
    private final ThreadLocal<CandidateSet> candidateSets =
            ThreadLocal.withInitial(CandidateSet::new);

    public CostBasedPrefillStrategy(EngineWorkerStatus engineWorkerStatus,
                                    CacheAwareService cacheAwareService,
                                    ResourceMeasureFactory resourceMeasureFactory,
                                    EngineHealthReporter engineHealthReporter) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.cacheAwareService = cacheAwareService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.engineHealthReporter = engineHealthReporter;
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.COST_BASED_PREFILL, this);
    }

    @Override
    public ServerStatus select(BalanceContext balanceContext, RoleType roleType, String group) {
        try {
            return doSelect(balanceContext, roleType, group);
        } catch (Exception e) {
            Logger.warn("CostBasedPrefillStrategy select failed", e);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }
    }

    @Override
    public void rollBack(WorkerEndpoint ep, long requestId) {
        // Release non-batch prefill inflight reservation on routing failure.
        // Batch path inflight is managed by FlexlbBatchScheduler — no-op here.
        if (ep instanceof PrefillEndpoint pe) {
            pe.releaseBatch(requestId);
        }
    }

    private ServerStatus doSelect(BalanceContext balanceContext, RoleType roleType, String group) {
        long requestId = balanceContext.getRequestId();
        long seqLen = balanceContext.getRequest().getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();

        EndpointFilterResult filterResult = getAvailableEndpoints(roleType, group,
                config.getResourceMeasureIndicator(roleType), balanceContext.getExcludedPrefillIpPort());
        CandidateSet eligible = filterResult.endpoints();
        if (eligible.size() == 0) {
            Logger.debug("Prefill select failed: no available endpoints, request_id={}, rejections={}",
                    requestId, filterResult.rejections());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        FilterResult hardFilterResult = applyHardFilters(eligible, balanceContext, config, cacheMatchResults);
        CandidateSet survivors = hardFilterResult.candidates();

        // First pass: find the exact minimum score.
        long minScore = Long.MAX_VALUE;
        for (int i = 0; i < survivors.size(); i++) {
            long score = survivors.score(i);
            if (score < minScore) {
                minScore = score;
            }
        }

        int selectedIndex = selectBestCandidate(
                survivors, minScore, roleType, group, seqLen, config);

        if (selectedIndex < 0) {
            Map<String, Integer> merged = new java.util.HashMap<>(filterResult.rejections());
            hardFilterResult.rejections().forEach((k, v) -> merged.merge(k, v, Integer::sum));
            Logger.debug("Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId, merged);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        long selectedScore = survivors.score(selectedIndex);
        long selectedPrefillMs = survivors.prefillMs(selectedIndex);
        reportSelectedEstimates(
                roleType, best, balanceContext, selectedScore, selectedPrefillMs);
        reportCacheHitMetrics(roleType, bestCacheHit, seqLen);

        return buildServerStatus(
                best, roleType, requestId, selectedScore, selectedPrefillMs,
                balanceContext, bestCacheHit);
    }

    private int selectBestCandidate(CandidateSet survivors,
                                    long minScore,
                                    RoleType roleType,
                                    String group,
                                    long seqLen,
                                    FlexlbConfig config) {
        if (survivors.size() == 0 || minScore == Long.MAX_VALUE) {
            return -1;
        }
        if (!config.isCacheAffinityEnabled()) {
            return selectBaselineCandidate(survivors, minScore, config);
        }

        long referenceHitTokens = 0L;
        for (int i = 0; i < survivors.size(); i++) {
            if (survivors.score(i) == minScore) {
                referenceHitTokens = Math.max(referenceHitTokens, survivors.cacheHit(i));
            }
        }
        CacheAffinityPolicy.Decision affinity = CacheAffinityPolicy.evaluate(
                survivors.size(),
                survivors::score,
                survivors::cacheHit,
                minScore,
                referenceHitTokens,
                seqLen,
                config.getCacheAffinityMaxExtraTtftMs(),
                config.getCacheAffinityMinHitRate());

        int selectedIndex = affinity.hasPreference()
                ? selectCacheLeader(survivors, affinity)
                : selectBaselineCandidate(survivors, minScore, config);
        if (selectedIndex >= 0) {
            reportCacheAffinityDecision(
                    roleType, survivors.endpoint(selectedIndex), affinity.reason().name());
            Logger.debug(
                    "CostBasedPrefill cache-affinity decision - role: {}, group: {}, selected: {}, "
                            + "minScoreMs: {}, selectedScoreMs: {}, scoreCutoffMs: {}, "
                            + "hitTokens: {}, reason: {}",
                    roleType, group, survivors.endpoint(selectedIndex).ipPort(),
                    affinity.minScoreMs(), survivors.score(selectedIndex),
                    affinity.scoreCutoffMs(), survivors.cacheHit(selectedIndex),
                    affinity.reason());
        }
        return selectedIndex;
    }

    private int selectBaselineCandidate(
            CandidateSet survivors, long minScore, FlexlbConfig config) {
        long tieThreshold = 0L;
        if (config.isScoreTieRandomEnabled()) {
            tieThreshold = Math.max(
                    Math.max(0L, (long) (minScore * config.getScoreTieThresholdPct())),
                    Math.max(0L, config.getScoreTieThresholdMs()));
        }
        long scoreCutoff = saturatingAdd(minScore, tieThreshold);
        int selectedIndex = -1;
        int tiedCount = 0;
        for (int i = 0; i < survivors.size(); i++) {
            if (survivors.score(i) <= scoreCutoff
                    && ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                selectedIndex = i;
            }
        }
        return selectedIndex;
    }

    private int selectCacheLeader(
            CandidateSet survivors, CacheAffinityPolicy.Decision affinity) {
        int firstIndex = affinity.preferredIndex(0);
        long bestHit = survivors.cacheHit(firstIndex);
        long bestScore = survivors.score(firstIndex);
        int selectedIndex = firstIndex;
        int tiedCount = 1;
        for (int i = 1; i < affinity.preferredCount(); i++) {
            int candidateIndex = affinity.preferredIndex(i);
            if (survivors.cacheHit(candidateIndex) != bestHit
                    || survivors.score(candidateIndex) != bestScore) {
                break;
            }
            if (ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                selectedIndex = candidateIndex;
            }
        }
        return selectedIndex;
    }

    private record EndpointFilterResult(CandidateSet endpoints, Map<String, Integer> rejections) {}
    private static final class CandidateSet {
        private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
        private long[] cacheHits = new long[0];
        private long[] scores = new long[0];
        private long[] prefillMs = new long[0];
        private long[] endpointWaitMs = new long[0];
        private long[] pendingCounts = new long[0];
        private int size;

        private void reset(int expectedCapacity) {
            if (expectedCapacity > endpoints.length) {
                grow(expectedCapacity);
            }
            size = 0;
        }

        private void addEndpoint(PrefillEndpoint endpoint) {
            if (size == endpoints.length) {
                grow(size + 1);
            }
            endpoints[size++] = endpoint;
        }

        private void setCandidate(int index, PrefillEndpoint endpoint,
                                  long cacheHit, long score, long singlePrefillMs,
                                  long waitMs, long pendingCount) {
            endpoints[index] = endpoint;
            cacheHits[index] = cacheHit;
            scores[index] = score;
            prefillMs[index] = singlePrefillMs;
            endpointWaitMs[index] = waitMs;
            pendingCounts[index] = pendingCount;
        }

        private void grow(int requiredCapacity) {
            int newCapacity = Math.max(requiredCapacity,
                    Math.max(16, endpoints.length + (endpoints.length >> 1)));
            endpoints = Arrays.copyOf(endpoints, newCapacity);
            cacheHits = Arrays.copyOf(cacheHits, newCapacity);
            scores = Arrays.copyOf(scores, newCapacity);
            prefillMs = Arrays.copyOf(prefillMs, newCapacity);
            endpointWaitMs = Arrays.copyOf(endpointWaitMs, newCapacity);
            pendingCounts = Arrays.copyOf(pendingCounts, newCapacity);
        }

        private void moveSelectionFields(int from, int to) {
            endpoints[to] = endpoints[from];
            cacheHits[to] = cacheHits[from];
            scores[to] = scores[from];
            prefillMs[to] = prefillMs[from];
        }

        private void setSelectionFields(int index, PrefillEndpoint endpoint,
                                        long cacheHit, long score, long singlePrefillMs) {
            endpoints[index] = endpoint;
            cacheHits[index] = cacheHit;
            scores[index] = score;
            prefillMs[index] = singlePrefillMs;
        }

        private PrefillEndpoint endpoint(int index) {
            return endpoints[index];
        }

        private long cacheHit(int index) {
            return cacheHits[index];
        }

        private long score(int index) {
            return scores[index];
        }

        private long prefillMs(int index) {
            return prefillMs[index];
        }

        private int size() {
            return size;
        }
    }
    private record FilterResult(CandidateSet candidates, Map<String, Integer> rejections) {}

    private FilterResult applyHardFilters(CandidateSet eligible, BalanceContext balanceContext,
                                          FlexlbConfig config, Map<String, Integer> cacheMatchResults) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        long sloMs = config.resolveSloMs(seqLen);
        long sloRiskMarginMs = config.getCostSloRiskMarginMs();
        boolean sloFilterEnabled = config.isCostSloFilterEnabled();
        double hotspotMultiplier = config.getCostHotspotMultiplier();
        double imbalanceMultiplier = config.getCostImbalanceMultiplier();

        int eligibleSize = eligible.size();
        CandidateSet feasible = eligible;
        Map<String, Integer> rejections = new java.util.HashMap<>();
        FormulaEstimateMemo formulaEstimateMemo = new FormulaEstimateMemo(seqLen);
        long sumWaitMs = 0;
        long sumPendingCount = 0;

        // Round 1: SLO filter + cache wait time / pending count for feasible endpoints
        int feasibleCount = 0;
        for (int i = 0; i < eligibleSize; i++) {
            PrefillEndpoint ep = eligible.endpoint(i);
            PrefillTimePredictor predictor = ep.getPredictor();
            if (predictor == null) {
                rejections.merge("PREDICTOR_MISSING", 1, Integer::sum);
                continue;
            }

            long cacheHit = calculateCacheHit(ep, cacheMatchResults, request);
            // Cache benefit is already reflected by predictor.estimateMs(seqLen, cacheHit).
            long singlePrefillMs = Math.max(0L, formulaEstimateMemo.estimate(predictor, cacheHit));

            long endpointWaitMs = Math.max(0L, ep.realWaitTimeMs());

            if (sloFilterEnabled
                    && saturatingAdd(endpointWaitMs, singlePrefillMs)
                            > sloMs - sloRiskMarginMs) {
                rejections.merge("SLO_VIOLATION", 1, Integer::sum);
                continue;
            }

            long pendingCount = ep.realPendingCount();
            long batcherWaitMs = estimatedBatcherWaitMs(ep, balanceContext, config);
            long score = saturatingAdd(
                    singlePrefillMs, saturatingAdd(endpointWaitMs, batcherWaitMs));
            feasible.setCandidate(feasibleCount++, ep, cacheHit, score,
                    singlePrefillMs, endpointWaitMs, pendingCount);
            sumWaitMs = saturatingAdd(sumWaitMs, endpointWaitMs);
            sumPendingCount = saturatingAdd(sumPendingCount, Math.max(0L, pendingCount));
        }
        feasible.size = feasibleCount;

        if (feasible.size() == 0) {
            return new FilterResult(feasible, rejections);
        }

        long avgWaitMs = sumWaitMs / feasible.size();
        long avgPendingCount = sumPendingCount / feasible.size();

        // Round 2: hotspot / imbalance filter using cached values (no re-computation)
        int survivorCount = 0;
        PrefillEndpoint leastLoadedEndpoint = null;
        long leastLoadedCacheHit = 0;
        long leastLoadedScore = 0;
        long leastLoadedPrefillMs = 0;
        long leastWaitMs = Long.MAX_VALUE;
        int feasibleSize = feasible.size();
        for (int i = 0; i < feasibleSize; i++) {
            long endpointWaitMs = feasible.endpointWaitMs[i];
            long pendingCount = feasible.pendingCounts[i];

            if (endpointWaitMs < leastWaitMs) {
                leastWaitMs = endpointWaitMs;
                leastLoadedEndpoint = feasible.endpoint(i);
                leastLoadedCacheHit = feasible.cacheHit(i);
                leastLoadedScore = feasible.score(i);
                leastLoadedPrefillMs = feasible.prefillMs(i);
            }

            if (hotspotMultiplier > 0 && avgPendingCount > 0 && pendingCount > avgPendingCount * hotspotMultiplier) {
                rejections.merge("HOTSPOT_FILTERED", 1, Integer::sum);
                continue;
            }
            if (imbalanceMultiplier > 0 && avgWaitMs > 0 && endpointWaitMs > avgWaitMs * imbalanceMultiplier) {
                rejections.merge("IMBALANCE_FILTERED", 1, Integer::sum);
                continue;
            }

            feasible.moveSelectionFields(i, survivorCount++);
        }

        if (survivorCount == 0 && leastLoadedEndpoint != null) {
            feasible.setSelectionFields(
                    0, leastLoadedEndpoint, leastLoadedCacheHit,
                    leastLoadedScore, leastLoadedPrefillMs);
            survivorCount = 1;
        }
        feasible.size = survivorCount;

        return new FilterResult(feasible, rejections);
    }

    private long estimatedBatcherWaitMs(PrefillEndpoint ep,
                                        BalanceContext balanceContext,
                                        FlexlbConfig config) {
        if (!isBatchPath(balanceContext)) {
            return 0L;
        }
        long waitMs = config.isAutoTpmEnabled()
                ? ep.batcherEstimatedWaitMs(
                        balanceContext.getPriority(),
                        balanceContext.getDeadlineMs(),
                        balanceContext.getRequestId())
                : ep.batcherWaitMs();
        return Math.max(0L, waitMs);
    }

    private EndpointFilterResult getAvailableEndpoints(RoleType roleType, String group,
                                                       ResourceMeasureIndicatorEnum indicator,
                                                       String excludedIpPort) {
        CandidateSet result = candidateSets.get();
        result.reset(engineWorkerStatus.getModelWorkerCapacity(roleType));
        PrefillResourceMeasure measure = (PrefillResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new EndpointFilterResult(result, Map.of("NO_REGISTERED", 1));
        }
        Map<String, Integer> rejections = new java.util.HashMap<>();

        PrefillEndpoint[] excludedEligible = new PrefillEndpoint[1];
        int registered = engineWorkerStatus.forEachModelWorkerEndpoint(roleType, group, (ipPort, ep) -> {
            if (!(ep instanceof PrefillEndpoint pe)) {
                return;
            }
            if (!pe.getStatus().isAlive()) {
                rejections.merge("NOT_ALIVE", 1, Integer::sum);
                return;
            }
            if (!measure.isResourceAvailable(pe)) {
                rejections.merge("RESOURCE_UNAVAILABLE", 1, Integer::sum);
                return;
            }
            // P1-4: skip the worker whose queue just rejected the offer — the
            // fallback re-route must land elsewhere (kept below when it is
            // the only eligible worker).
            if (excludedIpPort != null && excludedIpPort.equals(ipPort)) {
                excludedEligible[0] = pe;
                rejections.merge("EXCLUDED_RETRY", 1, Integer::sum);
                return;
            }
            result.addEndpoint(pe);
        });
        if (registered == 0) {
            return new EndpointFilterResult(result, Map.of("NO_REGISTERED", 1));
        }
        if (result.size() == 0 && excludedEligible[0] != null) {
            // P1-4: single-worker (or fully-filtered) cluster — excluding the
            // only eligible worker would turn a queue-full retry into a hard
            // NO_AVAILABLE_WORKER; keep the legacy candidate set instead.
            result.addEndpoint(excludedEligible[0]);
        }
        return new EndpointFilterResult(result, rejections);
    }

    private static final class FormulaEstimateMemo {
        private static final int MAX_CACHE_HITS = 16;

        private final long seqLen;
        private String formulaKey;
        private long[] estimates;
        private int estimateCount;

        private FormulaEstimateMemo(long seqLen) {
            this.seqLen = seqLen;
        }

        private long estimate(PrefillTimePredictor predictor, long cacheHit) {
            if (!(predictor instanceof FormulaPredictor formulaPredictor)) {
                return predictor.estimateMs(seqLen, cacheHit);
            }
            String key = formulaPredictor.immutableFormulaKey();
            if (formulaKey == null) {
                formulaKey = key;
                estimates = new long[MAX_CACHE_HITS * 2];
            } else if (!formulaKey.equals(key)) {
                return predictor.estimateMs(seqLen, cacheHit);
            }
            for (int i = 0; i < estimateCount; i++) {
                int offset = i * 2;
                if (estimates[offset] == cacheHit) {
                    return estimates[offset + 1];
                }
            }
            long estimate = predictor.estimateMs(seqLen, cacheHit);
            if (estimateCount < MAX_CACHE_HITS) {
                int offset = estimateCount++ * 2;
                estimates[offset] = cacheHit;
                estimates[offset + 1] = estimate;
            }
            return estimate;
        }
    }

    private Map<String, Integer> getCacheMatchResults(BalanceContext balanceContext, RoleType roleType, String group) {
        List<Long> blockCacheKeys = balanceContext.getRequest().getBlockCacheKeys();
        return cacheAwareService.findMatchingEngines(blockCacheKeys, roleType, group);
    }

    private long calculateCacheHit(PrefillEndpoint ep,
                                   Map<String, Integer> cacheMatchResults,
                                   Request request) {
        if (cacheMatchResults == null || request == null || request.getSeqLen() <= 0L) {
            return 0L;
        }
        Integer prefixMatchLength = cacheMatchResults.get(ep.ipPort());
        if (prefixMatchLength == null || prefixMatchLength <= 0) {
            return 0L;
        }
        long blockSize = request.getCacheKeyBlockSize();
        if (blockSize <= 0L && ep.getStatus().getCacheStatus() != null) {
            blockSize = ep.getStatus().getCacheStatus().getBlockSize();
        }
        if (blockSize <= 0L) {
            return 0L;
        }
        long rawHit;
        try {
            rawHit = Math.multiplyExact(blockSize, prefixMatchLength.longValue());
        } catch (ArithmeticException overflow) {
            rawHit = request.getSeqLen();
        }
        if (rawHit >= request.getSeqLen()) {
            return Math.max(0L, request.getSeqLen() - blockSize);
        }
        return Math.max(0L, rawHit);
    }

    private void reportCacheHitMetrics(RoleType roleType, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, hitCacheTokens, hitRate);
    }

    private void reportSelectedEstimates(RoleType roleType,
                                         PrefillEndpoint endpoint,
                                         BalanceContext balanceContext,
                                         long estimatedTtftMs,
                                         long executionTimeMs) {
        String deliveryMode = isBatchPath(balanceContext) ? "BATCH" : "NON_BATCH";
        try {
            engineHealthReporter.reportPrefillSelectedEstimates(
                    roleType, endpoint.getIp(), deliveryMode,
                    estimatedTtftMs, executionTimeMs);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Prefill selected-estimate metric failed: engine={}, delivery_mode={}",
                    endpoint.ipPort(), deliveryMode, telemetryFailure);
        }
    }

    private void reportCacheAffinityDecision(RoleType roleType,
                                             PrefillEndpoint endpoint,
                                             String decision) {
        try {
            engineHealthReporter.reportCacheAffinityDecision(
                    roleType, endpoint.getIp(), decision);
        } catch (RuntimeException telemetryFailure) {
            Logger.warn("Cache-affinity metric failed: engine={}, decision={}",
                    endpoint.ipPort(), decision, telemetryFailure);
        }
    }

    private ServerStatus buildServerStatus(PrefillEndpoint ep,
                                           RoleType roleType,
                                           long requestId,
                                           long score,
                                           long selectedPrefillMs,
                                           BalanceContext balanceContext,
                                           long bestCacheHit) {
        // Non-batch path: reserve only execution work; queue wait is not new work.
        if (!isBatchPath(balanceContext)) {
            ep.commitBatch(requestId, selectedPrefillMs, Collections.emptyList());
        }

        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(bestCacheHit);

        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setRequestId(requestId);
        result.setPrefillTime(score);
        result.setGroup(ep.getStatus().getGroup());
        result.setServerIp(ep.getIp());
        result.setHttpPort(ep.getHttpPort());
        result.setGrpcPort(CommonUtils.toGrpcPort(ep.getHttpPort()));
        result.setDpRank(ep.getStatus().getDpRank());
        result.setDebugInfo(debugInfo);
        return result;
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

    /** Request-level mode is authoritative because configured BATCH may downgrade to DIRECT. */
    private static boolean isBatchPath(BalanceContext balanceContext) {
        return balanceContext != null
                && balanceContext.getScheduleMode() == ScheduleModeEnum.BATCH;
    }
}
