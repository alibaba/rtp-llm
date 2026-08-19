package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
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

    /**
     * Fixed engine-wait penalty weight: milliseconds added to the Round-1
     * score per engine-reported waiting stream (waitingQueryLen). 20ms
     * matches one engine status-sync period, so a steadily growing
     * engine-side queue makes the endpoint lose routing attractiveness in
     * real time even when the master-side ledger looks clean.
     */
    private static final double ENGINE_WAIT_PENALTY_MS_PER_WAIT_STREAM = 20.0;

    /**
     * Queue-occupancy ratio (0-1, compile-time constant — zero new config)
     * at which the Round-2 congested-queue filter excludes a prefill
     * endpoint from routing candidates: an endpoint whose batcher queue
     * holds at least {@code ceil(CONGESTED_QUEUE_RATIO ×
     * flexlbBatchQueueMaxSize)} queued requests is "congested" and skipped
     * with a {@code CONGESTED_QUEUE_FILTERED} rejection. Evidence (8/17
     * slow-engine attractor): a slow engine kept winning the score race on
     * its low engineWait report while its queue was flooded far past what
     * the score terms reflected — once the queue is at/above 80% of its
     * hard cap the endpoint is benched until it drains, regardless of what
     * it reports. A non-positive {@code flexlbBatchQueueMaxSize}
     * (unbounded) disables the filter, and when every feasible endpoint is
     * congested the existing least-loaded fallback still returns one
     * endpoint, so routing never fails closed.
     */
    private static final double CONGESTED_QUEUE_RATIO = 0.8;

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

        int selectedIndex = -1;
        if (minScore != Long.MAX_VALUE) {
            long tieThreshold = 0;
            if (config.isScoreTieRandomEnabled()) {
                tieThreshold = Math.max((long) (minScore * config.getScoreTieThresholdPct()), config.getScoreTieThresholdMs());
            }
            long scoreCutoff = minScore + tieThreshold;
            int tiedCount = 0;
            for (int i = 0; i < survivors.size(); i++) {
                if (survivors.score(i) <= scoreCutoff
                        && ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                    selectedIndex = i;
                }
            }
        }

        if (selectedIndex < 0) {
            Map<String, Integer> merged = new java.util.HashMap<>(filterResult.rejections());
            hardFilterResult.rejections().forEach((k, v) -> merged.merge(k, v, Integer::sum));
            Logger.debug("Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId, merged);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        reportCacheHitMetrics(roleType, bestCacheHit, seqLen);

        return buildServerStatus(best, roleType, requestId, minScore, config, bestCacheHit);
    }

    private record EndpointFilterResult(CandidateSet endpoints, Map<String, Integer> rejections) {}
    private static final class CandidateSet {
        private PrefillEndpoint[] endpoints = new PrefillEndpoint[0];
        private long[] cacheHits = new long[0];
        private long[] scores = new long[0];
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
                                  long cacheHit, long score,
                                  long waitMs, long pendingCount) {
            endpoints[index] = endpoint;
            cacheHits[index] = cacheHit;
            scores[index] = score;
            endpointWaitMs[index] = waitMs;
            pendingCounts[index] = pendingCount;
        }

        private void grow(int requiredCapacity) {
            int newCapacity = Math.max(requiredCapacity,
                    Math.max(16, endpoints.length + (endpoints.length >> 1)));
            endpoints = Arrays.copyOf(endpoints, newCapacity);
            cacheHits = Arrays.copyOf(cacheHits, newCapacity);
            scores = Arrays.copyOf(scores, newCapacity);
            endpointWaitMs = Arrays.copyOf(endpointWaitMs, newCapacity);
            pendingCounts = Arrays.copyOf(pendingCounts, newCapacity);
        }

        private void moveSelectionFields(int from, int to) {
            endpoints[to] = endpoints[from];
            cacheHits[to] = cacheHits[from];
            scores[to] = scores[from];
        }

        private void setSelectionFields(int index, PrefillEndpoint endpoint,
                                        long cacheHit, long score) {
            endpoints[index] = endpoint;
            cacheHits[index] = cacheHit;
            scores[index] = score;
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

        private int size() {
            return size;
        }
    }
    private record FilterResult(CandidateSet candidates, Map<String, Integer> rejections) {}

    private FilterResult applyHardFilters(CandidateSet eligible, BalanceContext balanceContext,
                                          FlexlbConfig config, Map<String, Integer> cacheMatchResults) {
        long seqLen = balanceContext.getRequest().getSeqLen();
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

            long cacheHit = calculateCacheHit(ep, cacheMatchResults, seqLen);
            // Cache 收益已折进 estimate(predictor, cacheHit)（hitTokens 参与公式求值），不再单列扣减
            long singlePrefillMs = formulaEstimateMemo.estimate(predictor, cacheHit);

            long endpointWaitMs = ep.realWaitTimeMs();

            if (sloFilterEnabled && endpointWaitMs + singlePrefillMs > sloMs - sloRiskMarginMs) {
                rejections.merge("SLO_VIOLATION", 1, Integer::sum);
                continue;
            }

            long pendingCount = ep.realPendingCount();
            // Auto-TPM (design doc 6.2 simplified): endpointWaitMs +
            // measured queue-age (priority-blind head age) + predictedPrefill.
            // When Auto-TPM is on, all requests carry a normalized priority
            // (1-100); the estimate deliberately ignores it — see
            // PrefillQueueManager.estimateWaitMs.
            long batcherWaitMs = config.isAutoTpmEnabled()
                    ? ep.batcherEstimatedWaitMs(balanceContext.getPriority(),
                            balanceContext.getDeadlineMs(), balanceContext.getRequestId())
                    : ep.batcherWaitMs();
            // Engine-reported wait penalty: each engine-side queued request
            // (waitingQueryLen, ~20ms sync) adds the fixed weight below to
            // the score, so engines whose engine-side queue keeps growing
            // lose routing attractiveness even when the master-side view
            // looks clean. Clamped at 1L<<40 so a pathological
            // waitingQueryLen × weight cannot overflow the long score.
            long engineWaitMs = Math.min(
                    (long) (ep.getReportedWaitingQueryLen() * ENGINE_WAIT_PENALTY_MS_PER_WAIT_STREAM),
                    1L << 40);
            feasible.setCandidate(feasibleCount++, ep, cacheHit,
                    singlePrefillMs + endpointWaitMs + batcherWaitMs + engineWaitMs,
                    endpointWaitMs, pendingCount);
            sumWaitMs += endpointWaitMs;
            sumPendingCount += pendingCount;
        }
        feasible.size = feasibleCount;

        if (feasible.size() == 0) {
            return new FilterResult(feasible, rejections);
        }

        long avgWaitMs = sumWaitMs / feasible.size();
        long avgPendingCount = sumPendingCount / feasible.size();

        // Round 2: congested-queue / hotspot / imbalance filter using cached
        // values (no re-computation; the queue-depth probe is an O(1)
        // atomic read on the batcher)
        int survivorCount = 0;
        PrefillEndpoint leastLoadedEndpoint = null;
        long leastLoadedCacheHit = 0;
        long leastLoadedScore = 0;
        long leastWaitMs = Long.MAX_VALUE;
        long congestedQueueThreshold = congestedQueueThreshold(config);
        int feasibleSize = feasible.size();
        for (int i = 0; i < feasibleSize; i++) {
            long endpointWaitMs = feasible.endpointWaitMs[i];
            long pendingCount = feasible.pendingCounts[i];

            if (endpointWaitMs < leastWaitMs) {
                leastWaitMs = endpointWaitMs;
                leastLoadedEndpoint = feasible.endpoint(i);
                leastLoadedCacheHit = feasible.cacheHit(i);
                leastLoadedScore = feasible.score(i);
            }

            // Congested-queue filter (8/17 slow-engine attractor): bench any
            // endpoint whose batcher queue is at/above 80% of its hard cap —
            // its score can no longer win routing regardless of how low its
            // reported wait is. Tracked above so the least-loaded fallback
            // still sees it when every feasible endpoint is congested.
            if (congestedQueueThreshold > 0
                    && feasible.endpoint(i).getBatcher().queueSize() >= congestedQueueThreshold) {
                rejections.merge("CONGESTED_QUEUE_FILTERED", 1, Integer::sum);
                continue;
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
            feasible.setSelectionFields(0, leastLoadedEndpoint, leastLoadedCacheHit, leastLoadedScore);
            survivorCount = 1;
        }
        feasible.size = survivorCount;

        return new FilterResult(feasible, rejections);
    }

    /**
     * Batch-queue depth at or above which an endpoint is congested:
     * {@code ceil(CONGESTED_QUEUE_RATIO × flexlbBatchQueueMaxSize)}. A
     * non-positive {@code flexlbBatchQueueMaxSize} (unbounded) returns 0,
     * which disables the filter — the caller guards the trigger with
     * {@code congestedQueueThreshold > 0} so 0 never fires on an empty
     * queue.
     */
    private static long congestedQueueThreshold(FlexlbConfig config) {
        int maxQueueSize = config.getFlexlbBatchQueueMaxSize();
        if (maxQueueSize <= 0) {
            return 0;
        }
        return (long) Math.ceil(CONGESTED_QUEUE_RATIO * maxQueueSize);
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

    private long calculateCacheHit(PrefillEndpoint ep, Map<String, Integer> cacheMatchResults, long seqLen) {
        if (ep.getStatus().getCacheStatus() == null
                || cacheMatchResults == null || cacheMatchResults.isEmpty()) {
            return 0L;
        }
        Integer prefixMatchLength = cacheMatchResults.get(ep.ipPort());
        if (prefixMatchLength == null) {
            return 0L;
        }
        long blockSize = ep.getStatus().getCacheStatus().getBlockSize();
        long rawHit = blockSize * prefixMatchLength;
        if (rawHit >= seqLen) {
            return Math.max(0L, seqLen - blockSize);
        }
        return Math.max(0L, rawHit);
    }

    private void reportCacheHitMetrics(RoleType roleType, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, hitCacheTokens, hitRate);
    }

    private ServerStatus buildServerStatus(PrefillEndpoint ep, RoleType roleType, long requestId, long score,
                                            FlexlbConfig config, long bestCacheHit) {
        // Non-batch path: reserve prefill inflight for load-aware scoring.
        // Batch path uses FlexlbBatchScheduler.commitBatch() instead — skip here to avoid double-counting.
        if (isNonBatchPath(config)) {
            ep.commitBatch(requestId, score, Collections.emptyList());
        }

        // Populate DebugInfo so BatchItem.hitCache() can read hitCacheLen for batch metrics
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

    /**
     * Whether batch dispatching is globally disabled.
     * <p>When batch mode is active, FlexlbBatchScheduler handles all inflight tracking;
     * placeholders are only needed when the schedule mode is not BATCH.
     */
    private static boolean isNonBatchPath(FlexlbConfig config) {
        return !config.isBatchPath();
    }
}
