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

        EndpointFilterResult filterResult = getAvailableEndpoints(roleType, group, config.getResourceMeasureIndicator(roleType));
        CandidateSet eligible = filterResult.endpoints();
        if (eligible.size() == 0) {
            Logger.warn("Prefill select failed: no available endpoints, request_id={}, rejections={}",
                    requestId, filterResult.rejections());
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        Map<String, Integer> cacheMatchResults = getCacheMatchResults(balanceContext, roleType, group);

        FilterResult hardFilterResult = applyHardFilters(eligible, seqLen, config, cacheMatchResults);
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
            if (config.isScoreTieRandomEnabled()) {
                // Enabled: reservoir sampling among candidates within threshold
                long tieThreshold = Math.max((long) (minScore * config.getScoreTieThresholdPct()), config.getScoreTieThresholdMs());
                long scoreCutoff = minScore + tieThreshold;
                int tiedCount = 0;
                for (int i = 0; i < survivors.size(); i++) {
                    if (survivors.score(i) <= scoreCutoff
                            && ThreadLocalRandom.current().nextInt(++tiedCount) == 0) {
                        selectedIndex = i;
                    }
                }
            } else {
                // Disabled: deterministically select the first minimum-score candidate
                for (int i = 0; i < survivors.size(); i++) {
                    if (survivors.score(i) == minScore) {
                        selectedIndex = i;
                        break;
                    }
                }
            }
        }

        if (selectedIndex < 0) {
            Map<String, Integer> merged = new java.util.HashMap<>(filterResult.rejections());
            hardFilterResult.rejections().forEach((k, v) -> merged.merge(k, v, Integer::sum));
            Logger.warn("Prefill select failed: all filtered out, request_id={}, rejections={}",
                    requestId, merged);
            return ServerStatus.code(StrategyErrorType.NO_AVAILABLE_WORKER);
        }

        PrefillEndpoint best = survivors.endpoint(selectedIndex);
        long bestCacheHit = survivors.cacheHit(selectedIndex);
        reportCacheHitMetrics(roleType, best.getIp(), best.ipPort(), bestCacheHit, seqLen);

        return buildServerStatus(best, roleType, requestId, minScore, config, balanceContext, bestCacheHit);
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

    private FilterResult applyHardFilters(CandidateSet eligible, long seqLen,
                                          FlexlbConfig config, Map<String, Integer> cacheMatchResults) {
        long sloMs = config.resolveSloMs(seqLen);
        long sloRiskMarginMs = config.getCostSloRiskMarginMs();
        boolean sloFilterEnabled = config.isCostSloFilterEnabled();
        double hotspotMultiplier = config.getCostHotspotMultiplier();
        double imbalanceMultiplier = config.getCostImbalanceMultiplier();

        int eligibleSize = eligible.size();
        CandidateSet feasible = eligible;
        Map<String, Integer> rejections = new java.util.HashMap<>();
        long sumWaitMs = 0;
        long sumPendingCount = 0;

        // Round 1: SLO filter + cache wait time / pending count for feasible endpoints
        int feasibleCount = 0;
        for (int i = 0; i < eligibleSize; i++) {
            PrefillEndpoint ep = eligible.endpoint(i);

            long cacheHit = calculateCacheHit(ep, cacheMatchResults, seqLen);
            long prefillMs = ep.estimateBatchPrefillMs(seqLen, cacheHit);

            long endpointWaitMs = ep.realWaitTimeMs();

            if (sloFilterEnabled && endpointWaitMs + prefillMs > sloMs - sloRiskMarginMs) {
                rejections.merge("SLO_VIOLATION", 1, Integer::sum);
                continue;
            }

            long pendingCount = ep.realPendingCount();
            long batcherWaitMs = ep.batcherWaitMs();
            feasible.setCandidate(feasibleCount++, ep, cacheHit,
                    prefillMs + endpointWaitMs + batcherWaitMs,
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

        // Round 2: hotspot / imbalance filter using cached values (no re-computation)
        int survivorCount = 0;
        PrefillEndpoint leastLoadedEndpoint = null;
        long leastLoadedCacheHit = 0;
        long leastLoadedScore = 0;
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

    private EndpointFilterResult getAvailableEndpoints(RoleType roleType, String group, ResourceMeasureIndicatorEnum indicator) {
        CandidateSet result = candidateSets.get();
        result.reset(engineWorkerStatus.getModelWorkerCapacity(roleType));
        PrefillResourceMeasure measure = (PrefillResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new EndpointFilterResult(result, Map.of("NO_REGISTERED", 1));
        }
        Map<String, Integer> rejections = new java.util.HashMap<>();

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
            result.addEndpoint(pe);
        });
        if (registered == 0) {
            return new EndpointFilterResult(result, Map.of("NO_REGISTERED", 1));
        }
        return new EndpointFilterResult(result, rejections);
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

    private void reportCacheHitMetrics(RoleType roleType, String ip, String engineIpPort, long hitCacheTokens, long seqLen) {
        double hitRate = seqLen > 0 ? hitCacheTokens / (double) seqLen : 0.0;
        engineHealthReporter.reportCacheHitMetrics(roleType, ip, engineIpPort, hitCacheTokens, hitRate);
    }

    private ServerStatus buildServerStatus(PrefillEndpoint ep, RoleType roleType, long requestId, long score,
                                            FlexlbConfig config, BalanceContext balanceContext,
                                            long bestCacheHit) {
        // Non-batch path: reserve prefill inflight for load-aware scoring.
        // Batch path uses FlexlbBatchScheduler.commitBatch() instead — skip here to avoid double-counting.
        if (isNonBatchPath(config, balanceContext)) {
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
     * <p>When batch is enabled, FlexlbBatchScheduler handles all inflight tracking;
     * placeholders are only needed when batch is fully off ({@code flexlbBatchEnabled=false}).
     */
    private static boolean isNonBatchPath(FlexlbConfig config, BalanceContext ctx) {
        return !config.isFlexlbBatchEnabled();
    }
}
