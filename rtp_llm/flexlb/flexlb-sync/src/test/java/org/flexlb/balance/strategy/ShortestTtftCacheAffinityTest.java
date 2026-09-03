package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.StrategySelectionReason;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;

class ShortestTtftCacheAffinityTest {

    private EngineWorkerStatus engineWorkerStatus;
    private CacheAwareService cacheAwareService;
    private ResourceMeasureFactory resourceMeasureFactory;
    private EngineHealthReporter engineHealthReporter;
    private EndpointRegistry endpointRegistry;
    private ShortestTTFTStrategy strategy;

    @BeforeEach
    void setUp() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        cacheAwareService = Mockito.mock(CacheAwareService.class);
        resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        PriorityScheduler batchScheduler = Mockito.mock(PriorityScheduler.class);

        endpointRegistry = new EndpointRegistry(
                configService,
                () -> batchScheduler,
                Mockito.mock(BatchSchedulerReporter.class));
        engineWorkerStatus = new EngineWorkerStatus(endpointRegistry);

        PrefillResourceMeasure prefillResourceMeasure = Mockito.mock(PrefillResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(any())).thenReturn(prefillResourceMeasure);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(any(CacheMatchQuery.class)))
                .thenReturn(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));

        strategy = new ShortestTTFTStrategy(
                engineWorkerStatus,
                cacheAwareService,
                resourceMeasureFactory,
                engineHealthReporter);
    }

    @Test
    void selectsGlobalCacheLeaderWithinExtraTtftBound() {
        FlexlbConfig config = cacheAffinityConfig(150, 5);
        useFixedCandidatePool(config, 1);

        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 650);
        stubCacheMatches(Map.of("10.0.0.2:8080", 3));

        ServerStatus result = strategy.select(buildContext(1000, "1", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
        assertEquals(768, result.getDebugInfo().getHitCacheLen());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
    }

    @Test
    void selectsCacheLeaderWhenItIsAlsoShortestTtft() {
        FlexlbConfig config = cacheAffinityConfig(0, 5);
        useFixedCandidatePool(config, 2);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 0);
        stubCacheMatches(Map.of("10.0.0.1:8080", 3));

        endpointRegistry.getPrefill("10.0.0.1:8080").getLastSelectedTime().set(2000L);
        endpointRegistry.getPrefill("10.0.0.2:8080").getLastSelectedTime().set(1000L);

        ServerStatus result = strategy.select(buildContext(1000, "11", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "CACHE_LEADER");
    }

    @Test
    void fallsBackToShortestTtftWhenCacheLeaderExceedsBound() {
        FlexlbConfig config = cacheAffinityConfig(50, 5);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 650);
        stubCacheMatches(Map.of("10.0.0.2:8080", 3));

        ServerStatus result = strategy.select(buildContext(1000, "2", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void fallsBackToShortestTtftBelowMinimumHitRate() {
        FlexlbConfig config = cacheAffinityConfig(200, 30);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 250);
        stubCacheMatches(Map.of("10.0.0.2:8080", 1));

        ServerStatus result = strategy.select(buildContext(1000, "3", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void coldRequestKeepsShortestTtftOrdering() {
        FlexlbConfig config = cacheAffinityConfig(1000, 0);
        addWorker("10.0.0.1", 200);
        addWorker("10.0.0.2", 0);

        ServerStatus result = strategy.select(buildContext(1000, "4", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void coldRequestRetainsShortestTtftCandidatePoolFairness() {
        FlexlbConfig config = cacheAffinityConfig(1000, 0);
        useFixedCandidatePool(config, 2);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 10);

        endpointRegistry.getPrefill("10.0.0.1:8080").getLastSelectedTime().set(2000L);
        endpointRegistry.getPrefill("10.0.0.2:8080").getLastSelectedTime().set(1000L);

        ServerStatus result = strategy.select(buildContext(1000, "42", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void batcherQueueWaitParticipatesInTtftScore() {
        PrefillEndpoint endpoint = Mockito.mock(PrefillEndpoint.class);
        Mockito.when(endpoint.realWaitTimeMs()).thenReturn(100L);
        Mockito.when(endpoint.batcherWaitMs()).thenReturn(200L);
        BalanceContext context = buildContext(1000, "41", new FlexlbConfig());

        assertEquals(300L, strategy.estimatedQueueWaitMs(endpoint, context));
    }

    @Test
    void directFallbackDoesNotCountBatcherQueueWait() {
        PrefillEndpoint endpoint = Mockito.mock(PrefillEndpoint.class);
        Mockito.when(endpoint.realWaitTimeMs()).thenReturn(100L);
        Mockito.when(endpoint.batcherWaitMs()).thenReturn(200L);
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(new DirectSchedulerConfig());
        config.setDispatcher(new NonBatchDispatcherConfig());
        BalanceContext context = buildContext(1000, "44", config);

        assertEquals(100L, strategy.estimatedQueueWaitMs(endpoint, context));
    }

    @Test
    void retryExcludesRejectedCacheLeaderWhenAnotherWorkerIsAvailable() {
        FlexlbConfig config = cacheAffinityConfig(1000, 0);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 0);
        stubCacheMatches(Map.of("10.0.0.2:8080", 3));
        BalanceContext context = buildContext(1000, "5", config);
        context.setExcludedPrefillIpPort("10.0.0.2:8080");

        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void retryRetainsExcludedWorkerWhenItIsTheOnlyEligibleWorker() {
        FlexlbConfig config = cacheAffinityConfig(1000, 0);
        addWorker("10.0.0.1", 0);
        BalanceContext context = buildContext(1000, "6", config);
        context.setExcludedPrefillIpPort("10.0.0.1:8080");

        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void usesRequestBlockSizeForPageRrCacheBenefit() {
        FlexlbConfig config = cacheAffinityConfig(100, 5);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 800);
        stubCacheMatches(Map.of("10.0.0.2:8080", 1));
        BalanceContext context = buildContext(4096, "7", config);
        context.getRequest().setCacheKeyBlockSize(1024L);

        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
        assertEquals(1024, result.getDebugInfo().getHitCacheLen());
    }

    @Test
    void discountsP2pOnlyMatchesAndRecordsKvcmDecisionAndTaskMetadata() {
        FlexlbConfig config = cacheAffinityConfig(1000, 0);
        config.getRouter().getRoles().getPrefill().getCacheAffinity()
                .setP2pHitDiscount(0.25);
        WorkerStatus coldWorker = addWorker("10.0.0.1", 0);
        WorkerStatus p2pWorker = addWorker("10.0.0.2", 100);
        Mockito.when(cacheAwareService.findMatchingEngines(any(CacheMatchQuery.class)))
                .thenReturn(new CacheMatchResult(
                        Map.of(p2pWorker.getIpPort(), new HostCacheMatch(0, 4, 4)),
                        CacheMatchSource.KVCM,
                        25L,
                        256L));

        BalanceContext context = buildContext(1000, "71", config);
        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals(p2pWorker.getIp(), result.getServerIp());
        assertEquals(256L, result.getDebugInfo().getHitCacheLen(),
                "four P2P-only blocks receive 25% routing credit");
        var decision = context.getShortestTtftDecisionByRole().get(RoleType.PREFILL);
        assertEquals(0.25, decision.p2pHitDiscount());
        var workerDecision = decision.workers().stream()
                .filter(worker -> worker.ip().equals(p2pWorker.getIp()))
                .findFirst()
                .orElseThrow();
        assertEquals(0L, workerDecision.requestLocalMatchTokens());
        assertEquals(1000L, workerDecision.requestP2pTotalMatchTokens());
        assertTrue(workerDecision.selected());
        TaskInfo task = p2pWorker.getLocalTaskMap().get("71");
        assertTrue(task.isKvcmMatchAvailable());
        assertEquals(1000L, task.getKvcmP2pTotalMatchTokens());
        assertFalse(coldWorker.getLocalTaskMap().containsKey("71"));
    }

    @Test
    void outstandingGuardSkipsCacheLeaderAndRecordsReason() {
        FlexlbConfig config = cacheAffinityConfig(1000, 0);
        config.getRouter().getRoles().getPrefill().getCacheAffinity()
                .setMaxOutstandingUncachedTokens(1000L);
        WorkerStatus cacheLeader = addWorker("10.0.0.1", 0);
        WorkerStatus eligibleWorker = addWorker("10.0.0.2", 0);
        TaskInfo existing = new TaskInfo();
        existing.setRequestId("700");
        existing.setInputLength(800L);
        existing.setPrefixLength(0L);
        cacheLeader.putLocalTask("700", existing);
        stubCacheMatches(Map.of(cacheLeader.getIpPort(), 3));

        BalanceContext context = buildContext(1000, "72", config);
        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals(eligibleWorker.getIp(), result.getServerIp());
        assertEquals(StrategySelectionReason.SHORTEST_TTFT_OUTSTANDING_GUARD.name(),
                context.getSelectionReasonByRole().get(RoleType.PREFILL));
        var decision = context.getShortestTtftDecisionByRole().get(RoleType.PREFILL);
        assertFalse(decision.cacheAffinityDecision().cacheLeaderOutstandingEligible());
        assertFalse(cacheLeader.getLocalTaskMap().containsKey("72"));
        assertTrue(eligibleWorker.getLocalTaskMap().containsKey("72"));
    }

    @Test
    void concurrentCacheLeaderClaimFallsBackToAnotherAffinityCandidate() {
        FlexlbConfig config = cacheAffinityConfig(250, 5);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 650);
        addWorker("10.0.0.3", 550);
        stubCacheMatches(Map.of(
                "10.0.0.2:8080", 3,
                "10.0.0.3:8080", 2));

        strategy = new ShortestTTFTStrategy(
                engineWorkerStatus,
                cacheAwareService,
                resourceMeasureFactory,
                engineHealthReporter) {
            @Override
            protected ScoredEndpoint selectBestEndpoint(
                    List<ScoredEndpoint> scoredEndpoints,
                    BalanceContext balanceContext,
                    RoleType roleType,
                    String group,
                    long seqLen,
                    FlexlbConfig flexlbConfig) {
                ScoredEndpoint cacheLeader = scoredEndpoints.stream()
                        .max(Comparator.comparingLong(ScoredEndpoint::hitCache))
                        .orElseThrow();
                cacheLeader.ep().getLastSelectedTime().incrementAndGet();
                return super.selectBestEndpoint(
                        scoredEndpoints,
                        balanceContext,
                        roleType,
                        group,
                        seqLen,
                        flexlbConfig);
            }
        };

        ServerStatus result = strategy.select(buildContext(1000, "8", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.3", result.getServerIp());
    }

    @Test
    void concurrentShortestCacheLeaderClaimPrefersEqualHitPeerOverColdWorker() {
        FlexlbConfig config = cacheAffinityConfig(250, 5);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 0);
        addWorker("10.0.0.3", 190);
        stubCacheMatches(Map.of(
                "10.0.0.1:8080", 1,
                "10.0.0.3:8080", 1));

        strategy = new ShortestTTFTStrategy(
                engineWorkerStatus,
                cacheAwareService,
                resourceMeasureFactory,
                engineHealthReporter) {
            @Override
            protected ScoredEndpoint selectBestEndpoint(
                    List<ScoredEndpoint> scoredEndpoints,
                    BalanceContext balanceContext,
                    RoleType roleType,
                    String group,
                    long seqLen,
                    FlexlbConfig flexlbConfig) {
                ScoredEndpoint cacheLeader = scoredEndpoints.stream()
                        .filter(scored -> "10.0.0.1".equals(scored.ep().getIp()))
                        .findFirst()
                        .orElseThrow();
                cacheLeader.ep().getLastSelectedTime().incrementAndGet();
                return super.selectBestEndpoint(
                        scoredEndpoints,
                        balanceContext,
                        roleType,
                        group,
                        seqLen,
                        flexlbConfig);
            }
        };

        ServerStatus result = strategy.select(buildContext(1000, "9", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.3", result.getServerIp());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.3", "CACHE_AFFINITY_FALLBACK");
    }

    @Test
    void refreshesSelectionSnapshotsBeforeBaselineAfterAllAffinityClaimsFail() {
        FlexlbConfig config = cacheAffinityConfig(0, 5);
        useFixedCandidatePool(config, 2);
        addWorker("10.0.0.1", 0);
        addWorker("10.0.0.2", 0);
        stubCacheMatches(Map.of("10.0.0.1:8080", 3));

        PrefillEndpoint cacheLeader = endpointRegistry.getPrefill("10.0.0.1:8080");
        PrefillEndpoint coldPeer = endpointRegistry.getPrefill("10.0.0.2:8080");
        cacheLeader.getLastSelectedTime().set(1000L);
        coldPeer.getLastSelectedTime().set(2000L);
        AtomicInteger claimRounds = new AtomicInteger();

        strategy = new ShortestTTFTStrategy(
                engineWorkerStatus,
                cacheAwareService,
                resourceMeasureFactory,
                engineHealthReporter) {
            @Override
            protected ScoredEndpoint selectFirstWithoutConcurrentConflict(
                    List<ScoredEndpoint> selectionOrder) {
                if (claimRounds.getAndIncrement() == 0) {
                    // Simulate concurrent claims invalidating every selection snapshot. The
                    // cold peer is now the baseline LRU, so only a refreshed snapshot can claim it.
                    cacheLeader.getLastSelectedTime().set(4000L);
                    coldPeer.getLastSelectedTime().set(3000L);
                }
                return super.selectFirstWithoutConcurrentConflict(selectionOrder);
            }
        };

        ServerStatus result = strategy.select(buildContext(1000, "10", config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
        assertEquals(2, claimRounds.get());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.2", "CACHE_AFFINITY_FALLBACK");
    }

    private FlexlbConfig cacheAffinityConfig(long maxExtraTtftMs, double minHitRate) {
        FlexlbConfig config = new FlexlbConfig();
        RoutingConfig.CacheAffinityConfig affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(maxExtraTtftMs);
        affinity.setMinPrefixHitPercent(minHitRate);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        return config;
    }

    private void stubCacheMatches(Map<String, Integer> matches) {
        Mockito.when(cacheAwareService.findMatchingEngines(any(CacheMatchQuery.class)))
                .thenAnswer(invocation -> {
                    CacheMatchQuery query = invocation.getArgument(0);
                    long blockSize = query.blockSize() > 0L ? query.blockSize() : 256L;
                    return new CacheMatchResult(
                            HostCacheMatch.fromLocalMatches(matches),
                            CacheMatchSource.LOCAL_SYNC,
                            0L,
                            blockSize);
                });
    }

    private WorkerStatus addWorker(String ip, long estimatedWaitMs) {
        WorkerStatus worker = createWorker(ip, estimatedWaitMs);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put(ip + ":8080", worker);
        return worker;
    }

    private WorkerStatus createWorker(String ip, long estimatedWaitMs) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setGrpcPort(8081);
        worker.setAlive(true);
        worker.setRole(RoleType.PREFILL);
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10000);
        cacheStatus.setBlockSize(256);
        worker.setCacheStatus(cacheStatus);
        worker.setRunningTaskList(new HashMap<>());

        PrefillEndpoint endpoint = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ip + ":8080", worker);
        if (estimatedWaitMs > 0) {
            long batchId = 900000L + ip.hashCode();
            endpoint.commitBatch(
                    batchId,
                    estimatedWaitMs,
                    List.of(batchItem(String.valueOf(batchId), estimatedWaitMs)));
        }
        return worker;
    }

    private BatchItem batchItem(String requestId, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());
        return new BatchItem(context, null, null, null, null, null, null, 0);
    }

    private BalanceContext buildContext(long seqLen, String requestId, FlexlbConfig config) {
        Request request = new Request();
        request.setSeqLen(seqLen);
        request.setRequestId(requestId);
        request.setBlockCacheKeys(new ArrayList<>(List.of(1L, 2L, 3L, 4L)));
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        return context;
    }

    private static void useFixedCandidatePool(FlexlbConfig config, int workers) {
        RoutingConfig.FixedCandidatePoolConfig pool = new RoutingConfig.FixedCandidatePoolConfig();
        pool.setWorkers(workers);
        RoutingConfig.LeastRecentlyUsedInPoolConfig choice =
                new RoutingConfig.LeastRecentlyUsedInPoolConfig();
        choice.setPool(pool);
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter().getRoles()
                        .getPrefill().getSelector();
        selector.setCandidateChoice(choice);
    }
}
