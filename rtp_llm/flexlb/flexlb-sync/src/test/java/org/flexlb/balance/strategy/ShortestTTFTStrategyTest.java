package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.session.SessionPlacementStore;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;

class ShortestTTFTStrategyTest {

    private EngineWorkerStatus engineWorkerStatus;
    private CacheAwareService cacheAwareService;
    private ResourceMeasureFactory resourceMeasureFactory;
    private EngineHealthReporter engineHealthReporter;
    private PriorityScheduler batchScheduler;
    private EndpointRegistry endpointRegistry;
    private ShortestTTFTStrategy strategy;
    private FlexlbConfig endpointConfig;

    @BeforeEach
    void setUp() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        ConfigService configService = Mockito.mock(ConfigService.class);
        endpointConfig = new FlexlbConfig();
        Mockito.when(configService.loadBalanceConfig()).thenReturn(endpointConfig);
        cacheAwareService = Mockito.mock(CacheAwareService.class);
        resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        batchScheduler = Mockito.mock(PriorityScheduler.class);

        // Create registry first to break circular dependency
        endpointRegistry = new EndpointRegistry(configService, () -> batchScheduler,
                Mockito.mock(BatchSchedulerReporter.class));
        engineWorkerStatus = new EngineWorkerStatus(endpointRegistry);

        PrefillResourceMeasure prefillResourceMeasure = Mockito.mock(PrefillResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(any())).thenReturn(prefillResourceMeasure);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any())).thenReturn(new HashMap<>());

        strategy = new ShortestTTFTStrategy(
                engineWorkerStatus, cacheAwareService, resourceMeasureFactory,
                engineHealthReporter, new SessionPlacementStore());
    }

    // ==================== Test Cases ====================

    @Test
    void selectsWorkerWithLowestTTFT() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        // Worker 1: TTFT = estimateMs(1000,0)=1000 + wait 100 = 1100
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 100));
        // Worker 2: TTFT = estimateMs(1000,0)=1000 + wait 50 = 1050  (lower)
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 50));

        ServerStatus result = strategy.select(buildContext(1000, 1L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void reportsSelectedEstimatesWithNonBatchDeliveryMode() {
        endpointConfig.setDispatcher(new NonBatchDispatcherConfig());
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        PrefillEndpoint selectedEndpoint = Mockito.spy(
                endpointRegistry.getPrefill("10.0.0.1:8080"));
        Mockito.doReturn(250L).when(selectedEndpoint).realWaitTimeMs();
        endpointRegistry.getPrefillEndpoints().put(
                "10.0.0.1:8080", selectedEndpoint);

        ServerStatus result = strategy.select(
                buildContext(1_000, 10_011L, endpointConfig), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        Mockito.verify(engineHealthReporter).reportPrefillSelectedEstimates(
                RoleType.PREFILL, "10.0.0.1", "NON_BATCH", 1_250L, 1_000L);
    }

    @Test
    void reportsSelectedEstimatesWithBatchDeliveryMode() {
        endpointConfig.setDispatcher(new BatchDispatcherConfig());
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        PrefillEndpoint selectedEndpoint = Mockito.spy(
                endpointRegistry.getPrefill("10.0.0.1:8080"));
        Mockito.doReturn(250L).when(selectedEndpoint).realWaitTimeMs();
        endpointRegistry.getPrefillEndpoints().put(
                "10.0.0.1:8080", selectedEndpoint);

        ServerStatus result = strategy.select(
                buildContext(1_000, 10_012L, endpointConfig), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        Mockito.verify(engineHealthReporter).reportPrefillSelectedEstimates(
                RoleType.PREFILL, "10.0.0.1", "BATCH", 1_550L, 1_000L);
    }

    @Test
    void unavailableWaitEstimateCannotWinBySignedOverflow() {
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));

        PrefillEndpoint first = Mockito.spy(
                endpointRegistry.getPrefill("10.0.0.1:8080"));
        Mockito.doReturn(Long.MAX_VALUE).when(first).realWaitTimeMs();
        endpointRegistry.getPrefillEndpoints().put("10.0.0.1:8080", first);

        ServerStatus mixed = strategy.select(
                buildContext(1_000, 8_001L), RoleType.PREFILL, null);

        assertTrue(mixed.isSuccess());
        assertEquals("10.0.0.2", mixed.getServerIp(),
                "an unavailable wait sentinel must not wrap into the minimum TTFT");

        PrefillEndpoint second = Mockito.spy(
                endpointRegistry.getPrefill("10.0.0.2:8080"));
        Mockito.doReturn(Long.MAX_VALUE).when(second).realWaitTimeMs();
        endpointRegistry.getPrefillEndpoints().put("10.0.0.2:8080", second);

        ServerStatus allUnavailable = strategy.select(
                buildContext(1_000, 8_002L), RoleType.PREFILL, null);

        assertFalse(allUnavailable.isSuccess(),
                "selection must retry elsewhere when no coherent wait snapshot exists");
    }

    @Test
    void candidatePoolFixedSizeOneShortCircuits() {
        FlexlbConfig config = new FlexlbConfig();
        useFixedCandidatePool(config, 1);

        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        // TTFTs: 600, 510, 700 — worker 2 has the lowest
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 100));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 10));
        prefillMap.put("10.0.0.3:8080", createWorker("10.0.0.3", 200));

        // candidateCount = min(1, 3) = 1 → only lowest-TTFT worker in pool, short-circuit
        ServerStatus result = strategy.select(buildContext(500, 1L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void casFairnessSpreadsAcrossSimilarWorkers() {
        // ratio=1.0 so candidateCount = max(1, floor(2*1.0)) = 2 — both workers in pool
        FlexlbConfig config = new FlexlbConfig();
        useRatioCandidatePool(config, 1.0, 1);

        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        // Both have same TTFT (estimateMs(500,0)=500 + wait 0 = 500)
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));

        // Same TTFT, but different lastSelectedTime — CAS should pick the least-recently-selected
        PrefillEndpoint ep1 = endpointRegistry.getPrefill("10.0.0.1:8080");
        ep1.getLastSelectedTime().set(1000);  // earlier → less recently used
        PrefillEndpoint ep2 = endpointRegistry.getPrefill("10.0.0.2:8080");
        ep2.getLastSelectedTime().set(2000);  // later  → more recently used

        ServerStatus result = strategy.select(buildContext(500, 1L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        // CAS fairness selects the worker with the earlier lastSelectedTime
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void cacheHitReducesTTFT() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        // Both have wait=0; TTFT is driven by estimateMs alone
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));

        // Give worker 2 a cache hit: 3 blocks * 256 = 768 hit tokens
        // estimateMs(1000, 768) = (1000-768) + 0.3*768 = 232 + 230 = 462  <  estimateMs(1000, 0) = 1000
        Map<String, Integer> cacheResults = new HashMap<>();
        cacheResults.put("10.0.0.2:8080", 3);
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any())).thenReturn(cacheResults);

        ServerStatus result = strategy.select(buildContext(1000, 1L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void rollBackReleasesInflight() {
        PrefillEndpoint mockEp = Mockito.mock(PrefillEndpoint.class);
        long requestId = 42L;

        strategy.rollBack(mockEp, requestId);

        Mockito.verify(mockEp).releaseBatch(requestId);
    }

    @Test
    void casClaimAlwaysAdvancesSelectionTimestamp() {
        PrefillEndpoint endpoint = Mockito.mock(PrefillEndpoint.class);
        long snapshot = System.nanoTime() / 1000 + 1_000_000L;
        AtomicLong lastSelectedTime = new AtomicLong(snapshot);
        Mockito.when(endpoint.getLastSelectedTime()).thenReturn(lastSelectedTime);
        ShortestTTFTStrategy.ScoredEndpoint scored =
                new ShortestTTFTStrategy.ScoredEndpoint(endpoint, 0L, 0L, 0L, snapshot);

        ShortestTTFTStrategy.ScoredEndpoint selected =
                strategy.selectFirstWithoutConcurrentConflict(List.of(scored));

        assertSame(scored, selected);
        assertEquals(snapshot + 1L, lastSelectedTime.get());
    }

    @Test
    void directPathCommitsOnlyMarginalPrefillTime() {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduler(new DirectSchedulerConfig());
        config.setDispatcher(new NonBatchDispatcherConfig());
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 1000));
        PrefillEndpoint endpoint = endpointRegistry.getPrefill("10.0.0.1:8080");
        BalanceContext context = buildContext(500, 43L, config);

        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals(2, endpoint.getInflightBatchCount());
        assertTrue(endpoint.realWaitTimeMs() < 2000L);
    }

    @Test
    void noAvailableWorkersReturnsError() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();

        ServerStatus result = strategy.select(buildContext(500, 1L), RoleType.PREFILL, null);

        assertFalse(result.isSuccess());
    }

    @Test
    void candidatePoolMinSizeZeroDoesNotCrash() {
        // Config: minSize=0, ratio=0.3, 1 worker → resolveCandidateCount returns max(1, max(0, 0)) = 1
        // Without the floor of 1 this would yield 0 → empty candidate pool → NoSuchElementException
        FlexlbConfig config = new FlexlbConfig();
        useRatioCandidatePool(config, 0.3, 0);

        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));

        ServerStatus result = strategy.select(buildContext(500, 1L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void candidatePoolRatioMode() {
        FlexlbConfig config = new FlexlbConfig();
        useRatioCandidatePool(config, 0.3, 1);

        // ---- Scenario 1: 5 workers → candidateCount = max(1, floor(5*0.3)) = 1 ----
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.clear();
        for (int i = 1; i <= 5; i++) {
            String ip = "10.0.0." + i;
            // TTFTs: 500, 600, 700, 800, 900
            prefillMap.put(ip + ":8080", createWorker(ip, (i - 1) * 100));
        }
        // Make the lowest-TTFT worker (10.0.0.1) recently selected
        endpointRegistry.getPrefill("10.0.0.1:8080").getLastSelectedTime().set(1_000_000L);
        // Make the 2nd-lowest (10.0.0.2) least recently selected
        endpointRegistry.getPrefill("10.0.0.2:8080").getLastSelectedTime().set(0L);

        ServerStatus result1 = strategy.select(buildContext(500, 1L, config), RoleType.PREFILL, null);
        assertTrue(result1.isSuccess());
        // candidateCount=1 → only 10.0.0.1 in pool → short-circuit selects it
        // despite being recently selected (if pool were larger, CAS would pick 10.0.0.2)
        assertEquals("10.0.0.1", result1.getServerIp());

        // ---- Scenario 2: 10 workers → candidateCount = max(1, floor(10*0.3)) = 3 ----
        prefillMap.clear();
        for (int i = 1; i <= 10; i++) {
            String ip = "10.0.1." + i;
            // TTFTs: 500, 510, 520, 530, …, 590
            prefillMap.put(ip + ":8080", createWorker(ip, (i - 1) * 10));
        }
        // Worker 1 (lowest TTFT): most recently selected
        endpointRegistry.getPrefill("10.0.1.1:8080").getLastSelectedTime().set(1_000_000L);
        // Worker 2 (2nd lowest): slightly less recent
        endpointRegistry.getPrefill("10.0.1.2:8080").getLastSelectedTime().set(999_999L);
        // Worker 3 (3rd lowest): least recently selected (oldest)
        endpointRegistry.getPrefill("10.0.1.3:8080").getLastSelectedTime().set(0L);

        ServerStatus result2 = strategy.select(buildContext(500, 2L, config), RoleType.PREFILL, null);
        assertTrue(result2.isSuccess());
        // candidateCount=3 → workers 1, 2, 3 are in the pool
        // CAS fairness picks worker 3 (oldest lastSelectedTime=0)
        // If candidateCount were 2 → worker 2 would be selected
        // If candidateCount were 1 → worker 1 would be selected
        assertEquals("10.0.1.3", result2.getServerIp());
    }

    @Test
    void reportsCandidateMaxAndSelectedRoutingCacheMatchTokens() {
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 5000));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));

        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(Map.of(
                        "10.0.0.1:8080", 4,
                        "10.0.0.2:8080", 1));

        BalanceContext context = buildContext(4096, 1L);
        context.getRequest().setCacheKeyBlockSize(1024L);

        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
        Mockito.verify(engineHealthReporter).reportRoutingSelectedCacheMatchMetrics(
                RoleType.PREFILL, 1024L, 4096L);
        Mockito.verify(engineHealthReporter).reportRoutingCandidateMaxCacheMatchMetrics(
                RoleType.PREFILL, 4096L);
    }

    // ==================== Helpers (mirrors CostBasedPrefillStrategyTest) ====================

    private WorkerStatus createWorker(String ip, long estimatedWaitMs) {
        WorkerStatus w = new WorkerStatus();
        w.setIp(ip);
        w.setPort(8080);
        w.setAlive(true);
        w.setRole(RoleType.PREFILL);
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10000);
        cacheStatus.setBlockSize(256);
        w.setCacheStatus(cacheStatus);
        w.setRunningTaskList(new HashMap<>());

        String ipPort = ip + ":8080";
        w.setGrpcPort(8081);
        PrefillEndpoint ep = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ipPort, w);
        if (estimatedWaitMs > 0) {
            ep.commitBatch(900000L + ip.hashCode(), estimatedWaitMs,
                    List.of(batchItem(900000L + ip.hashCode(), estimatedWaitMs, 0)));
        }
        return w;
    }

    private BatchItem batchItem(long requestId, long seqLen, long hitCache) {
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        ctx.setConfig(SchedulingTestConfig.batchConfig());
        if (hitCache > 0) {
            org.flexlb.dao.loadbalance.DebugInfo di = new org.flexlb.dao.loadbalance.DebugInfo();
            di.setHitCacheLen(hitCache);
            org.flexlb.dao.loadbalance.ServerStatus ss = new org.flexlb.dao.loadbalance.ServerStatus();
            ss.setDebugInfo(di);
            return new BatchItem(ctx, null, null, ss, null, null, null, 0);
        }
        return new BatchItem(ctx, null, null, null, null, null, null, 0);
    }

    private BalanceContext buildContext(long seqLen, long requestId) {
        return buildContext(seqLen, requestId, new FlexlbConfig());
    }

    private BalanceContext buildContext(long seqLen, long requestId, FlexlbConfig config) {
        Request req = new Request();
        req.setSeqLen(seqLen);
        req.setRequestId(requestId);
        req.setBlockCacheKeys(new ArrayList<>(List.of(1L, 2L)));
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        ctx.setConfig(config);
        return ctx;
    }

    private static void useFixedCandidatePool(FlexlbConfig config, int workers) {
        RoutingConfig.FixedCandidatePoolConfig pool = new RoutingConfig.FixedCandidatePoolConfig();
        pool.setWorkers(workers);
        useLruPool(config, pool);
    }

    private static void useRatioCandidatePool(
            FlexlbConfig config, double ratio, int minimumWorkers) {
        RoutingConfig.RatioCandidatePoolConfig pool = new RoutingConfig.RatioCandidatePoolConfig();
        pool.setRatio(ratio);
        pool.setMinimumWorkers(minimumWorkers);
        useLruPool(config, pool);
    }

    private static void useLruPool(
            FlexlbConfig config, RoutingConfig.CandidatePoolConfig pool) {
        RoutingConfig.LeastRecentlyUsedInPoolConfig choice =
                new RoutingConfig.LeastRecentlyUsedInPoolConfig();
        choice.setPool(pool);
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter().getRoles()
                        .getPrefill().getSelector();
        selector.setCandidateChoice(choice);
    }
}
