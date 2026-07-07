package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
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

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;

class ShortestTTFTStrategyTest {

    private EngineWorkerStatus engineWorkerStatus;
    private CacheAwareService cacheAwareService;
    private ResourceMeasureFactory resourceMeasureFactory;
    private EngineHealthReporter engineHealthReporter;
    private FlexlbBatchScheduler batchScheduler;
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
        batchScheduler = Mockito.mock(FlexlbBatchScheduler.class);

        // Create registry first to break circular dependency
        endpointRegistry = new EndpointRegistry(configService, batchScheduler, Mockito.mock(BatchSchedulerReporter.class));
        engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), endpointRegistry);

        PrefillResourceMeasure prefillResourceMeasure = Mockito.mock(PrefillResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(any())).thenReturn(prefillResourceMeasure);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any())).thenReturn(new HashMap<>());

        strategy = new ShortestTTFTStrategy(
                engineWorkerStatus, cacheAwareService, resourceMeasureFactory,
                engineHealthReporter);
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
    void candidatePoolFixedSizeOneShortCircuits() {
        FlexlbConfig config = new FlexlbConfig();
        config.setShortestTtftCandidatePoolMode("FIXED");
        config.setShortestTtftCandidatePoolSize(1);

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
        config.setShortestTtftCandidatePoolMode("RATIO");
        config.setShortestTtftCandidatePoolRatio(1.0);

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
        config.setShortestTtftCandidatePoolMode("RATIO");
        config.setShortestTtftCandidatePoolMinSize(0);
        config.setShortestTtftCandidatePoolRatio(0.3);

        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));

        ServerStatus result = strategy.select(buildContext(500, 1L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void candidatePoolRatioMode() {
        FlexlbConfig config = new FlexlbConfig();
        config.setShortestTtftCandidatePoolMode("RATIO");
        config.setShortestTtftCandidatePoolRatio(0.3);
        config.setShortestTtftCandidatePoolMinSize(1);

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
        PrefillEndpoint ep = endpointRegistry.ensurePrefillEndpoint(ipPort, w);
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
        if (hitCache > 0) {
            org.flexlb.dao.loadbalance.DebugInfo di = new org.flexlb.dao.loadbalance.DebugInfo();
            di.setHitCacheLen(hitCache);
            org.flexlb.dao.loadbalance.ServerStatus ss = new org.flexlb.dao.loadbalance.ServerStatus();
            ss.setDebugInfo(di);
            return new BatchItem(ctx, null, null, ss, null, null, null, 0, 0);
        }
        return new BatchItem(ctx, null, null, null, null, null, null, 0, 0);
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
}
