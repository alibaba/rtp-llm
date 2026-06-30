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

class CostBasedPrefillStrategyTest {

    private EngineWorkerStatus engineWorkerStatus;
    private CacheAwareService cacheAwareService;
    private ResourceMeasureFactory resourceMeasureFactory;
    private EngineHealthReporter engineHealthReporter;
    private FlexlbBatchScheduler batchScheduler;
    private EndpointRegistry endpointRegistry;
    private CostBasedPrefillStrategy strategy;

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

        strategy = new CostBasedPrefillStrategy(
                engineWorkerStatus, cacheAwareService, resourceMeasureFactory,
                engineHealthReporter, endpointRegistry);
    }

    /** Helper: register PrefillEndpoints for all entries in the given worker map. */
    private void registerPrefillEndpoints(Map<String, WorkerStatus> workerMap) {
        for (Map.Entry<String, WorkerStatus> entry : workerMap.entrySet()) {
            WorkerStatus ws = entry.getValue();
            ws.setGrpcPort(9090);
            endpointRegistry.ensurePrefillEndpoint(entry.getKey(), ws);
        }
    }

    @Test
    void selectsWorkerWithLowestCostScore() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 100));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 50));

        ServerStatus result = strategy.select(buildContext(1000, 1L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void batcherQueueReducesWaitCost() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        WorkerStatus w1 = createWorker("10.0.0.1", 0);
        WorkerStatus w2 = createWorker("10.0.0.2", 0);
        prefillMap.put("10.0.0.1:8080", w1);
        prefillMap.put("10.0.0.2:8080", w2);

        ServerStatus result = strategy.select(buildContext(500, 2L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
    }

    @Test
    void deltaPrefillCostFavorsCacheHitWorker() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));

        Map<String, Integer> cacheResults = new HashMap<>();
        cacheResults.put("10.0.0.2:8080", 3); // 3 blocks * 256 = 768 tokens
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any())).thenReturn(cacheResults);

        ServerStatus result = strategy.select(buildContext(1000, 3L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void sloRiskFilterExcludesOverloadedWorker() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 2000));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 10));

        ServerStatus result = strategy.select(buildContext(500, 4L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void allFilteredFallsBackToLeastLoaded() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 5000));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 3000));

        ServerStatus result = strategy.select(buildContext(500, 5L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void hotspotFilterExcludesBatcherOverloadedWorker() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        prefillMap.put("10.0.0.3:8080", createWorker("10.0.0.3", 0));

        ServerStatus result = strategy.select(buildContext(500, 6L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertNotEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void imbalanceFilterExcludesOverloadedEngineQueue() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.clear();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 1000));
        for (int i = 2; i <= 10; i++) {
            String ip = "10.0.0." + i;
            prefillMap.put(ip + ":8080", createWorker(ip, 10));
        }

        FlexlbConfig config = new FlexlbConfig();
        config.setCostSloMs(50000L);

        ServerStatus result = strategy.select(buildContext(500, 7L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertNotEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void noAvailableWorkersReturnsError() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();

        ServerStatus result = strategy.select(buildContext(500, 8L), RoleType.PREFILL, null);

        assertFalse(result.isSuccess());
    }

    @Test
    void rollBackDoesNotThrow() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        WorkerStatus w = createWorker("10.0.0.1", 0);
        prefillMap.put("10.0.0.1:8080", w);

        ServerStatus result = strategy.select(buildContext(500, 9L), RoleType.PREFILL, null);
        assertTrue(result.isSuccess());

        assertDoesNotThrow(() -> strategy.rollBack("10.0.0.1:8080", 9L));
    }

    @Test
    void endpointWaitMsFavorsEndpointWithLowerEstimate() {
        WorkerStatus w1 = createWorker("10.0.0.1", 0);
        WorkerStatus w2 = createWorker("10.0.0.2", 0);
        w1.setGrpcPort(8081);
        w2.setGrpcPort(8081);

        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", w1);
        prefillMap.put("10.0.0.2:8080", w2);

        PrefillEndpoint ep1 = endpointRegistry.ensurePrefillEndpoint("10.0.0.1:8080", w1);
        endpointRegistry.ensurePrefillEndpoint("10.0.0.2:8080", w2);
        ep1.commitBatch(1L, 4000, List.of(batchItem(1L, 1000, 0)));

        ServerStatus result = strategy.select(buildContext(500, 10L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void predictorUsesPolynomialFormula() {
        PrefillTimePredictor predictor = new PrefillTimePredictor(10, 0.5, 0.001, 0.0005, 0.2, 5);

        // Single request: n=1000, p=200 → c=800, bs=1
        // = 10 + 0.5*800 + (0.001*640000 + 0.0005*160000) + 0.2*200 + 5*1
        // = 10 + 400 + (640 + 80) + 40 + 5 = 1175
        long single = predictor.predictBatchMs(List.of(batchItem(0, 1000, 200)));
        assertEquals(1175, single);

        // Batch of 2: req1=(1000,200) req2=(500,100)
        // c1=800, p1=200, c2=400, p2=100
        // Σc=1200, Σ(640+80, 160+20)=900, Σp=300, bs=2
        // = 10 + 0.5*1200 + 900 + 0.2*300 + 5*2 = 1580
        long batch = predictor.predictBatchMs(List.of(
                batchItem(0, 1000, 200),
                batchItem(1, 500, 100)));
        assertEquals(1580, batch);

        assertEquals(0, predictor.predictBatchMs(List.of()));
    }

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
        // For prediction, hitCache comes from prefill.debugInfo.  Use null prefill → 0,
        // but the caller's hitCache parameter is what matters for prediction — we set it
        // via the constructor as a convenience; the predictor will call item.hitCache()
        // which reads prefill.debugInfo.hitCacheLen, so we must build a real ServerStatus.
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
        FlexlbConfig config = new FlexlbConfig();
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        return buildContext(seqLen, requestId, config);
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
