package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.TestCapacityAdmission;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;

class CostBasedPrefillStrategyTest {

    private EngineWorkerStatus engineWorkerStatus;
    private CacheAwareService cacheAwareService;
    private ResourceMeasureFactory resourceMeasureFactory;
    private EngineHealthReporter engineHealthReporter;
    private PrefillResourceMeasure prefillResourceMeasure;
    private PriorityScheduler batchScheduler;
    private EndpointRegistry endpointRegistry;
    private CostBasedPrefillStrategy strategy;
    private FlexlbConfig endpointConfig;

    @BeforeEach
    void setUp() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        endpointConfig = new FlexlbConfig();
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(endpointConfig);
        cacheAwareService = Mockito.mock(CacheAwareService.class);
        resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        batchScheduler = Mockito.mock(PriorityScheduler.class);

        // Create registry first to break circular dependency
        endpointRegistry = new EndpointRegistry(configService, () -> batchScheduler,
                Mockito.mock(BatchSchedulerReporter.class));
        engineWorkerStatus = new EngineWorkerStatus(endpointRegistry);

        prefillResourceMeasure = Mockito.mock(PrefillResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(any())).thenReturn(prefillResourceMeasure);
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any())).thenReturn(new HashMap<>());

        strategy = new CostBasedPrefillStrategy(
                engineWorkerStatus, cacheAwareService, resourceMeasureFactory,
                engineHealthReporter);
    }

    @AfterEach
    void tearDown() {
        endpointRegistry.close();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
    }

    @Test
    void selectsWorkerWithLowestCostScore() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 500));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 50));

        ServerStatus result = strategy.select(buildContext(1000, 1L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
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
                buildContext(1_000, 7_001L), RoleType.PREFILL, null);

        assertTrue(mixed.isSuccess());
        assertEquals("10.0.0.2", mixed.getServerIp(),
                "an unavailable wait sentinel must not wrap into the minimum score");

        PrefillEndpoint second = Mockito.spy(
                endpointRegistry.getPrefill("10.0.0.2:8080"));
        Mockito.doReturn(Long.MAX_VALUE).when(second).realWaitTimeMs();
        endpointRegistry.getPrefillEndpoints().put("10.0.0.2:8080", second);

        ServerStatus allUnavailable = strategy.select(
                buildContext(1_000, 7_002L), RoleType.PREFILL, null);

        assertFalse(allUnavailable.isSuccess(),
                "selection must retry elsewhere when no coherent wait snapshot exists");
    }

    @Test
    void scoreTieRandomDisabledSelectsExactMinimum() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 500));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 50));

        FlexlbConfig config = new FlexlbConfig();
        useBestOnly(config);

        ServerStatus result = strategy.select(buildContext(1000, 11L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void scoreTieRandomDisabledWithExactTieStillSelects() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));

        FlexlbConfig config = new FlexlbConfig();
        useBestOnly(config);

        ServerStatus result = strategy.select(buildContext(1000, 12L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
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
    void cacheAffinityDisabledKeepsCostBasedSelection() {
        setFormula(endpointConfig, "sum(computeTokens) + 2*sum(hitCacheTokens)");
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(Map.of("10.0.0.2:8080", 1));

        FlexlbConfig config = affinityConfig(10_000, 0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(null);
        ServerStatus result = strategy.select(
                buildContext(1000, 301L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
        Mockito.verify(engineHealthReporter, Mockito.never())
                .reportCacheAffinityDecision(any(), any(), any());
    }

    @Test
    void cacheAffinitySelectsLongestPrefixInsideCostCap() {
        setFormula(endpointConfig, "sum(computeTokens) + 2*sum(hitCacheTokens)");
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        prefillMap.put("10.0.0.3:8080", createWorker("10.0.0.3", 0));
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(Map.of(
                        "10.0.0.2:8080", 1,
                        "10.0.0.3:8080", 2));

        ServerStatus result = strategy.select(
                buildContext(1000, 302L, affinityConfig(300, 5)),
                RoleType.PREFILL,
                null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp(),
                "The global leader is over cap, but the one-block candidate is admissible");
        assertEquals(256, result.getDebugInfo().getHitCacheLen());
        assertEquals(1556, result.getPrefillTime());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
    }

    @Test
    void cacheAffinityFallsBackToCostBasedWhenCacheCostIsOverCap() {
        setFormula(endpointConfig, "sum(computeTokens) + 2*sum(hitCacheTokens)");
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(Map.of("10.0.0.2:8080", 1));

        ServerStatus result = strategy.select(
                buildContext(1000, 303L, affinityConfig(255, 5)),
                RoleType.PREFILL,
                null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "OVER_CAP");
    }

    @Test
    void costBasedCacheAffinityUsesRequestBlockSizeForPageRr() {
        setFormula(endpointConfig, "sum(computeTokens) + 2*sum(hitCacheTokens)");
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(Map.of("10.0.0.2:8080", 1));
        BalanceContext context = buildContext(4096, 304L, affinityConfig(1024, 20));
        context.getRequest().setCacheKeyBlockSize(1024L);

        ServerStatus result = strategy.select(context, RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
        assertEquals(1024, result.getDebugInfo().getHitCacheLen());
    }

    @Test
    void costBasedHardFilterDoesNotReviveUnavailableCacheLeader() {
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        Mockito.when(prefillResourceMeasure.isResourceAvailable(any()))
                .thenAnswer(invocation -> !"10.0.0.2".equals(
                        ((PrefillEndpoint) invocation.getArgument(0)).getIp()));
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(Map.of("10.0.0.2:8080", 3));

        ServerStatus result = strategy.select(
                buildContext(1000, 305L, affinityConfig(10_000, 0)),
                RoleType.PREFILL,
                null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp());
        Mockito.verify(engineHealthReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "NO_CACHE_LEAD");
    }

    @Test
    void fullCacheHitKeepsFinalBlockAsComputeTokens() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));

        Map<String, Integer> cacheResults = new HashMap<>();
        cacheResults.put("10.0.0.1:8080", 4); // 4 blocks * 256 >= seqLen=1000
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any())).thenReturn(cacheResults);

        ServerStatus result = strategy.select(buildContext(1000, 31L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertNotNull(result.getDebugInfo());
        assertEquals(744, result.getDebugInfo().getHitCacheLen());
    }

    @Test
    void scoringPrefersLowerWaitWithoutSloFiltering() {
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
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 500));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 0));
        prefillMap.put("10.0.0.3:8080", createWorker("10.0.0.3", 0));

        ServerStatus result = strategy.select(buildContext(500, 6L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertNotEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void engineOnlyBacklogOverridesCacheHitAdvantageAtResourceFilter() {
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        for (int i = 1; i <= 4; i++) {
            String ip = "10.0.0." + i;
            prefillMap.put(ip + ":8080", createWorker(ip, 0));
        }

        PrefillEndpoint hot = (PrefillEndpoint) endpointRegistry.get(
                RoleType.PREFILL, "10.0.0.1:8080");
        Map<String, TaskInfo> engineOnlyTasks = new HashMap<>();
        for (long requestId = 1_000; requestId < 1_100; requestId++) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setBatchId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            engineOnlyTasks.put(String.valueOf(requestId), task);
        }
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(engineOnlyTasks);
        hot.onWorkerStatusUpdate(hot.getStatus(), response);

        FlexlbConfig resourceConfig = new FlexlbConfig();
        resourceConfig.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(64);
        ConfigService resourceConfigService = Mockito.mock(ConfigService.class);
        Mockito.when(resourceConfigService.loadBalanceConfig()).thenReturn(resourceConfig);
        PrefillResourceMeasure actualMeasure = new PrefillResourceMeasure(resourceConfigService);
        Mockito.when(resourceMeasureFactory.getMeasure(any())).thenReturn(actualMeasure);

        Map<String, Integer> cacheResults = new HashMap<>();
        cacheResults.put("10.0.0.1:8080", 3);
        Mockito.when(cacheAwareService.findMatchingEngines(anyList(), any(), any()))
                .thenReturn(cacheResults);

        ServerStatus result = strategy.select(
                buildContext(1_000, 61L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertNotEquals("10.0.0.1", result.getServerIp(),
                "Engine-only active tasks must make the hot worker unavailable even with a cache advantage");
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

        ServerStatus result = strategy.select(buildContext(500, 7L, config), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertNotEquals("10.0.0.1", result.getServerIp());
    }

    @Test
    void imbalanceFilterUsesOthersAverageExcludingSelf() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 350));
        prefillMap.put("10.0.0.2:8080", createWorker("10.0.0.2", 100));

        // Keep BOTH engines inside the RANDOM_WITHIN_TOLERANCE window
        // (constant 5s prefill estimate => scores 5350 vs 5100, tolerance
        // = 10% * 5100 = 510) so only the outlier filter — not score
        // ordering — can exclude the busy one. With the old self-inclusive
        // average the threshold was 3.0 * 225 = 675, so wait=350 could
        // mathematically NEVER be filtered with n=2 and 10.0.0.1 stayed
        // selectable. The leave-one-out baseline (others avg = 100) must
        // reject it: 350 > 3.0 * 100.
        FlexlbConfig config = new FlexlbConfig();
        setFormula(config, "5000");

        BalanceContext ctx = buildContext(500, 61L, config);
        for (int i = 0; i < 30; i++) {
            long requestId = 6_100L + i;
            ctx.getRequest().setRequestId(requestId);
            ServerStatus result = strategy.select(ctx, RoleType.PREFILL, null);
            assertTrue(result.isSuccess());
            assertNotEquals("10.0.0.1", result.getServerIp(),
                    "wait=350ms vs others-avg=100ms must be outlier-rejected");
            strategy.rollBack(
                    endpointRegistry.get(RoleType.PREFILL, result.getServerIp() + ":8080"),
                    requestId);
        }
    }

    @Test
    void singleCandidateSkipsOutlierRejection() {
        Map<String, WorkerStatus> prefillMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 100_000));

        // A lone engine with a huge wait: there are no "other" engines to
        // be an outlier against, so the relative outlier checks must be
        // skipped — rejecting the only candidate could only yield
        // NO_AVAILABLE_WORKER with nothing to gain.
        ServerStatus result = strategy.select(buildContext(500, 62L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.1", result.getServerIp(),
                "a lone engine has no 'others' to be an outlier against and must stay selectable");
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

        assertDoesNotThrow(() -> strategy.rollBack(
                endpointRegistry.get(RoleType.PREFILL, "10.0.0.1:8080"), 9L));
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

        PrefillEndpoint ep1 = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, "10.0.0.1:8080", w1);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, "10.0.0.2:8080", w2);
        TestCapacityAdmission.registerQueueBatchLifecycle(ep1, 1L, 4000, List.of(batchItem(1L, 1000, 0)));

        ServerStatus result = strategy.select(buildContext(500, 10L), RoleType.PREFILL, null);

        assertTrue(result.isSuccess());
        assertEquals("10.0.0.2", result.getServerIp());
    }

    @Test
    void predictorUsesFormula() {
        FormulaPredictor predictor = new FormulaPredictor(
                "10 + 0.5*sum(computeTokens)"
                        + " + 0.001*sum(computeTokens^2)"
                        + " + 0.0005*sum(computeTokens * hitCacheTokens)"
                        + " + 0.2*sum(hitCacheTokens)"
                        + " + 5*batchSize");

        // Single request: inputTokens=1000, hitCacheTokens=200, computeTokens=800, batchSize=1
        // = 10 + 0.5*800 + 0.001*640000 + 0.0005*160000 + 0.2*200 + 5*1
        // = 10 + 400 + 640 + 80 + 40 + 5 = 1175
        long single = (long) predictor.predictBatchMs(List.of(batchItem(0, 1000, 200)));
        assertEquals(1175, single);

        // Batch of 2: req1=(1000,200) req2=(500,100)
        // computeTokens=(800,400), hitCacheTokens=(200,100), batchSize=2
        // sum(computeTokens)=1200, sum(computeTokens^2)=800000,
        // sum(computeTokens * hitCacheTokens)=200000, sum(hitCacheTokens)=300
        // = 10 + 0.5*1200 + 0.001*800000 + 0.0005*200000 + 0.2*300 + 5*2
        // = 10 + 600 + 800 + 100 + 60 + 10 = 1580
        long batch = (long) predictor.predictBatchMs(List.of(
                batchItem(0, 1000, 200),
                batchItem(1, 500, 100)));
        assertEquals(1580, batch);

        assertEquals(0, (long) predictor.predictBatchMs(List.of()));
    }

    @Test
    void selectsPdFusionEndpointFromItsOwnRegistry() {
        String ipPort = "10.0.0.1:8080";
        WorkerStatus worker = createUnregisteredWorker("10.0.0.1");
        worker.setRole(RoleType.PDFUSION);
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().put(ipPort, worker);
        endpointRegistry.ensureEndpoint(RoleType.PDFUSION, ipPort, worker);

        ServerStatus result = strategy.select(
                buildContext(500, 41L), RoleType.PDFUSION, null);

        assertTrue(result.isSuccess());
        assertEquals(RoleType.PDFUSION, result.getRole());
    }

    @Test
    void candidateBufferGrowsWithRegistryAndRemainsReusable() {
        Map<String, WorkerStatus> prefillMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        prefillMap.put("10.0.0.1:8080", createWorker("10.0.0.1", 0));
        assertTrue(strategy.select(buildContext(500, 51L), RoleType.PREFILL, null).isSuccess());

        for (int i = 2; i <= 40; i++) {
            String ip = "10.0.1." + i;
            prefillMap.put(ip + ":8080", createWorker(ip, 0));
        }

        for (long requestId = 52; requestId < 72; requestId++) {
            assertTrue(strategy.select(buildContext(500, requestId), RoleType.PREFILL, null).isSuccess());
        }
    }

    private WorkerStatus createWorker(String ip, long estimatedWaitMs) {
        WorkerStatus w = createUnregisteredWorker(ip);

        String ipPort = ip + ":8080";
        w.setGrpcPort(8081);
        PrefillEndpoint ep = (PrefillEndpoint) endpointRegistry.ensureEndpoint(
                RoleType.PREFILL, ipPort, w);
        if (estimatedWaitMs > 0) {
            TestCapacityAdmission.registerQueueBatchLifecycle(ep, 900000L + ip.hashCode(), estimatedWaitMs,
                    List.of(batchItem(900000L + ip.hashCode(), estimatedWaitMs, 0)));
        }
        return w;
    }

    private WorkerStatus createUnregisteredWorker(String ip) {
        WorkerStatus w = new WorkerStatus();
        w.setIp(ip);
        w.setPort(8080);
        w.setGrpcPort(8081);
        w.setAlive(true);
        w.setRole(RoleType.PREFILL);
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10000);
        cacheStatus.setBlockSize(256);
        w.setCacheStatus(cacheStatus);
        w.setRunningTaskList(new HashMap<>());
        return w;
    }

    private BatchItem batchItem(long requestId, long seqLen, long hitCache) {
        Request req = new Request();
        req.setRequestId(requestId);
        req.setSeqLen(seqLen);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        ctx.setConfig(SchedulingTestConfig.batchConfig());
        // For prediction, hitCache comes from prefill.debugInfo.  Use null prefill → 0,
        // but the caller's hitCache parameter is what matters for prediction — we set it
        // via the constructor as a convenience; the predictor will call item.hitCache()
        // which reads prefill.debugInfo.hitCacheLen, so we must build a real ServerStatus.
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

    private FlexlbConfig affinityConfig(long maxExtraTtftMs, double minHitRate) {
        FlexlbConfig config = new FlexlbConfig();
        useBestOnly(config);
        RoutingConfig.CacheAffinityConfig affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(maxExtraTtftMs);
        affinity.setMinPrefixHitPercent(minHitRate);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        return config;
    }

    private static void useBestOnly(FlexlbConfig config) {
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter().getRoles()
                        .getPrefill().getSelector();
        selector.setCandidateChoice(new RoutingConfig.BestOnlyConfig());
    }

    private static void setFormula(FlexlbConfig config, String expression) {
        RoutingConfig.FormulaEstimatorConfig estimator =
                (RoutingConfig.FormulaEstimatorConfig) config.getRouter().getRoles()
                        .getPrefill().getExecutionTimeEstimator();
        estimator.setExpression(expression);
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
