package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.domain.CacheMatchQuery;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.ScoredWorker;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

class CacheAffinityFirstStrategyTest {

    private static final long BLOCK_SIZE = 2000;
    private static final long INPUT_TOKENS = 50000;

    @BeforeEach
    void setUp() {
        clearWorkerStatuses();
    }

    @AfterEach
    void tearDown() {
        clearWorkerStatuses();
    }

    @Test
    void spreadsColdRequestsAcrossThreeWorkers() {
        WorkerStatus workerA = createWorker("127.0.0.1", 0);
        WorkerStatus workerB = createWorker("127.0.0.2", 0);
        WorkerStatus workerC = createWorker("127.0.0.3", 0);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(workerA, workerB, workerC), Map.of());

        Set<String> selectedWorkers = new HashSet<>();
        for (int i = 0; i < 3; i++) {
            selectedWorkers.add(select(strategy, cacheAffinityConfig(), "cold-" + i).getServerIp());
        }

        Assertions.assertEquals(
                Set.of(workerA.getIp(), workerB.getIp(), workerC.getIp()), selectedWorkers);
    }

    @Test
    void usesCacheLeaderWhenExtraTtftIsWithinTolerance() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 6000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus thirdWorker = createWorker("127.0.0.3", 1000);
        cacheLeader.getLastSelectedTime().set(2);
        shortestTtftWorker.getLastSelectedTime().set(1);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, thirdWorker),
                Map.of(
                        cacheLeader.getIpPort(), 16,
                        shortestTtftWorker.getIpPort(), 15,
                        thirdWorker.getIpPort(), 15));

        ServerStatus selected = select(strategy, cacheAffinityConfig(), "cache-affinity");

        Assertions.assertEquals(cacheLeader.getIp(), selected.getServerIp());
        Assertions.assertSame(
                strategy,
                LoadBalanceStrategyFactory.getLoadBalancer(
                        LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST));
    }

    @Test
    void usesShortestTtftWorkerWhenExtraTtftExceedsTolerance() {
        WorkerStatus overloadedCacheLeader = createWorker("127.0.0.1", 30000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus thirdWorker = createWorker("127.0.0.3", 1000);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(overloadedCacheLeader, shortestTtftWorker, thirdWorker),
                Map.of(
                        overloadedCacheLeader.getIpPort(), 17,
                        shortestTtftWorker.getIpPort(), 15,
                        thirdWorker.getIpPort(), 15));

        ServerStatus selected = select(strategy, cacheAffinityConfig(), "queue-spillover");

        Assertions.assertEquals(shortestTtftWorker.getIp(), selected.getServerIp());
    }

    @Test
    void usesCacheLeaderWhenExtraWorkIsWithinConfiguredTolerance() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 13000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus thirdWorker = createWorker("127.0.0.3", 1000);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, thirdWorker),
                Map.of(
                        cacheLeader.getIpPort(), 16,
                        shortestTtftWorker.getIpPort(), 15,
                        thirdWorker.getIpPort(), 15));
        FlexlbConfig config = cacheAffinityConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(12_000);

        ServerStatus selected = select(strategy, config, "configured-extra-work-tolerance");

        Assertions.assertEquals(cacheLeader.getIp(), selected.getServerIp());
    }

    @Test
    void usesCacheLeaderAtConfiguredExtraWorkBoundary() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 12000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus thirdWorker = createWorker("127.0.0.3", 1000);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, thirdWorker),
                Map.of(
                        cacheLeader.getIpPort(), 16,
                        shortestTtftWorker.getIpPort(), 15,
                        thirdWorker.getIpPort(), 15));
        FlexlbConfig config = cacheAffinityConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(10_000);

        ServerStatus selected = select(strategy, config, "extra-work-tolerance-boundary");

        Assertions.assertEquals(cacheLeader.getIp(), selected.getServerIp());
    }

    @Test
    void usesShortestQueueWhenAllWorkersHaveSameCommonPrefix() {
        WorkerStatus workerA = createWorker("127.0.0.1", 5000);
        WorkerStatus shortestQueueWorker = createWorker("127.0.0.2", 0);
        WorkerStatus workerC = createWorker("127.0.0.3", 1000);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(workerA, shortestQueueWorker, workerC),
                Map.of(
                        workerA.getIpPort(), 15,
                        shortestQueueWorker.getIpPort(), 15,
                        workerC.getIpPort(), 15));

        ServerStatus selected = select(strategy, cacheAffinityConfig(), "common-prefix");

        Assertions.assertEquals(shortestQueueWorker.getIp(), selected.getServerIp());
    }

    @Test
    void doesNotProactivelySelectNeverSelectedIdleColdWorker() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 4000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus coldWorker = createWorker("127.0.0.3", 0);
        cacheLeader.getLastSelectedTime().set(2);
        shortestTtftWorker.getLastSelectedTime().set(1);
        coldWorker.getLastSelectedTime().set(-1);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, coldWorker),
                Map.of(
                        cacheLeader.getIpPort(), 17,
                        shortestTtftWorker.getIpPort(), 15,
                        coldWorker.getIpPort(), 0));

        ServerStatus selected = select(strategy, cacheAffinityConfig(), "no-cold-warmup");

        Assertions.assertEquals(cacheLeader.getIp(), selected.getServerIp());
        Assertions.assertNotEquals(coldWorker.getIp(), selected.getServerIp());
        Assertions.assertEquals(-1, coldWorker.getLastSelectedTime().get());
        Assertions.assertEquals(0, coldWorker.getRunningQueueTime().get());
        Assertions.assertTrue(coldWorker.getLocalTaskMap().isEmpty());
    }

    @Test
    void keepsCacheAffinityOrderWhenCacheLeaderWasClaimedConcurrently() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 0);
        WorkerStatus cacheFallback = createWorker("127.0.0.2", 0);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.3", 0);
        cacheLeader.getLastSelectedTime().set(101);
        cacheFallback.getLastSelectedTime().set(200);
        shortestTtftWorker.getLastSelectedTime().set(300);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, cacheFallback, shortestTtftWorker), Map.of());
        BalanceContext balanceContext = new BalanceContext();

        ScoredWorker selectedWorker = strategy.selectBestWorker(
                List.of(
                        new ScoredWorker(cacheLeader, 2000, 2000, 100),
                        new ScoredWorker(cacheFallback, 1500, 1500, 200),
                        new ScoredWorker(shortestTtftWorker, 1000, 1000, 300)),
                balanceContext, RoleType.PREFILL, null, INPUT_TOKENS, cacheAffinityConfig());

        Assertions.assertSame(cacheFallback, selectedWorker.worker());
        var decision = balanceContext.getShortestTtftDecisionByRole().get(RoleType.PREFILL);
        Assertions.assertEquals("CACHE_AFFINITY_FALLBACK", decision.selectionReason());
        Assertions.assertEquals("127.0.0.1:8080", decision.cacheAffinityDecision().cacheLeaderIpPort());
        Assertions.assertEquals("127.0.0.3:8080", decision.cacheAffinityDecision().shortestTtftWorkerIpPort());
    }

    @Test
    void usesShortestTtftWhenCacheLeaderReachesOutstandingThreshold() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 0);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus thirdWorker = createWorker("127.0.0.3", 1000);
        putPendingTask(cacheLeader, "existing", 1_000_000, 0);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, thirdWorker),
                Map.of(
                        cacheLeader.getIpPort(), 25,
                        shortestTtftWorker.getIpPort(), 15,
                        thirdWorker.getIpPort(), 10));
        FlexlbConfig config = cacheAffinityConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(2_000_000);
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(1_000_000);

        ServerStatus selected = select(strategy, config, "watermark-fallback");

        Assertions.assertEquals(shortestTtftWorker.getIp(), selected.getServerIp());
    }

    @Test
    void admitsUsingExistingWatermarkThenBlocksLaterRequests() {
        WorkerStatus worker = createWorker("127.0.0.1", 0);
        putPendingTask(worker, "existing", 990_000, 0);
        CacheAffinityFirstStrategy strategy = createStrategy(List.of(worker), Map.of());
        FlexlbConfig config = cacheAffinityConfig();
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(1_000_000);

        ServerStatus first = select(strategy, config, "crosses-watermark");
        ServerStatus second = select(strategy, config, "blocked-after-crossing");

        Assertions.assertTrue(first.isSuccess());
        Assertions.assertEquals(1_040_000, worker.getOutstandingUncachedTokens());
        Assertions.assertFalse(second.isSuccess());
        Assertions.assertFalse(worker.getLocalTaskMap().containsKey("blocked-after-crossing"));
    }

    @Test
    void usesLatestRunningRemainingTokensForOutstandingWatermark() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 0);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        putPendingTask(cacheLeader, "running-request", 1_200_000, 0);
        updateRunningProgress(cacheLeader, "running-request", 1_200_000, 1_000_000);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker),
                Map.of(
                        cacheLeader.getIpPort(), 25,
                        shortestTtftWorker.getIpPort(), 15));
        FlexlbConfig config = cacheAffinityConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(2_000_000);
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(1_000_000);

        ServerStatus atThreshold = select(strategy, config, "at-threshold");
        updateRunningProgress(cacheLeader, "running-request", 1_200_000, 900_000);
        ServerStatus belowThreshold = select(strategy, config, "below-threshold");

        Assertions.assertEquals(shortestTtftWorker.getIp(), atThreshold.getServerIp());
        Assertions.assertEquals(cacheLeader.getIp(), belowThreshold.getServerIp());
    }

    @Test
    void cacheAffinityGuardsAreDisabledByDefault() {
        FlexlbConfig config = new FlexlbConfig();

        Assertions.assertEquals(0, config.getCacheAffinityFirstMaxExtraWorkTokens());
        Assertions.assertEquals(
                0, config.getCacheAffinityFirstOutstandingUncachedTokensThreshold());
    }

    private CacheAffinityFirstStrategy createStrategy(
            List<WorkerStatus> workers, Map<String, Integer> cacheMatches) {
        Map<String, WorkerStatus> prefillWorkers =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        for (WorkerStatus worker : workers) {
            prefillWorkers.put(worker.getIpPort(), worker);
        }

        ResourceMeasure resourceMeasure = Mockito.mock(ResourceMeasure.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);

        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.any(CacheMatchQuery.class)))
                .thenReturn(new CacheMatchResult(
                        HostCacheMatch.fromLocalMatches(cacheMatches), CacheMatchSource.KVCM, 123, BLOCK_SIZE));

        return new CacheAffinityFirstStrategy(
                new EngineWorkerStatus(new ModelMetaConfig()),
                Mockito.mock(EngineHealthReporter.class),
                cacheAwareService,
                resourceMeasureFactory);
    }

    private ServerStatus select(
            CacheAffinityFirstStrategy strategy, FlexlbConfig config, String requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(INPUT_TOKENS);
        request.setBlockCacheKeys(List.of(1L));

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(config);
        balanceContext.setRequest(request);
        return strategy.select(balanceContext, RoleType.PREFILL, null);
    }

    private FlexlbConfig cacheAffinityConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(25_000);
        return config;
    }

    private WorkerStatus createWorker(String ip, long queueWork) {
        WorkerStatus worker = new WorkerStatus();
        worker.setIp(ip);
        worker.setPort(8080);
        worker.setRole(RoleType.PREFILL.getCode());
        worker.setGroup("default");
        worker.setAlive(true);
        worker.getRunningQueueTime().set(queueWork);
        worker.getLastSelectedTime().set(System.nanoTime() / 1000);

        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setBlockSize(BLOCK_SIZE);
        cacheStatus.setAvailableKvCache(1000000);
        worker.setCacheStatus(cacheStatus);
        return worker;
    }

    private void putPendingTask(WorkerStatus worker,
                                String requestId,
                                long inputTokens,
                                long predictedHitTokens) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputTokens);
        task.setPrefixLength(predictedHitTokens);
        task.setPredictedPrefixLength(predictedHitTokens);
        worker.putLocalTask(requestId, task);
    }

    private void updateRunningProgress(WorkerStatus worker,
                                       String requestId,
                                       long inputTokens,
                                       long remainingPrefillTokens) {
        TaskInfo runningTask = new TaskInfo();
        runningTask.setRequestId(requestId);
        runningTask.setInputLength(inputTokens);
        runningTask.setPrefixLength(0);
        runningTask.setPrefixLengthValid(true);
        runningTask.setRemainingPrefillTokens(remainingPrefillTokens);
        worker.updateTaskStates(Map.of(), Map.of(requestId, runningTask), Map.of());
    }

    private void clearWorkerStatuses() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }
}
