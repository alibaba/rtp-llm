package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.CacheMatchResult;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
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
    void prefersCacheLeaderWhileSavedWorkCoversExtraQueue() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 5000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus thirdWorker = createWorker("127.0.0.3", 1000);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, thirdWorker),
                Map.of(
                        cacheLeader.getIpPort(), 17,
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
    void spillsToShortestTtftWorkerWhenCacheLeaderQueueIsTooLong() {
        WorkerStatus overloadedCacheLeader = createWorker("127.0.0.1", 9000);
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
    void sendsOneWarmupRequestToAnIdleColdWorker() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 5000);
        WorkerStatus warmWorker = createWorker("127.0.0.2", 0);
        WorkerStatus coldWorker = createWorker("127.0.0.3", 0);
        coldWorker.getLastSelectedTime().set(-1);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, warmWorker, coldWorker),
                Map.of(
                        cacheLeader.getIpPort(), 17,
                        warmWorker.getIpPort(), 16,
                        coldWorker.getIpPort(), 0));

        ServerStatus first = select(strategy, cacheAffinityConfig(), "cold-warmup");
        ServerStatus second = select(strategy, cacheAffinityConfig(), "after-warmup");

        Assertions.assertEquals(coldWorker.getIp(), first.getServerIp());
        Assertions.assertNotEquals(coldWorker.getIp(), second.getServerIp());
        Assertions.assertTrue(coldWorker.getRunningQueueTime().get() > 0);
    }

    @Test
    void doesNotExploreARecentlySelectedColdWorker() {
        WorkerStatus cacheLeader = createWorker("127.0.0.1", 5000);
        WorkerStatus shortestTtftWorker = createWorker("127.0.0.2", 0);
        WorkerStatus coldWorker = createWorker("127.0.0.3", 0);
        CacheAffinityFirstStrategy strategy = createStrategy(
                List.of(cacheLeader, shortestTtftWorker, coldWorker),
                Map.of(
                        cacheLeader.getIpPort(), 17,
                        shortestTtftWorker.getIpPort(), 16,
                        coldWorker.getIpPort(), 0));

        ServerStatus selected = select(strategy, cacheAffinityConfig(), "recent-cold-worker");

        Assertions.assertEquals(shortestTtftWorker.getIp(), selected.getServerIp());
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
        Mockito.when(cacheAwareService.findMatchingEngines(
                        Mockito.anyString(), Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(new CacheMatchResult(cacheMatches, CacheMatchSource.KVCM, 123));

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
        config.setPrefillCacheHitDiscount(1.0);
        config.setPrefillCachePreferenceMinBlockGap(2);
        config.setCacheAffinityFirstQueueToleranceFactor(2.0);
        config.setCacheAffinityFirstColdWorkerProbeIntervalMs(5000);
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

    private void clearWorkerStatuses() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }
}
