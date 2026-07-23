package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.StrategyConfigs;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
class ShortestTTFTStrategyTest {

    @BeforeEach
    void setUp() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
    }

    @AfterEach
    void cleanUp() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
    }

    @Test
    void test() {

        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String/*ip*/, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        Map<String, TaskInfo> waitingTaskList = new HashMap<>();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        Map<String, TaskInfo> finishedTaskList = new HashMap<>();
        ConcurrentHashMap<String, TaskInfo> localTaskList = new ConcurrentHashMap<>();
        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1", 200, waitingTaskList, runningTaskList, finishedTaskList, localTaskList);

        Map<String, TaskInfo> waitingTaskList1 = new HashMap<>();
        Map<String, TaskInfo> runningTaskList1 = new HashMap<>();
        Map<String, TaskInfo> finishedTaskList1 = new HashMap<>();
        ConcurrentHashMap<String, TaskInfo> localTaskList1 = new ConcurrentHashMap<>();
        WorkerStatus workerStatus1 = createWorkerStatus("127.0.0.2", 100, waitingTaskList1, runningTaskList1, finishedTaskList1, localTaskList1);

        prefillStatusMap.put("127.0.0.1:8080", workerStatus);
        prefillStatusMap.put("127.0.0.2:8080", workerStatus1);
        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(12345L);
        List<Long> blockCacheKeys = new ArrayList<>();
        blockCacheKeys.add(1L);
        blockCacheKeys.add(2L);
        req.setBlockCacheKeys(blockCacheKeys);

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure resourceMeasure = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        Mockito.when(configService.getStrategyConfigs()).thenReturn(new StrategyConfigs());
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any())).thenReturn(new HashMap<>());

        ShortestTTFTStrategy staticCacheLoadBalancer =
                new ShortestTTFTStrategy(engineWorkerStatus, engineHealthReporter, cacheAwareService, resourceMeasureFactory, configService);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(new FlexlbConfig());
        balanceContext.setRequest(req);
        ServerStatus result = staticCacheLoadBalancer.select(balanceContext, RoleType.PREFILL, null);
        if (!result.isSuccess()) {
            System.out.println("Result not successful - code: " + result.getCode() + ", message: " + result.getMessage());
        }
        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
    }

    @Test
    void should_short_circuit_to_lowest_ttft_when_fixed_candidate_pool_size_is_one() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus lowestTtftWorker = createWorkerStatus("127.0.0.1", 0);
        lowestTtftWorker.getLastSelectedTime().set(1000L);
        WorkerStatus olderWorkerWithHigherTtft = createWorkerStatus("127.0.0.2", 5);
        olderWorkerWithHigherTtft.getLastSelectedTime().set(1L);

        prefillStatusMap.put(lowestTtftWorker.getIpPort(), lowestTtftWorker);
        prefillStatusMap.put(olderWorkerWithHigherTtft.getIpPort(), olderWorkerWithHigherTtft);

        ShortestTTFTStrategy strategy = createStrategy(engineWorkerStatus, fixedCandidatePoolConfigService(1));

        ServerStatus result = strategy.select(createBalanceContext(100L), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.1", result.getServerIp());
    }

    @Test
    void should_apply_fairness_when_fixed_candidate_pool_size_is_larger_than_one() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus lowestTtftWorker = createWorkerStatus("127.0.0.1", 0);
        lowestTtftWorker.getLastSelectedTime().set(1000L);
        WorkerStatus olderWorkerWithSimilarTtft = createWorkerStatus("127.0.0.2", 0);
        olderWorkerWithSimilarTtft.getLastSelectedTime().set(1L);

        prefillStatusMap.put(lowestTtftWorker.getIpPort(), lowestTtftWorker);
        prefillStatusMap.put(olderWorkerWithSimilarTtft.getIpPort(), olderWorkerWithSimilarTtft);

        ShortestTTFTStrategy strategy = createStrategy(engineWorkerStatus, fixedCandidatePoolConfigService(2));

        ServerStatus result = strategy.select(createBalanceContext(100L), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
    }

    @Test
    void should_apply_fairness_when_ratio_candidate_pool_has_multiple_workers() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus recentlySelectedWorker = createWorkerStatus("127.0.0.1", 0);
        recentlySelectedWorker.getLastSelectedTime().set(1000L);
        WorkerStatus olderWorker = createWorkerStatus("127.0.0.2", 0);
        olderWorker.getLastSelectedTime().set(1L);

        prefillStatusMap.put(recentlySelectedWorker.getIpPort(), recentlySelectedWorker);
        prefillStatusMap.put(olderWorker.getIpPort(), olderWorker);

        ShortestTTFTStrategy strategy = createStrategy(engineWorkerStatus, ratioCandidatePoolConfigService(1.0));

        ServerStatus result = strategy.select(createBalanceContext(100L), RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
    }

    @Test
    void should_report_candidate_and_selected_routing_cache_match_tokens() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> prefillStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();

        WorkerStatus longestMatchWorker = createWorkerStatus("127.0.0.1", 1000);
        WorkerStatus selectedWorker = createWorkerStatus("127.0.0.2", 0);
        prefillStatusMap.put(longestMatchWorker.getIpPort(), longestMatchWorker);
        prefillStatusMap.put(selectedWorker.getIpPort(), selectedWorker);

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure resourceMeasure = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
                .thenReturn(Map.of(
                        longestMatchWorker.getIpPort(), 4,
                        selectedWorker.getIpPort(), 1));

        ShortestTTFTStrategy strategy = new ShortestTTFTStrategy(
                engineWorkerStatus,
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                fixedCandidatePoolConfigService(1));

        BalanceContext balanceContext = createBalanceContext(4096L);
        // Page-RR sends virtual block tokens here: seq_size_per_block * cp_size.
        balanceContext.getRequest().setCacheKeyBlockSize(1024L);

        ServerStatus result = strategy.select(balanceContext, RoleType.PREFILL, null);

        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
        Mockito.verify(engineHealthReporter).reportRoutingSelectedCacheMatchMetrics(
                RoleType.PREFILL, "127.0.0.2", 1024L, 4096L);
        Mockito.verify(engineHealthReporter).reportRoutingCandidateMaxCacheMatchMetrics(
                RoleType.PREFILL, "127.0.0.2", 4096L);
    }

    private ShortestTTFTStrategy createStrategy(EngineWorkerStatus engineWorkerStatus, ConfigService configService) {
        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure resourceMeasure = Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);

        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any())).thenReturn(new HashMap<>());

        return new ShortestTTFTStrategy(
                engineWorkerStatus,
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory,
                configService);
    }

    private ConfigService fixedCandidatePoolConfigService(int size) {
        StrategyConfigs strategyConfigs = new StrategyConfigs();
        StrategyConfigs.CandidatePoolConfig candidatePool = strategyConfigs.getShortestTtft().getCandidatePool();
        candidatePool.setMode(StrategyConfigs.CandidatePoolMode.FIXED);
        candidatePool.setSize(size);
        strategyConfigs.normalize();

        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.getStrategyConfigs()).thenReturn(strategyConfigs);
        return configService;
    }

    private ConfigService ratioCandidatePoolConfigService(double ratio) {
        StrategyConfigs strategyConfigs = new StrategyConfigs();
        StrategyConfigs.CandidatePoolConfig candidatePool = strategyConfigs.getShortestTtft().getCandidatePool();
        candidatePool.setMode(StrategyConfigs.CandidatePoolMode.RATIO);
        candidatePool.setRatio(ratio);
        strategyConfigs.normalize();

        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.getStrategyConfigs()).thenReturn(strategyConfigs);
        return configService;
    }

    private BalanceContext createBalanceContext(long seqLen) {
        Request req = new Request();
        req.setSeqLen(seqLen);
        req.setRequestId(12345L);
        req.setBlockCacheKeys(List.of(1L, 2L));

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(new FlexlbConfig());
        balanceContext.setRequest(req);
        return balanceContext;
    }

    WorkerStatus createWorkerStatus(String ip, long runningQueueTime) {
        return createWorkerStatus(
                ip,
                runningQueueTime,
                new HashMap<>(),
                new HashMap<>(),
                new HashMap<>(),
                new ConcurrentHashMap<>());
    }

    WorkerStatus createWorkerStatus(String ip,
                                    long runningQueueTime,
                                    Map<String, TaskInfo> waitingTaskInfo,
                                    Map<String, TaskInfo> finishedTaskList,
                                    Map<String, TaskInfo> runningTaslList,
                                    ConcurrentHashMap<String, TaskInfo> localTaskList) {
        WorkerStatus workerStatus = new WorkerStatus();

        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        workerStatus.setRole(RoleType.PREFILL.getCode());
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10000);
        cacheStatus.setBlockSize(256);
        workerStatus.setCacheStatus(cacheStatus);
        workerStatus.getRunningQueueTime().getAndSet(runningQueueTime);
        workerStatus.setWaitingTaskList(waitingTaskInfo);
        workerStatus.updateTaskStates(waitingTaskInfo, runningTaslList, finishedTaskList);
        workerStatus.setRunningTaskList(runningTaslList);
        return workerStatus;
    }

}
