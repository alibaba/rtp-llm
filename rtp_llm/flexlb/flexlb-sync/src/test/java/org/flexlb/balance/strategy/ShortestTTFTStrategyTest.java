package org.flexlb.balance.strategy;

import ch.qos.logback.classic.Level;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.CacheMatchResult;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
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
import org.slf4j.LoggerFactory;

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

    private ch.qos.logback.classic.Logger businessLogger;
    private Level originalLevel;

    @BeforeEach
    void enableDebugDecisionSnapshots() {
        clearWorkerStatuses();
        businessLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        originalLevel = businessLogger.getLevel();
        businessLogger.setLevel(Level.DEBUG);
    }

    @AfterEach
    void restoreLogLevel() {
        businessLogger.setLevel(originalLevel);
        clearWorkerStatuses();
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
        waitingTaskList1.put("waiting-1", createTask("waiting-1", 400, 128));
        runningTaskList1.put("running-1", createTask("running-1", 600, 256));
        localTaskList1.put("waiting-1", createTask("waiting-1", 400, 0));
        localTaskList1.put("running-1", createTask("running-1", 600, 0));
        WorkerStatus workerStatus1 = createWorkerStatus("127.0.0.2", 100, waitingTaskList1, runningTaskList1, finishedTaskList1, localTaskList1);

        prefillStatusMap.put("127.0.0.1:8080", workerStatus);
        prefillStatusMap.put("127.0.0.2:8080", workerStatus1);
        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId("request-12345");
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
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(
                        Mockito.anyString(), Mockito.anyList(), Mockito.anyLong(), Mockito.any(), Mockito.any()))
                .thenReturn(new CacheMatchResult(
                        Map.of("127.0.0.2:8080", 3), CacheMatchSource.KVCM, 123));

        ShortestTTFTStrategy staticCacheLoadBalancer =
                new ShortestTTFTStrategy(engineWorkerStatus, engineHealthReporter, cacheAwareService, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(new FlexlbConfig());
        balanceContext.setRequest(req);
        ServerStatus result = staticCacheLoadBalancer.select(balanceContext, RoleType.PREFILL, null);
        if (!result.isSuccess()) {
            System.out.println("Result not successful - code: " + result.getCode() + ", message: " + result.getMessage());
        }
        Assertions.assertTrue(result.isSuccess(), "Result should be successful but got: " + result.getMessage());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
        Assertions.assertEquals("request-12345", result.getRequestId());
        Assertions.assertEquals("KVCM", balanceContext.getCacheMatchSource());
        Assertions.assertEquals(123, balanceContext.getCacheMatchQueryTimeUs());
        Assertions.assertEquals(1, balanceContext.getCacheMatchQueryCount());
        Mockito.verify(cacheAwareService).findMatchingEngines(
                "request-12345", blockCacheKeys, 256L, RoleType.PREFILL, null);
        TaskInfo selectedTask = workerStatus1.getLocalTaskMap().get("request-12345");
        Assertions.assertNotNull(selectedTask);
        Assertions.assertEquals(768, selectedTask.getPredictedPrefixLength());
        Assertions.assertEquals("KVCM", selectedTask.getCacheMatchSource());
        BalanceContext.CacheMatchSelection selection =
                balanceContext.getCacheMatchSelectionByRole().get(RoleType.PREFILL);
        Assertions.assertEquals("127.0.0.2", selection.selectedIp());
        Assertions.assertEquals(768, selection.hitCacheTokens());

        var decision = balanceContext.getShortestTtftDecisionByRole().get(RoleType.PREFILL);
        Assertions.assertNotNull(decision);
        Assertions.assertEquals(2, decision.workers().size());
        var selectedDecision = decision.workers().stream()
                .filter(worker -> worker.selected())
                .findFirst()
                .orElseThrow();
        Assertions.assertEquals("127.0.0.2", selectedDecision.ip());
        Assertions.assertEquals(768, selectedDecision.requestHitCacheTokens());
        Assertions.assertEquals(462, selectedDecision.requestPrefillTime());
        Assertions.assertEquals(100, selectedDecision.queueTime());
        Assertions.assertEquals(562, selectedDecision.estimatedTtft());
        Assertions.assertEquals(2, selectedDecision.trackedTaskCount());
        Assertions.assertEquals(1, selectedDecision.waitingTaskCount());
        Assertions.assertEquals(1, selectedDecision.runningTaskCount());
        Assertions.assertEquals(128, selectedDecision.waitingTasks().getFirst().hitCacheTokens());
        Assertions.assertEquals(256, selectedDecision.runningTasks().getFirst().hitCacheTokens());

        businessLogger.setLevel(Level.INFO);
        BalanceContext infoContext = new BalanceContext();
        infoContext.setConfig(new FlexlbConfig());
        infoContext.setRequest(req);
        staticCacheLoadBalancer.select(infoContext, RoleType.PREFILL, null);
        Assertions.assertTrue(infoContext.getShortestTtftDecisionByRole().isEmpty());
    }

    @Test
    void prefersCacheLeaderWithTwoBlockLeadWhenTtftIsSimilar() {
        FlexlbConfig config = cacheFocusedConfig();
        WorkerStatus shortestTtftWorker = createWorkerStatus("127.0.0.1", 0, 2128);
        WorkerStatus cacheLeader = createWorkerStatus("127.0.0.2", 7000, 2128);
        WorkerStatus thirdWorker = createWorkerStatus("127.0.0.3", 1000, 2128);

        SelectionResult selection = select(
                List.of(shortestTtftWorker, cacheLeader, thirdWorker),
                Map.of(
                        shortestTtftWorker.getIpPort(), 15,
                        cacheLeader.getIpPort(), 17,
                        thirdWorker.getIpPort(), 15),
                config,
                50000,
                "cache-lead-two-blocks");

        Assertions.assertEquals(cacheLeader.getIp(), selection.serverStatus().getServerIp());
        TaskInfo selectedTask = cacheLeader.getLocalTaskMap().get("cache-lead-two-blocks");
        Assertions.assertNotNull(selectedTask);
        Assertions.assertEquals(1.0, selectedTask.getCacheHitDiscount());
        Assertions.assertEquals(13824, selectedTask.estimatePrefillTime());

        var decision = selection.balanceContext()
                .getShortestTtftDecisionByRole()
                .get(RoleType.PREFILL);
        Assertions.assertNotNull(decision);
        Assertions.assertEquals(3616.0, decision.similarTtftThreshold());
        Assertions.assertEquals(3, decision.workers().stream()
                .filter(worker -> worker.topCandidate())
                .count());
    }

    @Test
    void keepsShortestTtftWorkerWhenCacheLeadIsBelowConfiguredBlocks() {
        FlexlbConfig config = cacheFocusedConfig();
        WorkerStatus shortestTtftWorker = createWorkerStatus("127.0.0.1", 0, 2128);
        WorkerStatus oneBlockLeader = createWorkerStatus("127.0.0.2", 2500, 2128);
        WorkerStatus thirdWorker = createWorkerStatus("127.0.0.3", 1000, 2128);

        SelectionResult selection = select(
                List.of(shortestTtftWorker, oneBlockLeader, thirdWorker),
                Map.of(
                        shortestTtftWorker.getIpPort(), 15,
                        oneBlockLeader.getIpPort(), 16,
                        thirdWorker.getIpPort(), 15),
                config,
                50000,
                "cache-lead-one-block");

        Assertions.assertEquals(shortestTtftWorker.getIp(), selection.serverStatus().getServerIp());
    }

    @Test
    void doesNotPreferCacheLeaderWhenItsQueueMakesTtftTooLong() {
        FlexlbConfig config = cacheFocusedConfig();
        WorkerStatus shortestTtftWorker = createWorkerStatus("127.0.0.1", 0, 2128);
        WorkerStatus busyCacheLeader = createWorkerStatus("127.0.0.2", 20000, 2128);
        WorkerStatus thirdWorker = createWorkerStatus("127.0.0.3", 1000, 2128);

        SelectionResult selection = select(
                List.of(shortestTtftWorker, busyCacheLeader, thirdWorker),
                Map.of(
                        shortestTtftWorker.getIpPort(), 15,
                        busyCacheLeader.getIpPort(), 20,
                        thirdWorker.getIpPort(), 15),
                config,
                50000,
                "busy-cache-leader");

        Assertions.assertEquals(shortestTtftWorker.getIp(), selection.serverStatus().getServerIp());
    }

    private FlexlbConfig cacheFocusedConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillCacheHitDiscount(1.0);
        config.setPrefillCachePreferenceMinBlockGap(2);
        config.setShortestTtftSimilarityThresholdRatio(0.2);
        return config;
    }

    private SelectionResult select(
            List<WorkerStatus> workers,
            Map<String, Integer> cacheMatches,
            FlexlbConfig config,
            long inputTokens,
            String requestId) {
        Map<String, WorkerStatus> prefillStatusMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap();
        for (WorkerStatus worker : workers) {
            prefillStatusMap.put(worker.getIpPort(), worker);
        }

        Request request = new Request();
        request.setSeqLen(inputTokens);
        request.setRequestId(requestId);
        request.setBlockCacheKeys(List.of(1L));

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        org.flexlb.balance.resource.ResourceMeasure resourceMeasure =
                Mockito.mock(org.flexlb.balance.resource.ResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(
                        Mockito.anyString(), Mockito.anyList(), Mockito.anyLong(), Mockito.any(), Mockito.any()))
                .thenReturn(new CacheMatchResult(cacheMatches, CacheMatchSource.KVCM, 123));

        ShortestTTFTStrategy strategy = new ShortestTTFTStrategy(
                new EngineWorkerStatus(new ModelMetaConfig()),
                engineHealthReporter,
                cacheAwareService,
                resourceMeasureFactory);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(config);
        balanceContext.setRequest(request);
        return new SelectionResult(
                strategy.select(balanceContext, RoleType.PREFILL, null), balanceContext);
    }

    private WorkerStatus createWorkerStatus(String ip, long runningQueueTime, long blockSize) {
        WorkerStatus workerStatus = createWorkerStatus(
                ip,
                runningQueueTime,
                new HashMap<>(),
                new HashMap<>(),
                new HashMap<>(),
                new ConcurrentHashMap<>());
        workerStatus.getCacheStatus().setBlockSize(blockSize);
        return workerStatus;
    }

    private void clearWorkerStatuses() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    private record SelectionResult(ServerStatus serverStatus, BalanceContext balanceContext) {
    }

    private TaskInfo createTask(String requestId, long inputLength, long prefixLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        return task;
    }

    WorkerStatus createWorkerStatus(String ip,
                                    long runningQueueTime,
                                    Map<String, TaskInfo> waitingTaskInfo,
                                    Map<String, TaskInfo> runningTaslList,
                                    Map<String, TaskInfo> finishedTaskList,
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
        workerStatus.setLocalTaskMap(localTaskList);
        workerStatus.updateTaskStates(waitingTaskInfo, runningTaslList, finishedTaskList);
        workerStatus.setRunningTaskList(runningTaslList);
        return workerStatus;
    }

}
