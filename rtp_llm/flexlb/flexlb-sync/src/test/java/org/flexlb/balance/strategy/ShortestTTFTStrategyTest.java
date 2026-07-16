package org.flexlb.balance.strategy;

import ch.qos.logback.classic.Level;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.CacheMatchResult;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.FlexlbConfig;
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
        businessLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("flexlbLogger");
        originalLevel = businessLogger.getLevel();
        businessLogger.setLevel(Level.DEBUG);
    }

    @AfterEach
    void restoreLogLevel() {
        businessLogger.setLevel(originalLevel);
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
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any()))
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
