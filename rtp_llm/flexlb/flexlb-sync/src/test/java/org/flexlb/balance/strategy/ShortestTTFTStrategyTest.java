package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.cache.service.CacheAwareService;
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
import org.junit.jupiter.api.Assertions;
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
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(resourceMeasure);
        Mockito.when(resourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        Mockito.when(cacheAwareService.findMatchingEngines(Mockito.anyList(), Mockito.any(), Mockito.any())).thenReturn(new HashMap<>());

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