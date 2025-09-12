package org.flexlb.balance.strategy;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.domain.balance.WhaleMasterConfig;
import org.flexlb.domain.batch.SubmitBatchResponse;
import org.flexlb.service.batch.BatchService;
import org.flexlb.service.config.ConfigService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */

public class ShortestTTFTStrategyTest {

    @Test
    public void test() {

        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        engineWorkerStatus.getModelRoleWorkerStatusMap().put("test-model", new ModelWorkerStatus());
        ConcurrentHashMap<String/*ip*/, WorkerStatus> prefillStatusMap = engineWorkerStatus.getModelRoleWorkerStatusMap().get("test-model").getPrefillStatusMap();
        List<TaskInfo> runningTaskList = new ArrayList<>();
        List<TaskInfo> finishedTaskList = new ArrayList<>();
        ConcurrentHashMap <Long, TaskInfo> localTaskList = new ConcurrentHashMap<>();
        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1", 200, runningTaskList, finishedTaskList, localTaskList);

        List<TaskInfo> runningTaskList1 = new ArrayList<>();
        List<TaskInfo> finishedTaskList1 = new ArrayList<>();
        ConcurrentHashMap <Long, TaskInfo> localTaskList1 = new ConcurrentHashMap<>();
        WorkerStatus workerStatus1 = createWorkerStatus("127.0.0.2", 100, runningTaskList1, finishedTaskList1, localTaskList1);

        prefillStatusMap.put("127.0.0.1:8080", workerStatus);
        prefillStatusMap.put("127.0.0.2:8080", workerStatus1);
        MasterRequest req = new MasterRequest();
        req.setModel("test-model");
        req.setSeqLen(1000);
        List<Long> blockCacheKeys = new ArrayList<>();
        blockCacheKeys.add(1L);
        blockCacheKeys.add(2L);
        req.setBlockCacheKeys(blockCacheKeys);

        EngineHealthReporter engineHealthReporter = Mockito.mock(EngineHealthReporter.class);
        CacheAwareService cacheAwareService = Mockito.mock(CacheAwareService.class);
        BatchService batchService = Mockito.mock(BatchService.class);
        Mockito.when(batchService.submitBatch(Mockito.any())).thenReturn(SubmitBatchResponse.success());
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new WhaleMasterConfig());

        ShortestTTFTStrategy staticCacheLoadBalancer =
                new ShortestTTFTStrategy(engineWorkerStatus, engineHealthReporter, cacheAwareService);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(new WhaleMasterConfig());
        balanceContext.setMasterRequest(req);
        balanceContext.setWorkerCalcParallel(4);
        ServerStatus result = staticCacheLoadBalancer.select(balanceContext, RoleType.PREFILL, null);
        Assertions.assertTrue(result.isSuccess());
        Assertions.assertEquals("127.0.0.2", result.getServerIp());
    }

    WorkerStatus createWorkerStatus(String ip,
                                    long runningQueueTime,
                                    List<TaskInfo> finishedTaskList,
                                    List<TaskInfo> runningTaslList,
                                    ConcurrentHashMap<Long, TaskInfo> localTaskList) {
        WorkerStatus workerStatus = new WorkerStatus();

        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        CacheStatus cacheStatus = new CacheStatus();
        cacheStatus.setAvailableKvCache(10000);
        workerStatus.setCacheStatus(cacheStatus);
        workerStatus.getRunningQueueTime().getAndSet(runningQueueTime);
        workerStatus.clearFinishedTaskAndTimeoutTask(finishedTaskList);
        workerStatus.setRunningTaskList(runningTaslList);
        workerStatus.setLocalTaskMap(localTaskList);
        return workerStatus;
    }

}