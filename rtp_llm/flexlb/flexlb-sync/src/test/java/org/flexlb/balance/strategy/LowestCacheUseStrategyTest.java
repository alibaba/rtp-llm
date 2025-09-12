package org.flexlb.balance.strategy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.domain.balance.WhaleMasterConfig;
import org.flexlb.domain.batch.SubmitBatchResponse;
import org.flexlb.service.batch.BatchService;
import org.flexlb.service.config.ConfigService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

public class LowestCacheUseStrategyTest {

    WorkerStatus createWorkerStatus(String ip) {

        WorkerStatus workerStatus = new WorkerStatus();

        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        return workerStatus;
    }

    @Test
    void test(){
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        engineWorkerStatus.getModelRoleWorkerStatusMap().put("test-model", new ModelWorkerStatus());
        Set<Long> blockCacheKeys = new HashSet<>();
        CacheStatus cacheStatus0 = new CacheStatus();
        cacheStatus0.setAvailableKvCache(900);
        cacheStatus0.setBlockSize(64);
        cacheStatus0.setCachedKeys(blockCacheKeys);
        cacheStatus0.setTotalKvCache(2100);
        cacheStatus0.setVersion(1);

        Set<Long> blockCacheKeys2 = new HashSet<>();
        CacheStatus  cacheStatus1 = new CacheStatus();
        cacheStatus1.setAvailableKvCache(1200);
        cacheStatus1.setBlockSize(64);
        cacheStatus1.setCachedKeys(blockCacheKeys2);
        cacheStatus1.setTotalKvCache(2000);
        cacheStatus1.setVersion(1);
        ConcurrentHashMap<String/*ip*/, WorkerStatus> decodemap = engineWorkerStatus.getModelRoleWorkerStatusMap().get("test-model").getDecodeStatusMap();

        WorkerStatus workerStatus = createWorkerStatus("127.0.0.1");
        workerStatus.getKvCacheUsed().set(1999);
        workerStatus.getKvCacheFree().set(10000);
        WorkerStatus workerStatus1 = createWorkerStatus("127.0.0.2");
        workerStatus1.getKvCacheUsed().set(2000);
        workerStatus1.getKvCacheFree().set(20000);
        decodemap.put("127.0.0.1:8080", workerStatus);
        decodemap.put("127.0.0.2:8080", workerStatus1);


        MasterRequest req = new MasterRequest();
        List<Long> blockCacheKeys0 = new ArrayList<>();
        blockCacheKeys0.add(1L);
        blockCacheKeys0.add(2L);
        blockCacheKeys0.add(3L);
        blockCacheKeys0.add(4L);
        blockCacheKeys0.add(5L);
        req.setBlockCacheKeys(blockCacheKeys0);
        req.setModel("test-model");
        req.setSeqLen(1000);

        BatchService batchService = Mockito.mock(BatchService.class);
        Mockito.when(batchService.submitBatch(Mockito.any())).thenReturn(SubmitBatchResponse.success());
        ConfigService configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(new WhaleMasterConfig());
        LowestCacheUsedStrategy lowestCacheUsedStrategy =
                new LowestCacheUsedStrategy(engineWorkerStatus);


        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setConfig(new WhaleMasterConfig());
        balanceContext.setMasterRequest(req);
        balanceContext.setWorkerCalcParallel(4);
        balanceContext.setInterRequestId(1000);

        ServerStatus status = lowestCacheUsedStrategy.select(balanceContext, RoleType.DECODE, null);

        BalanceContext balanceContext2 = new BalanceContext();
        balanceContext2.setConfig(new WhaleMasterConfig());
        balanceContext2.setMasterRequest(req);
        balanceContext2.setWorkerCalcParallel(4);
        balanceContext2.setInterRequestId(1001);
        lowestCacheUsedStrategy.select(balanceContext2, RoleType.DECODE, null);
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }
}
