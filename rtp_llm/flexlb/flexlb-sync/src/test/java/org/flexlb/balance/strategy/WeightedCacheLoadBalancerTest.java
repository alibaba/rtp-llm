package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.balance.BalanceContext;
import org.flexlb.service.config.SystemEnvConfigService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

@Slf4j
class WeightedCacheLoadBalancerTest {

    private SystemEnvConfigService systemEnvConfigService;

    @BeforeEach
    void setUp() {
        systemEnvConfigService = new SystemEnvConfigService();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.clear();
    }

    WorkerStatus createWorkerStatus(String ip) {

        WorkerStatus workerStatus = new WorkerStatus();

        workerStatus.setIp(ip);
        workerStatus.setPort(8080);
        workerStatus.setSite("na61");
        workerStatus.setAlive(true);
        return workerStatus;
    }

    @Test
    void should_handle_empty_worker_map_when_no_workers_available() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(systemEnvConfigService, engineWorkerStatus);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");
        req.setSeqLen(1000);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);
        balanceContext.setInterRequestId(1000);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertFalse(status.isSuccess());
        Assertions.assertNotNull(status.getMessage());
    }

    @Test
    void should_use_uniform_distribution_when_all_cache_usages_are_equal() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP
                .get("test-model").getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getUsedKvCacheTokens().set(1000);
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getUsedKvCacheTokens().set(1000);
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        worker3.getUsedKvCacheTokens().set(1000);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);
        decodeMap.put("127.0.0.3:8080", worker3);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");
        req.setSeqLen(1000);

        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(systemEnvConfigService, engineWorkerStatus);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);
        balanceContext.setInterRequestId(1000);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_prioritize_workers_with_lower_cache_usage_when_normalized_values_negative() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP
                .get("test-model").getDecodeStatusMap();

        // Worker1: cacheUsed = 500 (well below average)
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getUsedKvCacheTokens().set(500);

        // Worker2: cacheUsed = 1500 (above average)
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getUsedKvCacheTokens().set(1500);

        // Worker3: cacheUsed = 1000 (average)
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        worker3.getUsedKvCacheTokens().set(1000);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);
        decodeMap.put("127.0.0.3:8080", worker3);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");
        req.setSeqLen(1000);

        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(systemEnvConfigService, engineWorkerStatus);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);
        balanceContext.setInterRequestId(1000);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_handle_group_selection_when_group_parameter_provided() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());

        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.get("test-model");

        // Create workers for specific group
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.setGroup("group-a");
        worker1.getUsedKvCacheTokens().set(1000);

        modelStatus.getDecodeStatusMap().put("127.0.0.1:8080", worker1);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");
        req.setSeqLen(1000);

        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(systemEnvConfigService, engineWorkerStatus);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);
        balanceContext.setInterRequestId(1000);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, "group-a");

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void should_use_exponential_decay_for_balanced_weight_distribution_when_cache_usage_differs() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.put("test-model", new ModelWorkerStatus());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP
                .get("test-model").getDecodeStatusMap();

        // 创建两个worker，测试指数衰减算法的权重平衡性
        // 归一化后 normalizedValue = -500 和 +500
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getUsedKvCacheTokens().set(500);  // 低于平均值1000，normalizedValue = -500

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getUsedKvCacheTokens().set(1500); // 高于平均值1000，normalizedValue = +500

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);

        MasterRequest req = new MasterRequest();
        req.setModel("test-model");
        req.setSeqLen(1000);

        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(systemEnvConfigService, engineWorkerStatus);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setMasterRequest(req);

        // 进行大量测试以验证权重分布
        int totalRuns = 10000;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.setInterRequestId(1000L + i);
            ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        int worker1Count = selectionCount.getOrDefault("127.0.0.1", 0);
        int worker2Count = selectionCount.getOrDefault("127.0.0.2", 0);

        // 验证worker1 (低缓存使用) 被选择次数更多
        Assertions.assertTrue(worker1Count > worker2Count,
                "低缓存使用的worker应该被选择次数更多");

        // 验证权重比例更均衡 (指数衰减算法的改进效果)
        double ratio = (double) worker1Count / worker2Count;
        Assertions.assertTrue(ratio >= 1.5 && ratio <= 3.0,
                "权重比例应该在1.5-3.0之间，实际比例: %.2f".formatted(ratio));

        double worker1Ratio = (double) worker1Count / totalRuns;
        double worker2Ratio = (double) worker2Count / totalRuns;

        log.info("指数衰减算法权重分布验证：worker1={} ({}%), worker2={} ({}%), 权重比例: {}",
                worker1Count, worker1Ratio * 100, worker2Count, worker2Ratio * 100, "%.2f".formatted(ratio));
    }
}
