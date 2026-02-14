package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.Map;

@Slf4j
class WeightedCacheLoadBalancerTest {

    private ConfigService configService;

    @BeforeEach
    void setUp() {
        configService = new ConfigService();
    }

    @org.junit.jupiter.api.AfterEach
    void tearDown() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
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
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = new DecodeResourceMeasure(configService, engineWorkerStatus);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(configService, engineWorkerStatus, resourceMeasureFactory);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId("1000");

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertFalse(status.isSuccess());
        Assertions.assertNotNull(status.getMessage());
    }

    @Test
    void should_use_uniform_distribution_when_all_cache_usages_are_equal() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getUsedKvCacheTokens().set(1000);
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getUsedKvCacheTokens().set(1000);
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        worker3.getUsedKvCacheTokens().set(1000);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);
        decodeMap.put("127.0.0.3:8080", worker3);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId("1000");

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_prioritize_workers_with_lower_cache_usage_when_normalized_values_negative() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

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

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId("1000");

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_handle_group_selection_when_group_parameter_provided() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;

        // Create workers for specific group
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.setGroup("group-a");
        worker1.getUsedKvCacheTokens().set(1000);

        modelStatus.getDecodeStatusMap().put("127.0.0.1:8080", worker1);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId("1000");

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, "group-a");

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void should_use_exponential_decay_for_balanced_weight_distribution_when_cache_usage_differs() {
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        // Create two workers to test exponential decay weight distribution
        // Normalized values are -500 and +500
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getUsedKvCacheTokens().set(500);  // Below average 1000, normalizedValue = -500

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getUsedKvCacheTokens().set(1500); // Above average 1000, normalizedValue = +500

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);

        Request req = new Request();
        req.setSeqLen(1000);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(Mockito.any())).thenReturn(true);
        WeightedCacheLoadBalancer weightedCacheLoadBalancer = new WeightedCacheLoadBalancer(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);

        // Run multiple iterations to verify weight distribution
        int totalRuns = 10000;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(String.valueOf(1000L + i));
            ServerStatus status = weightedCacheLoadBalancer.select(balanceContext, RoleType.DECODE, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
                // Rollback to reset local tasks and cache usage
                weightedCacheLoadBalancer.rollBack(selectedIp + ":8080", String.valueOf(1000L + i));
            }
        }

        int worker1Count = selectionCount.getOrDefault("127.0.0.1", 0);
        int worker2Count = selectionCount.getOrDefault("127.0.0.2", 0);
        log.info("Exponential decay weight distribution verification: worker1={} ({}%), worker2={} ({}%)",
                worker1Count, worker1Count * 100.0 / totalRuns, worker2Count, worker2Count * 100.0 / totalRuns);

        // Verify worker1 (lower cache usage) is selected more frequently
        Assertions.assertTrue(worker1Count > worker2Count,
                "Worker with lower cache usage should be selected more frequently");

        // Verify weight ratio is more balanced (improvement from exponential decay algorithm)
        double ratio = (double) worker1Count / worker2Count;
        Assertions.assertTrue(ratio >= 1.5 && ratio <= 3.0,
                "Weight ratio should be between 1.5-3.0, actual ratio: %.2f".formatted(ratio));

        double worker1Ratio = (double) worker1Count / totalRuns;
        double worker2Ratio = (double) worker2Count / totalRuns;

        log.info("Exponential decay weight distribution verification: worker1={} ({}%), worker2={} ({}%), weight ratio: {}",
                worker1Count, worker1Ratio * 100, worker2Count, worker2Ratio * 100, "%.2f".formatted(ratio));
    }
}
