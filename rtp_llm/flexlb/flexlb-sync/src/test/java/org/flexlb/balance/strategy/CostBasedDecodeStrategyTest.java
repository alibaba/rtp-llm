package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.HashMap;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;

@Slf4j
class CostBasedDecodeStrategyTest {

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

    /** Create an EndpointRegistry with DecodeEndpoints registered for each WorkerStatus entry. */
    private EndpointRegistry createDecodeRegistry(Map<String, WorkerStatus> workerMap) {
        EndpointRegistry registry = new EndpointRegistry(configService, null, Mockito.mock(BatchSchedulerReporter.class));
        for (Map.Entry<String, WorkerStatus> entry : workerMap.entrySet()) {
            WorkerStatus ws = entry.getValue();
            ws.setGrpcPort(9090);
            DecodeEndpoint ep = registry.ensureDecodeEndpoint(entry.getKey(), ws);
            // Initialize reported KV cache from status
            ep.calibrate(null, null, ws.getAvailableKvCacheTokens().get());
        }
        return registry;
    }

    @Test
    void should_handle_empty_worker_map_when_no_workers_available() {
        EndpointRegistry emptyRegistry = new EndpointRegistry(configService, null, Mockito.mock(BatchSchedulerReporter.class));
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), emptyRegistry);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = new DecodeResourceMeasure(configService);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertFalse(status.isSuccess());
        Assertions.assertNotNull(status.getMessage());
    }

    @Test
    void should_use_uniform_distribution_when_all_cache_usages_are_equal() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getTotalKvCacheTokens().set(10000);
        worker1.getAvailableKvCacheTokens().set(9000);
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getTotalKvCacheTokens().set(10000);
        worker2.getAvailableKvCacheTokens().set(9000);
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        worker3.getTotalKvCacheTokens().set(10000);
        worker3.getAvailableKvCacheTokens().set(9000);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);
        decodeMap.put("127.0.0.3:8080", worker3);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_prioritize_workers_with_lower_cache_usage_when_normalized_values_negative() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getTotalKvCacheTokens().set(10000);
        worker1.getAvailableKvCacheTokens().set(9500);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getTotalKvCacheTokens().set(10000);
        worker2.getAvailableKvCacheTokens().set(8500);

        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        worker3.getTotalKvCacheTokens().set(10000);
        worker3.getAvailableKvCacheTokens().set(9000);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);
        decodeMap.put("127.0.0.3:8080", worker3);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_handle_group_selection_when_group_parameter_provided() {
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.setGroup("group-a");

        modelStatus.getDecodeStatusMap().put("127.0.0.1:8080", worker1);

        EndpointRegistry registry = createDecodeRegistry(modelStatus.getDecodeStatusMap());
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, "group-a");

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void should_use_exponential_decay_for_balanced_weight_distribution_when_cache_usage_differs() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getTotalKvCacheTokens().set(10000);
        worker1.getAvailableKvCacheTokens().set(9500);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getTotalKvCacheTokens().set(10000);
        worker2.getAvailableKvCacheTokens().set(8500);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);

        Request req = new Request();
        req.setSeqLen(1000);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        int totalRuns = 10000;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(1000L + i);
            ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
                costBasedDecodeStrategy.rollBack(registry.get(selectedIp + ":8080"), 1000L + i);
            }
        }

        int worker1Count = selectionCount.getOrDefault("127.0.0.1", 0);
        int worker2Count = selectionCount.getOrDefault("127.0.0.2", 0);
        log.info("Exponential decay weight distribution verification: worker1={} ({}%), worker2={} ({}%)",
                worker1Count, worker1Count * 100.0 / totalRuns, worker2Count, worker2Count * 100.0 / totalRuns);

        Assertions.assertTrue(worker1Count > worker2Count,
                "Worker with lower cache usage should be selected more frequently");

        double ratio = (double) worker1Count / worker2Count;
        Assertions.assertTrue(ratio >= 1.5 && ratio <= 3.0,
                "Weight ratio should be between 1.5-3.0, actual ratio: %.2f".formatted(ratio));
    }

    @Test
    void should_skip_worker_with_insufficient_kv_cache_capacity() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getTotalKvCacheTokens().set(1000);
        worker1.getAvailableKvCacheTokens().set(100);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getTotalKvCacheTokens().set(1000);
        worker2.getAvailableKvCacheTokens().set(800);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);

        Request req = new Request();
        req.setSeqLen(500);
        req.setRequestId(2000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.2", status.getServerIp());
    }

    @Test
    void should_return_error_when_all_workers_kv_insufficient() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getTotalKvCacheTokens().set(1000);
        worker1.getAvailableKvCacheTokens().set(50);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getTotalKvCacheTokens().set(1000);
        worker2.getAvailableKvCacheTokens().set(100);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(new ModelMetaConfig(), registry);

        Request req = new Request();
        req.setSeqLen(200);
        req.setRequestId(3000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        Mockito.when(decodeResourceMeasure.isResourceAvailable(any())).thenReturn(true);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(configService, engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertFalse(status.isSuccess());
        Assertions.assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), status.getCode());
    }
}
