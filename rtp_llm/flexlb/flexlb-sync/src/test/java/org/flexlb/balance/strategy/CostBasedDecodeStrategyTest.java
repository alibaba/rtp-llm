package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
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
        EndpointRegistry registry = new EndpointRegistry(configService, () -> null,
                Mockito.mock(BatchSchedulerReporter.class));
        for (Map.Entry<String, WorkerStatus> entry : workerMap.entrySet()) {
            WorkerStatus ws = entry.getValue();
            ws.setGrpcPort(9090);
            DecodeEndpoint ep = (DecodeEndpoint) registry.ensureEndpoint(
                    RoleType.DECODE, entry.getKey(), ws);
            // Initialize reported KV cache from status
            ep.onWorkerStatusUpdate(ws, new WorkerStatusResponse());
        }
        return registry;
    }

    private void allowDecodeSelection(DecodeResourceMeasure measure) {
        Mockito.when(measure.isResourceAvailable(any())).thenReturn(true);
        Mockito.when(measure.isQueuePlacementAvailable(any())).thenReturn(true);
    }

    @Test
    void should_handle_empty_worker_map_when_no_workers_available() {
        EndpointRegistry emptyRegistry = new EndpointRegistry(configService, () -> null,
                Mockito.mock(BatchSchedulerReporter.class));
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(emptyRegistry);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = new DecodeResourceMeasure(configService);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

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
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

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
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

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
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        Request req = new Request();
        req.setSeqLen(1000);
        req.setRequestId(1000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

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
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        Request req = new Request();
        req.setSeqLen(1000);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

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
                costBasedDecodeStrategy.rollBack(
                        registry.get(RoleType.DECODE, selectedIp + ":8080"), 1000L + i);
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
    void should_not_overflow_weight_when_decode_workers_have_large_kv_usage_gap() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        int workerCount = 16;
        for (int i = 1; i <= workerCount; i++) {
            String ip = "127.0.0." + i;
            WorkerStatus worker = createWorkerStatus(ip);
            worker.getTotalKvCacheTokens().set(1_000_000);
            // With 15 workers using 800K tokens and one empty worker, the previous
            // average-centered formula evaluated exp(750), which overflows to Infinity.
            worker.getAvailableKvCacheTokens().set(i == 1 ? 1_000_000 : 200_000);
            decodeMap.put(ip + ":8080", worker);
        }

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(
                engineWorkerStatus, resourceMeasureFactory);

        Request req = new Request();
        req.setSeqLen(1);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        for (int i = 0; i < 100; i++) {
            long requestId = 10_000L + i;
            req.setRequestId(requestId);
            ServerStatus status = Assertions.assertDoesNotThrow(
                    () -> costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null));

            Assertions.assertTrue(status.isSuccess());
            Assertions.assertEquals("127.0.0.1", status.getServerIp(),
                    "The worker with the lowest KV usage should have the highest stable weight");
            costBasedDecodeStrategy.rollBack(
                    registry.get(RoleType.DECODE, status.getServerIp() + ":8080"), requestId);
        }
    }

    @Test
    void should_skip_worker_with_insufficient_kv_cache_capacity() {
        configService.loadBalanceConfig().setScheduler(new DirectSchedulerConfig());
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
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        Request req = new Request();
        req.setSeqLen(500);
        req.setRequestId(2000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.2", status.getServerIp());
    }

    @Test
    void should_return_error_when_all_workers_kv_insufficient() {
        configService.loadBalanceConfig().setScheduler(new DirectSchedulerConfig());
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
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        Request req = new Request();
        req.setSeqLen(200);
        req.setRequestId(3000L);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertFalse(status.isSuccess());
        Assertions.assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), status.getCode());
    }

    @Test
    void fifoQueueCanPlaceBehindTransientKvPressure_whilePriorityKeepsAdmissionGate() {
        Map<String, WorkerStatus> decodeMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.getTotalKvCacheTokens().set(1_000);
        worker.getAvailableKvCacheTokens().set(1_000);
        decodeMap.put("127.0.0.1:8080", worker);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        DecodeEndpoint endpoint = registry.getDecode("127.0.0.1:8080");
        endpoint.reserveQueued(1L, 400, 700, 50);
        endpoint.reserveQueued(2L, 400, 700, 50);

        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        Assertions.assertFalse(measure.isResourceAvailable(endpoint));

        ResourceMeasureFactory factory = Mockito.mock(ResourceMeasureFactory.class);
        Mockito.when(factory.getMeasure(Mockito.any())).thenReturn(measure);
        CostBasedDecodeStrategy strategy = new CostBasedDecodeStrategy(
                new EngineWorkerStatus(registry), factory);

        Request request = new Request();
        request.setSeqLen(100);
        request.setRequestId(3L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(configService.loadBalanceConfig());

        ServerStatus fifoResult = strategy.select(context, RoleType.DECODE, null);
        Assertions.assertTrue(fifoResult.isSuccess());
        Assertions.assertTrue(endpoint.layeredAdmissionView().queued().contains(3L));

        strategy.rollBack(endpoint, 3L);
        configService.loadBalanceConfig().queueScheduler()
                .setOrdering(new PriorityOrderingConfig());
        request.setRequestId(4L);
        ServerStatus priorityResult = strategy.select(context, RoleType.DECODE, null);
        Assertions.assertFalse(priorityResult.isSuccess(),
                "PRIORITY must preserve the inclusive admission gate for preemption/classification");
    }

    @Test
    void hotspotFilterUsesOthersAverageExcludingSelf() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        worker1.getTotalKvCacheTokens().set(10000);
        worker1.getAvailableKvCacheTokens().set(9000);
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        worker2.getTotalKvCacheTokens().set(10000);
        worker2.getAvailableKvCacheTokens().set(9000);

        decodeMap.put("127.0.0.1:8080", worker1);
        decodeMap.put("127.0.0.2:8080", worker2);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);

        // Hot engine carries 4 running-phase reservations (engineLoad = 4),
        // the cool one a single reservation (engineLoad = 1). With the old
        // self-inclusive average the threshold was 3.0 * (4+1)/2 = 7.5, so
        // the hot engine could mathematically NEVER be filtered with n=2.
        // The leave-one-out baseline (others avg = 1) must reject it:
        // 4 > 3.0 * 1.
        DecodeEndpoint hot = registry.getDecode("127.0.0.1:8080");
        DecodeEndpoint cool = registry.getDecode("127.0.0.2:8080");
        for (long rid = 1L; rid <= 4L; rid++) {
            hot.reserve(rid, 100, 500, 50);
        }
        cool.reserve(5L, 100, 500, 50);

        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

        Request req = new Request();
        req.setSeqLen(100);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        for (int i = 0; i < 30; i++) {
            long requestId = 20_000L + i;
            req.setRequestId(requestId);
            ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);
            Assertions.assertTrue(status.isSuccess());
            Assertions.assertEquals("127.0.0.2", status.getServerIp(),
                    "the hot engine (load=4 vs others-avg=1) must be outlier-rejected");
            costBasedDecodeStrategy.rollBack(
                    registry.get(RoleType.DECODE, status.getServerIp() + ":8080"), requestId);
        }
    }

    @Test
    void singleCandidateSkipsOutlierRejection() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        worker.getTotalKvCacheTokens().set(10000);
        worker.getAvailableKvCacheTokens().set(9000);
        decodeMap.put("127.0.0.1:8080", worker);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        DecodeEndpoint endpoint = registry.getDecode("127.0.0.1:8080");
        // A lone engine carrying heavy load: there are no "other" engines to
        // be an outlier against, so the relative outlier checks must be
        // skipped — rejecting the only candidate could only yield
        // NO_AVAILABLE_WORKER with nothing to gain.
        for (long rid = 1L; rid <= 6L; rid++) {
            endpoint.reserve(rid, 100, 500, 50);
        }

        EngineWorkerStatus engineWorkerStatus = new EngineWorkerStatus(registry);
        ResourceMeasureFactory resourceMeasureFactory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure decodeResourceMeasure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(resourceMeasureFactory.getMeasure(Mockito.any())).thenReturn(decodeResourceMeasure);
        allowDecodeSelection(decodeResourceMeasure);
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(engineWorkerStatus, resourceMeasureFactory);

        Request req = new Request();
        req.setSeqLen(100);
        req.setRequestId(30_000L);
        BalanceContext balanceContext = new BalanceContext();
        balanceContext.setRequest(req);
        balanceContext.setConfig(configService.loadBalanceConfig());

        ServerStatus status = costBasedDecodeStrategy.select(balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }
}
