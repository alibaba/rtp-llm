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
        return createWorkerStatus(ip, null);
    }

    WorkerStatus createWorkerStatus(String ip, String group) {
        return StrategyTestSupport.workerStatus(
                RoleType.DECODE, group, ip, 8080, 9090,
                true, 0L, 0L);
    }

    /** Create an EndpointRegistry with DecodeEndpoints registered for each WorkerStatus entry. */
    private EndpointRegistry createDecodeRegistry(Map<String, WorkerStatus> workerMap) {
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(configService);
        for (Map.Entry<String, WorkerStatus> entry : workerMap.entrySet()) {
            WorkerStatus ws = entry.getValue();
            registry.registerPreinitializedEndpoint(
                    RoleType.DECODE, entry.getKey(), ws);
        }
        return registry;
    }

    private void allowDecodeSelection(DecodeResourceMeasure measure) {
        Mockito.when(measure.isResourceAvailable(any())).thenReturn(true);
    }

    @Test
    void should_handle_empty_worker_map_when_no_workers_available() {
        EndpointRegistry emptyRegistry = StrategyTestSupport.endpointRegistry(configService);
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

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertNull(status);
    }

    @Test
    void should_use_uniform_distribution_when_all_cache_usages_are_equal() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        setKv(worker1, 10_000, 9_000);
        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        setKv(worker2, 10_000, 9_000);
        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        setKv(worker3, 10_000, 9_000);

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

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_prioritize_workers_with_lower_cache_usage_when_normalized_values_negative() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        setKv(worker1, 10_000, 9_500);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        setKv(worker2, 10_000, 8_500);

        WorkerStatus worker3 = createWorkerStatus("127.0.0.3");
        setKv(worker3, 10_000, 9_000);

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

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_handle_group_selection_when_group_parameter_provided() {
        ModelWorkerStatus modelStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1", "group-a");

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

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, "group-a");

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void should_use_exponential_decay_for_balanced_weight_distribution_when_cache_usage_differs() {
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        setKv(worker1, 10_000, 9_500);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        setKv(worker2, 10_000, 8_500);

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
            ServerStatus status = selectStatus(
                    costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        int worker1Count = selectionCount.getOrDefault("127.0.0.1", 0);
        int worker2Count = selectionCount.getOrDefault("127.0.0.2", 0);
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
            // With 15 workers using 800K tokens and one empty worker, the previous
            // average-centered formula evaluated exp(750), which overflows to Infinity.
            setKv(worker, 1_000_000, i == 1 ? 1_000_000 : 200_000);
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
                    () -> selectStatus(
                            costBasedDecodeStrategy, balanceContext,
                            RoleType.DECODE, null));

            Assertions.assertTrue(status.isSuccess());
            Assertions.assertEquals("127.0.0.1", status.getServerIp(),
                    "The worker with the lowest KV usage should have the highest stable weight");
        }
    }

    @Test
    void should_skip_worker_with_insufficient_kv_cache_capacity() {
        configService.loadBalanceConfig().setScheduler(new DirectSchedulerConfig());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        setKv(worker1, 1_000, 100);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        setKv(worker2, 1_000, 800);

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

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.2", status.getServerIp());
    }

    @Test
    void should_return_error_when_all_workers_kv_insufficient() {
        configService.loadBalanceConfig().setScheduler(new DirectSchedulerConfig());
        Map<String, WorkerStatus> decodeMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker1 = createWorkerStatus("127.0.0.1");
        setKv(worker1, 1_000, 50);

        WorkerStatus worker2 = createWorkerStatus("127.0.0.2");
        setKv(worker2, 1_000, 100);

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

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertNull(status);
    }

    @Test
    void fifoQueueCanPlaceBehindTransientKvPressure_whilePriorityKeepsAdmissionGate() {
        Map<String, WorkerStatus> decodeMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        setKv(worker, 1_000, 1_000);
        decodeMap.put("127.0.0.1:8080", worker);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        DecodeEndpoint endpoint = registry.getDecode("127.0.0.1:8080");
        reserveQueued(endpoint, 1L, 400, 700, 50);
        reserveQueued(endpoint, 2L, 400, 700, 50);

        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        Assertions.assertFalse(measure.isResourceAvailable(endpoint.routingView()));

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

        SelectedRole fifoSelection = strategy.select(
                context, RoleType.DECODE, null);
        ServerStatus fifoResult = fifoSelection.serverStatus();
        Assertions.assertTrue(fifoResult.isSuccess());
        Assertions.assertEquals(1_000L, fifoSelection.decodeTotalKv());
        Assertions.assertFalse(endpoint.layeredAdmissionView().queued().contains(3L),
                "selection must not mutate Decode reservation ownership");
        fifoSelection.close();

        configService.loadBalanceConfig().queueScheduler()
                .setOrdering(new PriorityOrderingConfig());
        request.setRequestId(4L);
        ServerStatus priorityResult = selectStatus(
                strategy, context, RoleType.DECODE, null);
        Assertions.assertNull(priorityResult,
                "PRIORITY must preserve the inclusive admission gate for preemption/classification");
    }

    @Test
    void singleCandidateSkipsOutlierRejection() {
        Map<String, WorkerStatus> decodeMap =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap();

        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        setKv(worker, 10_000, 10_000);
        decodeMap.put("127.0.0.1:8080", worker);

        EndpointRegistry registry = createDecodeRegistry(decodeMap);
        DecodeEndpoint endpoint = registry.getDecode("127.0.0.1:8080");
        // n == 1 with the upstream self-inclusive average: the average IS
        // the engine's own load, so own > multiplier * avg can never hold —
        // a lone engine always stays selectable regardless of its load.
        for (int i = 0; i < 6; i++) {
            reservePinned(endpoint, 400L + i, 0, 0, 50);
        }

        ResourceMeasureFactory factory = Mockito.mock(ResourceMeasureFactory.class);
        DecodeResourceMeasure measure = Mockito.mock(DecodeResourceMeasure.class);
        Mockito.when(factory.getMeasure(Mockito.any())).thenReturn(measure);
        allowDecodeSelection(measure);
        CostBasedDecodeStrategy strategy = new CostBasedDecodeStrategy(
                new EngineWorkerStatus(registry), factory);

        Request request = new Request();
        request.setSeqLen(1);
        request.setRequestId(500L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(configService.loadBalanceConfig());

        ServerStatus status = selectStatus(
                strategy, context, RoleType.DECODE, null);
        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp(),
                "a lone engine's self-inclusive average equals its own load, so it must stay selectable");
    }

    private static void setKv(
            WorkerStatus worker, long totalKv, long availableKv) {
        StrategyTestSupport.publish(worker, StrategyTestSupport.response(
                RoleType.DECODE, true, availableKv, totalKv,
                Math.max(1L, worker.appliedStatusCursor().statusVersion() + 1L)));
    }

    private static void reserveQueued(
            DecodeEndpoint endpoint,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        try (var pin = endpoint.tryPinGeneration()) {
            endpoint.reserveQueuedPinned(
                    pin, requestId, kvTokens, expectedKvTokens, priority);
        }
    }

    /** Non-queued inflight reservation: each call raises engineLoad by one. */
    private static void reservePinned(
            DecodeEndpoint endpoint,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        try (var pin = endpoint.tryPinGeneration()) {
            endpoint.reservePinned(
                    pin, requestId, kvTokens, expectedKvTokens, priority);
        }
    }

    private static ServerStatus selectStatus(
            CostBasedDecodeStrategy strategy,
            BalanceContext context,
            RoleType role,
            String group) {
        try (SelectedRole selected = strategy.select(context, role, group)) {
            return selected == null ? null : selected.serverStatus();
        }
    }
}
