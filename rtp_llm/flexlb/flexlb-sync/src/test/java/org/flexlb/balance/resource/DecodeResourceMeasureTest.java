package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.BatchDispatchExecutor;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class DecodeResourceMeasureTest {

    @Mock
    private ConfigService configService;

    private FlexlbConfig config;

    private EndpointRegistry registry;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = new EndpointRegistry(configService, Mockito.mock(EngineGrpcClient.class),
                Mockito.mock(BatchDispatchExecutor.class), Mockito.mock(InflightStore.class),
                Mockito.mock(BatchSchedulerReporter.class), null);
    }

    @Test
    void concurrency_limit_disabled_should_not_affect_decode_availability() {
        config.setDecodeConcurrencyLimit(0);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        DecodeEndpoint endpoint = registerDecodeEndpoint("worker", createAliveWorkerStatus());
        endpoint.getStatus().setRunningTaskList(taskMap(1L, 2L, 3L, 4L));

        assertTrue(measure.isResourceAvailable(endpoint));
        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", endpoint.getStatus())));
    }

    @Test
    void worker_should_be_unavailable_when_decode_concurrency_limit_reached() {
        config.setDecodeConcurrencyLimit(2);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        endpoint.reserve(1L, 0, 0);
        endpoint.reserve(2L, 0, 0);
        // decodeTotalLoad() = engineWork.size(0) + inflightRequests.size(2) = 2, limit = 2, 2 >= 2 → unavailable
        assertFalse(measure.isResourceAvailable(endpoint));
    }

    @Test
    void worker_should_be_available_when_inflight_below_concurrency_limit() {
        config.setDecodeConcurrencyLimit(3);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        endpoint.reserve(1L, 0, 0);
        // decodeTotalLoad() = engineWork.size(0) + inflightRequests.size(1) = 1, limit = 3, 1 < 3 → available
        assertTrue(measure.isResourceAvailable(endpoint));
    }

    @Test
    void concurrency_water_level_should_contribute_to_serviceability() {
        // No endpoint registered for "worker" → falls back to raw WorkerStatus fields
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        WorkerStatus worker = createAliveWorkerStatus();
        worker.setRunningTaskList(taskMap(1L, 2L, 3L));

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_should_use_higher_value_between_kv_cache_and_concurrency() {
        // No endpoint registered for "worker" → falls back to raw WorkerStatus fields
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        WorkerStatus worker = createAliveWorkerStatus();
        worker.getTotalKvCacheTokens().set(100);
        worker.getAvailableKvCacheTokens().set(30);
        worker.setRunningTaskList(taskMap(1L));

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_should_rise_with_inflight_reserve_via_real_view() {
        // Default thresholds: fullSpeed=40, stop=80
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        WorkerStatus worker = createAliveWorkerStatus(); // total=100, available=100 → raw used=0
        DecodeEndpoint endpoint = registerDecodeEndpoint("worker", worker);

        // Without inflight reserve, real view == raw view → water level 0
        double rawLevel = measure.calculateAverageWaterLevel(Map.of("worker", worker));
        assertEquals(0.0, rawLevel);

        // Reserve 60 expected KV tokens → decodeRealKvUsed=60, used%=60 → (60-40)/(80-40)*100 = 50
        endpoint.reserve(1L, 10, 60);
        double realLevel = measure.calculateAverageWaterLevel(Map.of("worker", worker));
        assertEquals(50.0, realLevel);
        assertTrue(realLevel > rawLevel, "real view with inflight reserve must be more conservative than raw view");
    }

    @Test
    void real_view_water_level_should_include_engine_reported_usage_plus_inflight() {
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);

        // Registered endpoint: reported used=50, inflight expected reserve=20 → decodeRealKvUsed=70 → (70-40)/40*100=75
        WorkerStatus realWorker = createAliveWorkerStatus();
        realWorker.getAvailableKvCacheTokens().set(50);
        DecodeEndpoint endpoint = registerDecodeEndpoint("real", realWorker);
        endpoint.reserve(1L, 10, 20);

        // Unregistered worker with identical raw fields → raw used=50 → (50-40)/40*100=25
        WorkerStatus rawWorker = createAliveWorkerStatus();
        rawWorker.getAvailableKvCacheTokens().set(50);

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("real", realWorker)));
        assertEquals(25.0, measure.calculateAverageWaterLevel(Map.of("raw", rawWorker)));
    }

    @Test
    void concurrency_water_level_should_use_endpoint_total_load() {
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService, registry);
        WorkerStatus worker = createAliveWorkerStatus();
        DecodeEndpoint endpoint = registerDecodeEndpoint("worker", worker);
        endpoint.reserve(1L, 0, 0);
        endpoint.reserve(2L, 0, 0);
        endpoint.reserve(3L, 0, 0);

        // decodeTotalLoad() = 3, limit = 4 → 75%; runningTaskList is empty, only inflight counts
        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    private DecodeEndpoint createAliveDecodeEndpoint() {
        WorkerStatus status = createAliveWorkerStatus();
        return new DecodeEndpoint(status, new FlexlbConfig(), null);
    }

    private DecodeEndpoint registerDecodeEndpoint(String ipPort, WorkerStatus status) {
        return (DecodeEndpoint) registry.ensureEndpoint(RoleType.DECODE, ipPort, status);
    }

    private WorkerStatus createAliveWorkerStatus() {
        WorkerStatus worker = new WorkerStatus();
        worker.setAlive(true);
        worker.getTotalKvCacheTokens().set(100);
        worker.getAvailableKvCacheTokens().set(100);
        return worker;
    }

    private Map<String, TaskInfo> taskMap(Long... requestIds) {
        return Arrays.stream(requestIds)
                .collect(Collectors.toMap(String::valueOf, this::taskInfo));
    }

    private TaskInfo taskInfo(long requestId) {
        TaskInfo taskInfo = new TaskInfo();
        taskInfo.setRequestId(requestId);
        return taskInfo;
    }
}
