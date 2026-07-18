package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
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

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(config);
    }

    @Test
    void concurrency_limit_disabled_should_not_affect_decode_availability() {
        config.setDecodeConcurrencyLimit(0);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        endpoint.getStatus().setRunningTaskList(taskMap(1L, 2L, 3L, 4L));

        assertTrue(measure.isResourceAvailable(endpoint));
        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", endpoint.getStatus())));
    }

    @Test
    void worker_should_be_unavailable_when_decode_concurrency_limit_reached() {
        config.setDecodeConcurrencyLimit(2);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        endpoint.reserve(1L, 0);
        endpoint.reserve(2L, 0);
        // getTotalLoad() = confirmedRunningCount(0) + inflightRequests.size(2) = 2, limit = 2, 2 >= 2 → unavailable
        assertFalse(measure.isResourceAvailable(endpoint));
    }

    @Test
    void worker_should_be_available_when_inflight_below_concurrency_limit() {
        config.setDecodeConcurrencyLimit(3);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        DecodeEndpoint endpoint = createAliveDecodeEndpoint();
        endpoint.reserve(1L, 0);
        // getTotalLoad() = confirmedRunningCount(0) + inflightRequests.size(1) = 1, limit = 3, 1 < 3 → available
        assertTrue(measure.isResourceAvailable(endpoint));
    }

    @Test
    void concurrency_water_level_should_contribute_to_serviceability() {
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        WorkerStatus worker = createAliveWorkerStatus();
        worker.setRunningTaskList(taskMap(1L, 2L, 3L));

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_should_use_higher_value_between_kv_cache_and_concurrency() {
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        WorkerStatus worker = createAliveWorkerStatus();
        worker.getTotalKvCacheTokens().set(100);
        worker.getAvailableKvCacheTokens().set(30);
        worker.setRunningTaskList(taskMap(1L));

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    private DecodeEndpoint createAliveDecodeEndpoint() {
        WorkerStatus status = createAliveWorkerStatus();
        return new DecodeEndpoint(status);
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
