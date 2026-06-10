package org.flexlb.balance.resource;

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
        WorkerStatus worker = createAliveDecodeWorker();
        worker.setRunningTaskList(taskMap(1L, 2L, 3L, 4L));
        worker.getLocalTaskMap().put(5L, taskInfo(5L));

        assertTrue(measure.isResourceAvailable(worker));
        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void worker_should_be_unavailable_when_decode_concurrency_limit_reached() {
        config.setDecodeConcurrencyLimit(2);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        WorkerStatus worker = createAliveDecodeWorker();
        worker.setRunningTaskList(taskMap(1L));
        worker.getLocalTaskMap().put(2L, taskInfo(2L));

        assertFalse(measure.isResourceAvailable(worker));
    }

    @Test
    void concurrency_count_should_deduplicate_reported_and_local_request_ids() {
        config.setDecodeConcurrencyLimit(2);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        WorkerStatus worker = createAliveDecodeWorker();
        worker.setRunningTaskList(taskMap(1L));
        worker.getLocalTaskMap().put(1L, taskInfo(1L));

        assertTrue(measure.isResourceAvailable(worker));
    }

    @Test
    void concurrency_water_level_should_contribute_to_serviceability() {
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        WorkerStatus worker = createAliveDecodeWorker();
        worker.setRunningTaskList(taskMap(1L, 2L));
        worker.getLocalTaskMap().put(3L, taskInfo(3L));

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_should_use_higher_value_between_kv_cache_and_concurrency() {
        config.setDecodeConcurrencyLimit(4);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService);
        WorkerStatus worker = createAliveDecodeWorker();
        worker.getUsedKvCacheTokens().set(70);
        worker.getAvailableKvCacheTokens().set(30);
        worker.setRunningTaskList(taskMap(1L));

        assertEquals(75.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    private WorkerStatus createAliveDecodeWorker() {
        WorkerStatus worker = new WorkerStatus();
        worker.setAlive(true);
        worker.getUsedKvCacheTokens().set(0);
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
