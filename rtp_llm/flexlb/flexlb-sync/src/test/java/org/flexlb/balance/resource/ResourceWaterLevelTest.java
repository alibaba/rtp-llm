package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ResourceWaterLevelTest {

    @Test
    void prefillWaterLevelUsesEngineWaitingQueueWhenItIsLarger() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(20);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setWaitingTaskList(Map.of(
                "request-1", new TaskInfo(),
                "request-2", new TaskInfo(),
                "request-3", new TaskInfo(),
                "request-4", new TaskInfo(),
                "request-5", new TaskInfo()));

        assertEquals(25.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void prefillWaterLevelUsesLocalOutstandingTasksWhenEngineStatusLags() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(20);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setWaitingTaskList(Map.of("request-1", new TaskInfo()));
        addLocalTasks(workerStatus, 8);

        assertEquals(40.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void prefillResourceStopsRoutingWhenLocalOutstandingTasksReachThreshold() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(3);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        workerStatus.setWaitingTaskList(Map.of("request-1", new TaskInfo()));
        addLocalTasks(workerStatus, 3);

        assertFalse(measure.isResourceAvailable(workerStatus));
    }

    @Test
    void prefillWaterLevelReachesFullScaleAtAvailabilityThreshold() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(3);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        addLocalTasks(workerStatus, 3);

        assertEquals(100.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void prefillAverageWaterLevelIsFullWhenEveryWorkerReachesAvailabilityThreshold() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(3);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        Map<String, WorkerStatus> workerStatusMap = new HashMap<>();

        for (int i = 0; i < 10; i++) {
            WorkerStatus workerStatus = new WorkerStatus();
            addLocalTasks(workerStatus, 3);
            workerStatusMap.put("worker-" + i, workerStatus);
        }

        assertEquals(100.0, measure.calculateAverageWaterLevel(workerStatusMap));
    }

    @Test
    void prefillResourceBecomesAvailableImmediatelyBelowThreshold() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(3);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        addLocalTasks(workerStatus, 3);

        assertFalse(measure.isResourceAvailable(workerStatus));

        workerStatus.getLocalTaskMap().remove("local-request-2");
        assertTrue(measure.isResourceAvailable(workerStatus));
    }

    @Test
    void decodeWaterLevelUsesConfiguredKvCacheRange() {
        FlexlbConfig config = new FlexlbConfig();
        config.setDecodeFullSpeedThreshold(40);
        config.setDecodeStopThreshold(80);
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setUsedKvCacheTokens(new AtomicLong(60));
        workerStatus.setAvailableKvCacheTokens(new AtomicLong(40));

        assertEquals(50.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    private static ConfigService configService(FlexlbConfig config) {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return configService;
    }

    private static void addLocalTasks(WorkerStatus workerStatus, int count) {
        for (int i = 0; i < count; i++) {
            workerStatus.getLocalTaskMap().put("local-request-" + i, new TaskInfo());
        }
    }
}
