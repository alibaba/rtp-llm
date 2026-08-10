package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskStateEnum;
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
    void prefillWaterLevelUsesOnlyLocalInTransitAndWaitingTasks() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(20);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.updateTaskStates(Map.of(
                "request-1", new TaskInfo(),
                "request-2", new TaskInfo(),
                "request-3", new TaskInfo(),
                "request-4", new TaskInfo(),
                "request-5", new TaskInfo()), Map.of(), Map.of());

        assertEquals(0.0, measure.calculateWorkerWaterLevel(workerStatus));

        addLocalTasks(workerStatus, 5);

        assertEquals(25.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void prefillWaterLevelUsesLocalInTransitTasks() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(20);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
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
        workerStatus.refreshInTransitAndWaitingStats();
        assertTrue(measure.isResourceAvailable(workerStatus));
    }

    @Test
    void prefillResourceDoesNotFilterByUncachedTokens() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(2);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        TaskInfo localTask = task("large-request", 100_000_000, 0, false);
        localTask.setPredictedPrefixLength(0);
        workerStatus.putLocalTask("large-request", localTask);

        assertEquals(1, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(100_000_000, workerStatus.getInTransitAndWaitingUncachedTokens());
        assertEquals(50.0, measure.calculateWorkerWaterLevel(workerStatus));
        assertTrue(measure.isResourceAvailable(workerStatus));
    }

    @Test
    void prefillQueueSizeExcludesLocalTaskAlreadyReportedAsRunningByEngine() {
        FlexlbConfig config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(1);
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService(config));
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setAlive(true);
        TaskInfo localTask = task("request", 48_000, 0, false);
        localTask.setPredictedPrefixLength(0);
        workerStatus.putLocalTask("request", localTask);
        updateTaskStatus(workerStatus, Map.of(), Map.of("request", task("request", 48_000, 0, false)));

        assertEquals(0.0, measure.calculateWorkerWaterLevel(workerStatus));
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

    private static TaskInfo task(String requestId, long inputLength, long prefixLength, boolean prefixLengthValid) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(prefixLength);
        task.setPrefixLengthValid(prefixLengthValid);
        return task;
    }

    private static void updateTaskStatus(WorkerStatus workerStatus,
                                         Map<String, TaskInfo> waitingTaskInfo,
                                         Map<String, TaskInfo> runningTaskInfo) {
        workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, Map.of());
    }

    private static void addLocalTasks(WorkerStatus workerStatus, int count) {
        for (int i = 0; i < count; i++) {
            String requestId = "local-request-" + i;
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.updateTaskState(TaskStateEnum.IN_TRANSIT);
            workerStatus.getLocalTaskMap().put(requestId, task);
        }
        workerStatus.refreshInTransitAndWaitingStats();
    }
}
