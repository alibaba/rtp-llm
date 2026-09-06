package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.TaskStateEnum;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ResourceWaterLevelTest {

    @Test
    void prefillWaterLevelIncludesWorkerObservedWaitingTasks() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService());
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.updateTaskStates(Map.of(
                "1", task(1L),
                "2", task(2L),
                "3", task(3L),
                "4", task(4L),
                "5", task(5L)), Map.of(), Map.of());

        assertEquals(25.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void prefillWaterLevelIncludesLocalInTransitTasks() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService());
        WorkerStatus workerStatus = new WorkerStatus();
        addLocalTasks(workerStatus, 8);

        assertEquals(40.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void prefillWaterLevelExcludesTaskAlreadyReportedRunningByEngine() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService());
        WorkerStatus workerStatus = new WorkerStatus();
        TaskInfo local = task(1L);
        workerStatus.putLocalTask("1", local);
        workerStatus.updateTaskStates(Map.of(), Map.of("1", task(1L)), Map.of());

        assertEquals(0.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    @Test
    void decodeWaterLevelUsesInternalKvCacheRange() {
        DecodeResourceMeasure measure = new DecodeResourceMeasure(configService());
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.getTotalKvCacheTokens().set(100);
        workerStatus.getAvailableKvCacheTokens().set(40);

        assertEquals(50.0, measure.calculateWorkerWaterLevel(workerStatus));
    }

    private static ConfigService configService() {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        return configService;
    }

    private static TaskInfo task(long requestId) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        return task;
    }

    private static void addLocalTasks(WorkerStatus workerStatus, int count) {
        for (int i = 0; i < count; i++) {
            String requestId = String.valueOf(i);
            TaskInfo task = task(i);
            task.updateTaskState(TaskStateEnum.IN_TRANSIT);
            workerStatus.putLocalTask(requestId, task);
        }
        workerStatus.refreshInTransitAndWaitingStats();
    }
}
