package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class PrefillResourceMeasureTest {

    @Mock
    private ConfigService configService;

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setPrefillQueueSizeThreshold(2);
        config.setMaxPrefillQueueSize(10);
        when(configService.loadBalanceConfig()).thenReturn(config);
    }

    @Test
    void local_pending_tasks_should_make_prefill_worker_unavailable() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        worker.setWaitingTaskList(new HashMap<>());
        worker.getLocalTaskMap().put(1L, taskInfo(1L, TaskStateEnum.IN_TRANSIT));
        worker.getLocalTaskMap().put(2L, taskInfo(2L, TaskStateEnum.CONFIRMED));

        assertFalse(measure.isResourceAvailable(worker));
    }

    @Test
    void running_local_tasks_should_not_count_as_prefill_queue() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        worker.setWaitingTaskList(new HashMap<>());
        worker.getLocalTaskMap().put(1L, taskInfo(1L, TaskStateEnum.RUNNING));
        worker.getLocalTaskMap().put(2L, taskInfo(2L, TaskStateEnum.RUNNING));

        assertTrue(measure.isResourceAvailable(worker));
    }

    @Test
    void water_level_should_use_higher_value_between_engine_waiting_and_local_pending() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        worker.setWaitingTaskList(Map.of("1", taskInfo(1L, TaskStateEnum.CONFIRMED)));
        worker.getLocalTaskMap().put(1L, taskInfo(1L, TaskStateEnum.IN_TRANSIT));
        worker.getLocalTaskMap().put(2L, taskInfo(2L, TaskStateEnum.CONFIRMED));
        worker.getLocalTaskMap().put(3L, taskInfo(3L, TaskStateEnum.CONFIRMED));

        assertEquals(30.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    private WorkerStatus createAlivePrefillWorker() {
        WorkerStatus worker = new WorkerStatus();
        worker.setAlive(true);
        worker.setRole(RoleType.PREFILL.getCode());
        return worker;
    }

    private TaskInfo taskInfo(long requestId, TaskStateEnum taskState) {
        TaskInfo taskInfo = new TaskInfo();
        taskInfo.setRequestId(requestId);
        taskInfo.updateTaskState(taskState);
        return taskInfo;
    }
}
