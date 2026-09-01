package org.flexlb.balance.resource;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link PrefillResourceMeasure}.
 *
 * <p>Since {@code isResourceAvailable(PrefillEndpoint)} depends on endpoint-level
 * state ({@code realPendingCount()}), it is tested in integration tests
 * ({@code FlexlbBatchSchedulerTest}). This unit test focuses on
 * {@code calculateAverageWaterLevel} using {@code WorkerStatus.runningTaskList}
 * with {@link TaskPhase} to distinguish running vs waiting tasks.
 */
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
    void pending_and_received_tasks_contribute_to_water_level() {
        // Non-RUNNING tasks (PENDING, RECEIVED, KV_ALLOCATED) are counted as waiting.
        // With maxQueueSize=10, 3 waiting tasks → water level = 30%
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.PENDING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RECEIVED));
        runningTaskList.put("3", taskInfo(3L, TaskPhase.KV_ALLOCATED));
        worker.setRunningTaskList(runningTaskList);

        assertEquals(30.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void running_tasks_do_not_count_as_prefill_queue() {
        // Only RUNNING tasks → water level = 0%
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.RUNNING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RUNNING));
        worker.setRunningTaskList(runningTaskList);

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_counts_all_non_running_tasks_from_engine_reported_list() {
        // Engine reports a unified runningTaskList;
        // tasks with phase != RUNNING are counted as waiting.
        // PENDING + RECEIVED + KV_ALLOCATED = 3 waiting → 30% with maxQueueSize=10
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.PENDING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RECEIVED));
        runningTaskList.put("3", taskInfo(3L, TaskPhase.KV_ALLOCATED));
        runningTaskList.put("4", taskInfo(4L, TaskPhase.RUNNING));
        worker.setRunningTaskList(runningTaskList);

        // 3 waiting out of max 10 = 30%
        assertEquals(30.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_capped_at_100_when_queue_full() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        for (int i = 1; i <= 12; i++) {
            runningTaskList.put(String.valueOf(i), taskInfo(i, TaskPhase.PENDING));
        }
        worker.setRunningTaskList(runningTaskList);

        // 12 waiting > maxQueueSize=10 → capped at 100%
        assertEquals(100.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void empty_task_list_gives_zero_water_level() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        worker.setRunningTaskList(new HashMap<>());

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void null_task_list_gives_zero_water_level() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        worker.setRunningTaskList(null);

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void all_running_tasks_gives_zero_water_level() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.RUNNING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RUNNING));
        runningTaskList.put("3", taskInfo(3L, TaskPhase.RUNNING));
        worker.setRunningTaskList(runningTaskList);

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    private WorkerStatus createAlivePrefillWorker() {
        WorkerStatus worker = new WorkerStatus();
        worker.setAlive(true);
        worker.setRole(RoleType.PREFILL);
        return worker;
    }

    private TaskInfo taskInfo(long requestId, TaskPhase phase) {
        TaskInfo taskInfo = new TaskInfo();
        taskInfo.setRequestId(requestId);
        taskInfo.setPhase(phase);
        return taskInfo;
    }
}
