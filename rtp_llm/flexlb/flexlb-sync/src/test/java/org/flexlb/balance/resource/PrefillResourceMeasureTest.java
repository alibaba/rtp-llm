package org.flexlb.balance.resource;

import org.flexlb.balance.endpoint.PrefillEndpoint;
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Tests for {@link PrefillResourceMeasure}.
 *
 * <p>The generic availability path consumes the endpoint's churn-safe admission
 * count. Projection-aware callers use the explicit-count overload so their
 * availability decision and TTFT share one coherent observation.
 */
@ExtendWith(MockitoExtension.class)
class PrefillResourceMeasureTest {

    @Mock
    private ConfigService configService;

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(2);
        config.getRouter().setAvailabilityHysteresisPercent(50);
        when(configService.loadBalanceConfig()).thenReturn(config);
    }

    @Test
    void coherentPendingCountDrivesAvailabilityWithoutReadingEndpointAgain() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);

        assertFalse(measure.isResourceAvailable(2L),
                "the coherent count at the upper threshold must close admission");
        assertTrue(measure.isResourceAvailable(1L),
                "the coherent count at the lower threshold must reopen admission");
        assertThrows(IllegalArgumentException.class,
                () -> measure.isResourceAvailable(-1L));
    }

    @Test
    void callerPassesTheEndpointChurnSafeAdmissionCount() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.admissionPendingRequestCount()).thenReturn(1L);

        assertTrue(measure.isResourceAvailable(
                endpoint.admissionPendingRequestCount()));
        verify(endpoint).admissionPendingRequestCount();
    }

    @Test
    void pending_and_received_tasks_contribute_to_water_level() {
        // Non-RUNNING tasks (PENDING, RECEIVED, KV_ALLOCATED) are counted as waiting.
        // With the internal saturation point at 20, 3 waiting tasks → water level = 15%
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.PENDING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RECEIVED));
        runningTaskList.put("3", taskInfo(3L, TaskPhase.KV_ALLOCATED));
        publishTasks(worker, runningTaskList);

        assertEquals(15.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void running_tasks_do_not_count_as_prefill_queue() {
        // Only RUNNING tasks → water level = 0%
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.RUNNING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RUNNING));
        publishTasks(worker, runningTaskList);

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_counts_all_non_running_tasks_from_engine_reported_list() {
        // Engine reports a unified runningTaskList;
        // tasks with phase != RUNNING are counted as waiting.
        // PENDING + RECEIVED + KV_ALLOCATED = 3 waiting → 15% at the internal saturation point of 20
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        runningTaskList.put("1", taskInfo(1L, TaskPhase.PENDING));
        runningTaskList.put("2", taskInfo(2L, TaskPhase.RECEIVED));
        runningTaskList.put("3", taskInfo(3L, TaskPhase.KV_ALLOCATED));
        runningTaskList.put("4", taskInfo(4L, TaskPhase.RUNNING));
        publishTasks(worker, runningTaskList);

        // 3 waiting out of the saturation point of 20 = 15%
        assertEquals(15.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void water_level_capped_at_100_when_queue_full() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        Map<String, TaskInfo> runningTaskList = new HashMap<>();
        for (int i = 1; i <= 24; i++) {
            runningTaskList.put(String.valueOf(i), taskInfo(i, TaskPhase.PENDING));
        }
        publishTasks(worker, runningTaskList);

        // 24 waiting > the internal saturation point of 20 → capped at 100%
        assertEquals(100.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void empty_task_list_gives_zero_water_level() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        publishTasks(worker, new HashMap<>());

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    @Test
    void null_task_list_gives_zero_water_level() {
        PrefillResourceMeasure measure = new PrefillResourceMeasure(configService);
        WorkerStatus worker = createAlivePrefillWorker();
        publishTasks(worker, null);

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
        publishTasks(worker, runningTaskList);

        assertEquals(0.0, measure.calculateAverageWaterLevel(Map.of("worker", worker)));
    }

    private WorkerStatus createAlivePrefillWorker() {
        return ResourceTestSupport.worker(
                RoleType.PREFILL, 0L, 0L, Map.of());
    }

    private void publishTasks(
            WorkerStatus worker, Map<String, TaskInfo> tasks) {
        ResourceTestSupport.publish(
                worker, true, 0L, 0L, tasks);
    }

    private TaskInfo taskInfo(long requestId, TaskPhase phase) {
        TaskInfo taskInfo = new TaskInfo();
        taskInfo.setRequestId(requestId);
        taskInfo.setPhase(phase);
        return taskInfo;
    }
}
