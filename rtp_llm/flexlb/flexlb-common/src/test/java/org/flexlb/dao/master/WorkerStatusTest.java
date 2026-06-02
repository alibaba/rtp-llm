package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskStateEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@DisplayName("WorkerStatus Hysteresis Tests")
class WorkerStatusTest {

    private WorkerStatus workerStatus;

    @BeforeEach
    void setUp() {
        workerStatus = new WorkerStatus();
        // Default state: available
        workerStatus.getResourceAvailable().set(true);
    }

    @ParameterizedTest
    @CsvSource({
            // currentState, currentMetric, upperThreshold, hysteresisBias, expectedResult
            "true,  50,  100, 20, true",   // Available, below threshold
            "true,  80,  100, 20, true",   // Available, at lower threshold
            "true,  90,  100, 20, true",   // Available, in hysteresis band
            "true,  99,  100, 20, true",   // Available, just below threshold
            "true,  100, 100, 20, false",  // Available, at threshold
            "true,  110, 100, 20, false",  // Available, above threshold
            "false, 50,  100, 20, true",   // Unavailable, below lower threshold
            "false, 80,  100, 20, true",   // Unavailable, at lower threshold
            "false, 81,  100, 20, false",  // Unavailable, just above lower threshold
            "false, 90,  100, 20, false",  // Unavailable, in hysteresis band
            "false, 99,  100, 20, false",  // Unavailable, below but above lower threshold
            "false, 100, 100, 20, false",  // Unavailable, at threshold
            "false, 110, 100, 20, false",  // Unavailable, above threshold
    })
    @DisplayName("Parameterized hysteresis behavior")
    void hysteresisParameterized(
            boolean currentState,
            long currentMetric,
            long upperThreshold,
            int hysteresisBias,
            boolean expectedResult
    ) {
        workerStatus.getResourceAvailable().set(currentState);
        boolean result =
                workerStatus.updateResourceAvailabilityWithHysteresis(currentMetric, upperThreshold, hysteresisBias);
        assertEquals(
                expectedResult, result, String.format(
                        "State=%s, metric=%d, threshold=%d, bias=%d",
                        currentState,
                        currentMetric,
                        upperThreshold,
                        hysteresisBias
                )
        );
    }

    @Nested
    @DisplayName("When resource is AVAILABLE")
    class AvailableStateTests {

        @BeforeEach
        void setAvailableState() {
            workerStatus.getResourceAvailable().set(true);
        }

        @Test
        @DisplayName("Should remain AVAILABLE when metric below upper threshold")
        void shouldRemainAvailableBelowThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(90, threshold, hysteresisBias);

            assertTrue(result, "Should remain available when metric is below threshold");
            assertTrue(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should remain AVAILABLE when metric equals upper threshold (exclusive)")
        void shouldRemainAvailableAtThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(99, threshold, hysteresisBias);

            assertTrue(result, "Should remain available when metric is just below threshold");
            assertTrue(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should become UNAVAILABLE when metric reaches upper threshold")
        void shouldBecomeUnavailableAtThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(100, threshold, hysteresisBias);

            assertFalse(result, "Should become unavailable when metric reaches threshold");
            assertFalse(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should become UNAVAILABLE when metric exceeds upper threshold")
        void shouldBecomeUnavailableAboveThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(110, threshold, hysteresisBias);

            assertFalse(result, "Should become unavailable when metric exceeds threshold");
            assertFalse(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should remain AVAILABLE when metric is very low")
        void shouldRemainAvailableVeryLow() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(10, threshold, hysteresisBias);

            assertTrue(result, "Should remain available when metric is very low");
            assertTrue(workerStatus.getResourceAvailable().get());
        }
    }

    @Nested
    @DisplayName("When resource is UNAVAILABLE")
    class UnavailableStateTests {

        @BeforeEach
        void setUnavailableState() {
            workerStatus.getResourceAvailable().set(false);
        }

        @Test
        @DisplayName("Should become AVAILABLE when metric falls below lower threshold")
        void shouldBecomeAvailableBelowLowerThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(70, threshold, hysteresisBias);

            assertTrue(result, "Should become available when metric is below lower threshold");
            assertTrue(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should become AVAILABLE when metric equals lower threshold")
        void shouldBecomeAvailableAtLowerThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(80, threshold, hysteresisBias);

            assertTrue(result, "Should become available when metric equals lower threshold");
            assertTrue(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should remain UNAVAILABLE when metric in hysteresis band")
        void shouldRemainUnavailableInHysteresisBand() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80, band = 80-100

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(90, threshold, hysteresisBias);

            assertFalse(result, "Should remain unavailable when metric is in hysteresis band");
            assertFalse(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should remain UNAVAILABLE when metric exceeds upper threshold")
        void shouldRemainUnavailableAboveThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(110, threshold, hysteresisBias);

            assertFalse(result, "Should remain unavailable when metric exceeds threshold");
            assertFalse(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Should remain UNAVAILABLE when metric is just above lower threshold")
        void shouldRemainUnavailableAboveLowerThreshold() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(81, threshold, hysteresisBias);

            assertFalse(result, "Should remain unavailable when metric is just above lower threshold");
            assertFalse(workerStatus.getResourceAvailable().get());
        }
    }

    @Nested
    @DisplayName("Hysteresis behavior - state transitions")
    class HysteresisTransitionTests {

        @Test
        @DisplayName("Full cycle: AVAILABLE -> UNAVAILABLE -> AVAILABLE hysteresis prevents oscillation")
        void fullCycleHysteresis() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80, band = 80-100

            // Start: AVAILABLE
            assertTrue(workerStatus.getResourceAvailable().get());

            // 1. Metric in hysteresis band (85) - should remain AVAILABLE
            boolean result1 = workerStatus.updateResourceAvailabilityWithHysteresis(85, threshold, hysteresisBias);
            assertTrue(result1);
            assertTrue(workerStatus.getResourceAvailable().get());

            // 2. Metric exceeds threshold (105) - should become UNAVAILABLE
            boolean result2 = workerStatus.updateResourceAvailabilityWithHysteresis(105, threshold, hysteresisBias);
            assertFalse(result2);
            assertFalse(workerStatus.getResourceAvailable().get());

            // 3. Metric in hysteresis band (85) - should remain UNAVAILABLE
            boolean result3 = workerStatus.updateResourceAvailabilityWithHysteresis(85, threshold, hysteresisBias);
            assertFalse(result3);
            assertFalse(workerStatus.getResourceAvailable().get());

            // 4. Metric below lower threshold (75) - should become AVAILABLE
            boolean result4 = workerStatus.updateResourceAvailabilityWithHysteresis(75, threshold, hysteresisBias);
            assertTrue(result4);
            assertTrue(workerStatus.getResourceAvailable().get());
        }

        @Test
        @DisplayName("Hysteresis band prevents rapid state toggling")
        void hysteresisPreventsToggling() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80, band = 80-100

            // Start: AVAILABLE
            assertTrue(workerStatus.getResourceAvailable().get());

            // Simulate metric fluctuating in hysteresis band
            for (long metric = 85; metric <= 95; metric++) {
                boolean result =
                        workerStatus.updateResourceAvailabilityWithHysteresis(metric, threshold, hysteresisBias);
                assertTrue(result, "Should remain AVAILABLE for metric " + metric);
                assertTrue(workerStatus.getResourceAvailable().get());
            }

            // Now exceed threshold
            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(100, threshold, hysteresisBias);
            assertFalse(result);
            assertFalse(workerStatus.getResourceAvailable().get());

            // Simulate metric fluctuating in hysteresis band
            for (long metric = 85; metric <= 95; metric++) {
                result = workerStatus.updateResourceAvailabilityWithHysteresis(metric, threshold, hysteresisBias);
                assertFalse(result, "Should remain UNAVAILABLE for metric " + metric);
                assertFalse(workerStatus.getResourceAvailable().get());
            }
        }
    }

    @Nested
    @DisplayName("Edge cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("Should handle zero hysteresis bias (no hysteresis)")
        void zeroHysteresisBias() {
            long threshold = 100;
            int hysteresisBias = 0; // lower = 100, no hysteresis band

            workerStatus.getResourceAvailable().set(true);
            boolean result1 = workerStatus.updateResourceAvailabilityWithHysteresis(100, threshold, hysteresisBias);
            assertFalse(result1);

            boolean result2 = workerStatus.updateResourceAvailabilityWithHysteresis(99, threshold, hysteresisBias);
            assertTrue(result2); // Should toggle immediately
        }

        @Test
        @DisplayName("Should handle large hysteresis bias")
        void largeHysteresisBias() {
            long threshold = 100;
            int hysteresisBias = 50; // lower = 50

            workerStatus.getResourceAvailable().set(true);
            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(100, threshold, hysteresisBias);
            assertFalse(result);

            // Metric at 75 is in hysteresis band (50-100)
            result = workerStatus.updateResourceAvailabilityWithHysteresis(75, threshold, hysteresisBias);
            assertFalse(result);
        }

        @Test
        @DisplayName("Should handle lower threshold clamped to zero")
        void lowerThresholdClampedToZero() {
            long threshold = 10;
            int hysteresisBias = 200; // lower = -10, but clamped to 0

            workerStatus.getResourceAvailable().set(false);
            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(0, threshold, hysteresisBias);
            assertTrue(result, "Should become available at metric 0 when lower threshold is clamped to 0");
        }

        @Test
        @DisplayName("Should handle zero threshold")
        void zeroThreshold() {
            long threshold = 0;
            int hysteresisBias = 20; // lower = 0

            workerStatus.getResourceAvailable().set(true);
            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(0, threshold, hysteresisBias);
            assertFalse(result, "Should become unavailable when metric equals zero threshold");
        }

        @Test
        @DisplayName("Should handle zero current metric")
        void zeroCurrentMetric() {
            long threshold = 100;
            int hysteresisBias = 20; // lower = 80

            workerStatus.getResourceAvailable().set(false);
            boolean result = workerStatus.updateResourceAvailabilityWithHysteresis(0, threshold, hysteresisBias);
            assertTrue(result, "Should become available when metric is zero");
        }
    }

    @Nested
    @DisplayName("updateTaskStates - waiting task handling")
    class UpdateTaskStatesTests {

        private static final Long REQUEST_ID = 1000L;

        @BeforeEach
        void setUpWorkerStatus() {
            workerStatus.setRole(RoleType.PREFILL.getCode());
        }

        @Test
        @DisplayName("Task in waiting list only: IN_TRANSIT becomes CONFIRMED and fields updated from waiting task")
        void taskInWaitingOnly_shouldBecomeConfirmedAndSyncFields() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(200);
            localTask.setPrefixLength(0);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo waitingTask = new TaskInfo();
            waitingTask.setRequestId(REQUEST_ID);
            waitingTask.setPrefixLength(50);
            waitingTask.setInputLength(200);
            waitingTask.setWaitingTime(100);
            waitingTask.setDpRank(1);
            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), waitingTask);

            workerStatus.updateTaskStates(waitingTaskInfo, new HashMap<>(), new HashMap<>());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated, "Task should remain in local map");
            assertEquals(TaskStateEnum.CONFIRMED, updated.getTaskState());
            assertEquals(50, updated.getPrefixLength());
            assertEquals(200, updated.getInputLength());
            assertEquals(100, updated.getWaitingTime());
            assertEquals(1, updated.getDpRank());
        }

        @Test
        @DisplayName("Task in waiting list with null running and finished maps should not NPE")
        void taskInWaitingWithNullMaps_shouldNotThrow() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), new TaskInfo());

            workerStatus.updateTaskStates(waitingTaskInfo, null, null);

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated);
            assertEquals(TaskStateEnum.CONFIRMED, updated.getTaskState());
        }

        @Test
        @DisplayName("Task CONFIRMED but not in waiting/running/finished should be marked LOST")
        void taskConfirmedButNotInAnyList_shouldBeMarkedLost() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.updateTaskState(TaskStateEnum.CONFIRMED);
            workerStatus.getLocalTaskMap().put(REQUEST_ID, localTask);

            workerStatus.updateTaskStates(new HashMap<>(), new HashMap<>(), new HashMap<>());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated);
            assertTrue(updated.isLost());
        }

        @Test
        @DisplayName("Task in finished list should be removed from local map")
        void taskInFinishedList_shouldBeRemoved() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(100);
            localTask.setPrefixLength(0);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo finishedTask = new TaskInfo();
            finishedTask.setRequestId(REQUEST_ID);
            finishedTask.setEndTimeMs(System.currentTimeMillis());
            Map<String, TaskInfo> finishedTaskInfo = new HashMap<>();
            finishedTaskInfo.put(String.valueOf(REQUEST_ID), finishedTask);

            workerStatus.updateTaskStates(new HashMap<>(), new HashMap<>(), finishedTaskInfo);

            assertNull(workerStatus.getLocalTaskMap().get(REQUEST_ID));
        }

        @Test
        @DisplayName("Task in running list should become RUNNING and sync fields")
        void taskInRunningList_shouldBecomeRunningAndSyncFields() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setPrefixLength(100);
            runningTask.setInputLength(200);
            runningTask.setPrefillTime(50);
            runningTask.setIterateCount(2);
            runningTask.setEndTimeMs(12345L);
            runningTask.setDpRank(0);
            Map<String, TaskInfo> runningTaskInfo = new HashMap<>();
            runningTaskInfo.put(String.valueOf(REQUEST_ID), runningTask);

            workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated);
            assertEquals(TaskStateEnum.RUNNING, updated.getTaskState());
            assertEquals(100, updated.getPrefixLength());
            assertEquals(200, updated.getInputLength());
            assertEquals(50, updated.getPrefillTime());
            assertEquals(2, updated.getIterateCount());
            assertEquals(12345L, updated.getEndTimeMs());
        }

        @Test
        @DisplayName("Task in waiting then in running on next call should be RUNNING")
        void taskInWaitingThenInRunning_shouldBeRunning() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), new TaskInfo());
            workerStatus.updateTaskStates(waitingTaskInfo, new HashMap<>(), new HashMap<>());
            assertEquals(TaskStateEnum.CONFIRMED, workerStatus.getLocalTaskMap().get(REQUEST_ID).getTaskState());

            Map<String, TaskInfo> runningTaskInfo = new HashMap<>();
            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTaskInfo.put(String.valueOf(REQUEST_ID), runningTask);
            workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());

            assertEquals(TaskStateEnum.RUNNING, workerStatus.getLocalTaskMap().get(REQUEST_ID).getTaskState());
        }

        @Test
        @DisplayName("Finished takes precedence over waiting when task in both")
        void taskInFinishedAndWaiting_shouldBeRemovedAsFinished() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo finishedTask = new TaskInfo();
            finishedTask.setRequestId(REQUEST_ID);
            finishedTask.setEndTimeMs(1);
            TaskInfo waitingTask = new TaskInfo();
            waitingTask.setRequestId(REQUEST_ID);
            Map<String, TaskInfo> finishedTaskInfo = new HashMap<>();
            finishedTaskInfo.put(String.valueOf(REQUEST_ID), finishedTask);
            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), waitingTask);

            workerStatus.updateTaskStates(waitingTaskInfo, new HashMap<>(), finishedTaskInfo);

            assertNull(workerStatus.getLocalTaskMap().get(REQUEST_ID));
        }
    }
}