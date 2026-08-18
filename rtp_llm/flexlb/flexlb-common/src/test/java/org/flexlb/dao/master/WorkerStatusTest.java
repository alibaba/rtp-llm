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

    @Test
    @DisplayName("In-transit and waiting stats include local tasks but exclude running tasks")
    void inTransitAndWaitingStats_includeInTransitAndConfirmed_excludeRunning() {
        TaskInfo inTransitTask = pendingTask("in-transit", 48_000, 16_000);
        TaskInfo waitingTask = pendingTask("waiting", 20_000, 5_000);
        TaskInfo runningTask = pendingTask("running", 30_000, 0);
        workerStatus.putLocalTask("in-transit", inTransitTask);
        workerStatus.putLocalTask("waiting", waitingTask);
        workerStatus.putLocalTask("running", runningTask);

        TaskInfo engineWaitingTask = new TaskInfo();
        engineWaitingTask.setRequestId("waiting");
        engineWaitingTask.setInputLength(20_000);
        engineWaitingTask.setPrefixLengthValid(false);
        TaskInfo engineRunningTask = new TaskInfo();
        engineRunningTask.setRequestId("running");
        engineRunningTask.setInputLength(30_000);

        workerStatus.updateTaskStates(
                Map.of("waiting", engineWaitingTask),
                Map.of("running", engineRunningTask),
                Map.of());

        assertEquals(2, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(47_000, workerStatus.getInTransitAndWaitingUncachedTokens());
    }

    @Test
    @DisplayName("In-transit and waiting stats include tasks first observed in worker status")
    void inTransitAndWaitingStats_includeWorkerObservedTasks() {
        TaskInfo engineTaskOne = pendingTask("engine-one", 4_000, 0);
        engineTaskOne.setPrefixLengthValid(true);
        TaskInfo engineTaskTwo = pendingTask("engine-two", 64_000, 0);
        engineTaskTwo.setPrefixLengthValid(true);
        TaskInfo engineTaskThree = pendingTask("engine-three", 64_000, 0);
        engineTaskThree.setPrefixLengthValid(true);
        TaskInfo overlappingLocalTask = pendingTask("engine-one", 4_000, 0);
        TaskInfo largeLocalTask = pendingTask("local-large", 48_000, 0);
        workerStatus.putLocalTask("engine-one", overlappingLocalTask);
        workerStatus.putLocalTask("local-large", largeLocalTask);
        workerStatus.updateTaskStates(Map.of(
                "engine-one", engineTaskOne,
                "engine-two", engineTaskTwo,
                "engine-three", engineTaskThree), Map.of(), Map.of());

        assertEquals(4, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(180_000, workerStatus.getInTransitAndWaitingUncachedTokens());
        assertEquals(TaskStateEnum.CONFIRMED,
                workerStatus.getLocalTaskMap().get("engine-two").getTaskState());
    }

    @Test
    @DisplayName("Running task first observed in worker status is added once to local load")
    void runningStats_includeWorkerObservedTaskOnce() {
        workerStatus.setRole(RoleType.PREFILL.name());
        TaskInfo engineRunningTask = pendingTask("engine-running", 64_000, 16_000);
        engineRunningTask.setRemainingPrefillTokens(32_000);

        workerStatus.updateTaskStates(
                Map.of(), Map.of("engine-running", engineRunningTask), Map.of());
        workerStatus.updateTaskStates(
                Map.of(), Map.of("engine-running", engineRunningTask), Map.of());

        assertEquals(1, workerStatus.getLocalTaskMap().size());
        assertEquals(TaskStateEnum.RUNNING,
                workerStatus.getLocalTaskMap().get("engine-running").getTaskState());
        assertEquals(48_000, workerStatus.getRunningQueueTime().get());
        assertEquals(32_000, workerStatus.getRunningRemainingPrefillTokens());
    }

    @Test
    @DisplayName("Pending queue uses local prediction until engine hit is valid and removes engine-running task")
    void inTransitAndWaitingStats_fallBackToPredictionAndExcludeEngineRunning() {
        TaskInfo localTask = pendingTask("request", 64_000, 48_000);
        workerStatus.putLocalTask("request", localTask);

        TaskInfo engineWaitingTask = new TaskInfo();
        engineWaitingTask.setRequestId("request");
        engineWaitingTask.setInputLength(64_000);
        engineWaitingTask.setPrefixLengthValid(false);
        workerStatus.updateTaskStates(Map.of("request", engineWaitingTask), Map.of(), Map.of());

        assertEquals(1, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(16_000, workerStatus.getInTransitAndWaitingUncachedTokens());

        TaskInfo engineRunningTask = new TaskInfo();
        engineRunningTask.setRequestId("request");
        workerStatus.updateTaskStates(Map.of(), Map.of("request", engineRunningTask), Map.of());

        assertEquals(0, workerStatus.getInTransitAndWaitingTaskCount());
        assertEquals(0, workerStatus.getInTransitAndWaitingUncachedTokens());
    }

    private static TaskInfo pendingTask(String requestId, long inputLength, long predictedPrefixLength) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setInputLength(inputLength);
        task.setPrefixLength(predictedPrefixLength);
        task.setPredictedPrefixLength(predictedPrefixLength);
        task.setPrefixLengthValid(false);
        return task;
    }

    @Nested
    @DisplayName("updateTaskStates - waiting task handling")
    class UpdateTaskStatesTests {

        private static final String REQUEST_ID = "request-1000";

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
            localTask.setLastActiveTimeUs(System.nanoTime() / 1000 - 5_000);

            TaskInfo waitingTask = new TaskInfo();
            waitingTask.setRequestId(REQUEST_ID);
            waitingTask.setPrefixLength(50);
            waitingTask.setPrefixLengthValid(true);
            waitingTask.setInputLength(200);
            waitingTask.setWaitingTime(100);
            waitingTask.setDpRank(1);
            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), waitingTask);

            var updateResult = workerStatus.updateTaskStates(
                    waitingTaskInfo, new HashMap<>(), new HashMap<>());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated, "Task should remain in local map");
            assertEquals(TaskStateEnum.CONFIRMED, updated.getTaskState());
            assertEquals(50, updated.getPrefixLength());
            assertEquals(200, updated.getInputLength());
            assertEquals(100, updated.getWaitingTime());
            assertEquals(1, updated.getDpRank());
            assertEquals(1, updateResult.decisionToWaitingObservedLatenciesMs().size());
            assertTrue(updateResult.decisionToWaitingObservedLatenciesMs().getFirst() >= 5);

            var repeatedUpdateResult = workerStatus.updateTaskStates(
                    waitingTaskInfo, new HashMap<>(), new HashMap<>());
            assertTrue(repeatedUpdateResult.decisionToWaitingObservedLatenciesMs().isEmpty());
        }

        @Test
        @DisplayName("Waiting task keeps KVCM prediction until engine cache hit is valid")
        void waitingTaskWithInvalidPrefix_shouldKeepPrediction() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(200);
            localTask.setPrefixLength(100);
            localTask.setPredictedPrefixLength(100);
            localTask.setCacheMatchSource("KVCM");
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo waitingTask = new TaskInfo();
            waitingTask.setRequestId(REQUEST_ID);
            waitingTask.setInputLength(200);
            waitingTask.setPrefixLength(0);
            waitingTask.setPrefixLengthValid(false);

            var updateResult = workerStatus.updateTaskStates(
                    Map.of(REQUEST_ID, waitingTask), Map.of(), Map.of());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertEquals(100, updated.getPrefixLength());
            assertFalse(updated.isPrefixLengthValid());
            assertTrue(updateResult.cacheHitFeedbacks().isEmpty());
        }

        @Test
        @DisplayName("Valid engine cache hit corrects queue estimate and produces one comparison")
        void validEnginePrefix_shouldCorrectQueueAndProduceComparisonOnce() {
            workerStatus.setIp("127.0.0.1");
            workerStatus.setPort(8080);
            workerStatus.setGroup("default");

            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(200);
            localTask.setPrefixLength(100);
            localTask.setPredictedPrefixLength(100);
            localTask.setCacheMatchSource("KVCM");
            workerStatus.putLocalTask(REQUEST_ID, localTask);
            long predictedQueueTime = TaskInfo.estimatePrefillTimeMs(200, 100);
            assertEquals(100, predictedQueueTime);

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setInputLength(200);
            runningTask.setPrefixLength(120);
            runningTask.setPrefixLengthValid(true);

            var firstUpdateResult = workerStatus.updateTaskStates(
                    Map.of(), Map.of(REQUEST_ID, runningTask), Map.of());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertEquals(120, updated.getPrefixLength());
            assertTrue(updated.isPrefixLengthValid());
            assertEquals(80, workerStatus.getRunningQueueTime().get());
            assertTrue(workerStatus.getRunningQueueTime().get() < predictedQueueTime);
            assertEquals(1, firstUpdateResult.cacheHitFeedbacks().size());
            assertEquals(100, firstUpdateResult.cacheHitFeedbacks().getFirst().predictedHitTokens());
            assertEquals(120, firstUpdateResult.cacheHitFeedbacks().getFirst().actualHitTokens());
            assertEquals(20, firstUpdateResult.cacheHitFeedbacks().getFirst().deltaHitTokens());

            var repeatedUpdateResult = workerStatus.updateTaskStates(
                    Map.of(), Map.of(REQUEST_ID, runningTask), Map.of());
            assertTrue(repeatedUpdateResult.cacheHitFeedbacks().isEmpty());
        }

        @Test
        @DisplayName("Lower actual cache hit increases the queue estimate")
        void lowerActualPrefix_shouldIncreaseQueueEstimate() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(200);
            localTask.setPrefixLength(120);
            localTask.setPredictedPrefixLength(120);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setInputLength(200);
            runningTask.setPrefixLength(100);
            runningTask.setPrefixLengthValid(true);

            workerStatus.updateTaskStates(
                    Map.of(), Map.of(REQUEST_ID, runningTask), Map.of());

            assertEquals(TaskInfo.estimatePrefillTimeMs(200, 100),
                    workerStatus.getRunningQueueTime().get());
        }

        @Test
        @DisplayName("Cache hit is compared again after a preempted task prepares KV again")
        void preemptedTask_shouldCompareCacheHitAfterKvIsReadyAgain() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(200);
            localTask.setPrefixLength(100);
            localTask.setPredictedPrefixLength(100);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo firstRunningTask = new TaskInfo();
            firstRunningTask.setInputLength(200);
            firstRunningTask.setPrefixLength(120);
            firstRunningTask.setPrefixLengthValid(true);
            assertEquals(1, workerStatus.updateTaskStates(
                    Map.of(), Map.of(REQUEST_ID, firstRunningTask), Map.of())
                    .cacheHitFeedbacks().size());

            TaskInfo preemptedWaitingTask = new TaskInfo();
            preemptedWaitingTask.setInputLength(200);
            workerStatus.updateTaskStates(
                    Map.of(REQUEST_ID, preemptedWaitingTask), Map.of(), Map.of());
            assertFalse(localTask.isPrefixLengthValid());
            assertEquals(100, localTask.getPrefixLength());

            TaskInfo resumedRunningTask = new TaskInfo();
            resumedRunningTask.setInputLength(200);
            resumedRunningTask.setPrefixLength(80);
            resumedRunningTask.setPrefixLengthValid(true);
            var resumedUpdateResult = workerStatus.updateTaskStates(
                    Map.of(), Map.of(REQUEST_ID, resumedRunningTask), Map.of());

            assertEquals(1, resumedUpdateResult.cacheHitFeedbacks().size());
            assertEquals(80, resumedUpdateResult.cacheHitFeedbacks().getFirst().actualHitTokens());
            assertEquals(TaskInfo.estimatePrefillTimeMs(200, 80),
                    workerStatus.getRunningQueueTime().get());
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
        @DisplayName("Fast finished task compares and corrects cache hit before removal")
        void fastFinishedTask_shouldCompareCacheHitBeforeRemoval() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(200);
            localTask.setPrefixLength(100);
            localTask.setPredictedPrefixLength(100);
            localTask.setCacheMatchSource("KVCM");
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo finishedTask = new TaskInfo();
            finishedTask.setRequestId(REQUEST_ID);
            finishedTask.setInputLength(200);
            finishedTask.setPrefixLength(120);
            finishedTask.setPrefixLengthValid(true);

            var updateResult = workerStatus.updateTaskStates(
                    Map.of(), Map.of(), Map.of(REQUEST_ID, finishedTask));

            assertNull(workerStatus.getLocalTaskMap().get(REQUEST_ID));
            assertEquals(0, workerStatus.getRunningQueueTime().get());
            assertEquals(1, updateResult.cacheHitFeedbacks().size());
            assertEquals("finished", updateResult.cacheHitFeedbacks().getFirst().taskState());
            assertEquals(100, updateResult.cacheHitFeedbacks().getFirst().predictedHitTokens());
            assertEquals(120, updateResult.cacheHitFeedbacks().getFirst().actualHitTokens());
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
            runningTask.setPrefixLengthValid(true);
            runningTask.setInputLength(200);
            runningTask.setPrefillTime(50);
            runningTask.setIterateCount(2);
            runningTask.setEndTimeMs(12345L);
            runningTask.setDpRank(0);
            Map<String, TaskInfo> runningTaskInfo = new HashMap<>();
            runningTaskInfo.put(String.valueOf(REQUEST_ID), runningTask);

            var updateResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated);
            assertEquals(TaskStateEnum.RUNNING, updated.getTaskState());
            assertEquals(100, updated.getPrefixLength());
            assertEquals(200, updated.getInputLength());
            assertEquals(50, updated.getPrefillTime());
            assertEquals(2, updated.getIterateCount());
            assertEquals(12345L, updated.getEndTimeMs());
            assertTrue(updateResult.waitingToRunningObservedLatenciesMs().isEmpty());
            assertTrue(updateResult.engineWaitingToRunningLatenciesMs().isEmpty());
        }

        @Test
        @DisplayName("RUNNING progress overwrites remaining work without changing pending aggregates")
        void runningProgress_shouldOverwriteSeparateRemainingWorkAggregate() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(64_000);
            localTask.setPredictedPrefixLength(0);
            workerStatus.putLocalTask(REQUEST_ID, localTask);
            assertEquals(1, workerStatus.getInTransitAndWaitingTaskCount());
            assertEquals(64_000, workerStatus.getInTransitAndWaitingUncachedTokens());

            TaskInfo initialRunning = new TaskInfo();
            initialRunning.setRequestId(REQUEST_ID);
            initialRunning.setInputLength(64_000);
            initialRunning.setPrefixLengthValid(true);
            initialRunning.setCompletedPrefillTokens(0);
            initialRunning.setRemainingPrefillTokens(64_000);
            initialRunning.setLastCompletedPrefillStepId(0);
            workerStatus.updateTaskStates(Map.of(), Map.of(REQUEST_ID, initialRunning), Map.of());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated);
            assertEquals(TaskStateEnum.RUNNING, updated.getTaskState());
            assertEquals(0, updated.getCompletedPrefillTokens());
            assertEquals(64_000, updated.getRemainingPrefillTokens());
            assertEquals(0, updated.getLastCompletedPrefillStepId());
            assertEquals(64_000, workerStatus.getRunningRemainingPrefillTokens());
            assertEquals(0, workerStatus.getInTransitAndWaitingTaskCount());
            assertEquals(0, workerStatus.getInTransitAndWaitingUncachedTokens());

            TaskInfo firstStep = new TaskInfo();
            firstStep.setRequestId(REQUEST_ID);
            firstStep.setInputLength(64_000);
            firstStep.setPrefixLengthValid(true);
            firstStep.setCompletedPrefillTokens(16_384);
            firstStep.setRemainingPrefillTokens(47_616);
            firstStep.setLastCompletedPrefillStepId(1);
            workerStatus.updateTaskStates(Map.of(), Map.of(REQUEST_ID, firstStep), Map.of());

            updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertNotNull(updated);
            assertEquals(TaskStateEnum.RUNNING, updated.getTaskState());
            assertEquals(16_384, updated.getCompletedPrefillTokens());
            assertEquals(47_616, updated.getRemainingPrefillTokens());
            assertEquals(1, updated.getLastCompletedPrefillStepId());
            assertEquals(47_616, workerStatus.getRunningRemainingPrefillTokens());
            assertEquals(0, workerStatus.getInTransitAndWaitingTaskCount());
            assertEquals(0, workerStatus.getInTransitAndWaitingUncachedTokens());

            TaskInfo secondStep = new TaskInfo();
            secondStep.setRequestId(REQUEST_ID);
            secondStep.setInputLength(64_000);
            secondStep.setPrefixLengthValid(true);
            secondStep.setCompletedPrefillTokens(32_768);
            secondStep.setRemainingPrefillTokens(31_232);
            secondStep.setLastCompletedPrefillStepId(2);
            workerStatus.updateTaskStates(Map.of(), Map.of(REQUEST_ID, secondStep), Map.of());

            assertEquals(31_232, workerStatus.getRunningRemainingPrefillTokens());
            assertEquals(0, workerStatus.getInTransitAndWaitingTaskCount());
            assertEquals(0, workerStatus.getInTransitAndWaitingUncachedTokens());

            TaskInfo preemptedWaiting = new TaskInfo();
            preemptedWaiting.setRequestId(REQUEST_ID);
            preemptedWaiting.setInputLength(64_000);
            workerStatus.updateTaskStates(Map.of(REQUEST_ID, preemptedWaiting), Map.of(), Map.of());

            assertEquals(TaskStateEnum.CONFIRMED,
                    workerStatus.getLocalTaskMap().get(REQUEST_ID).getTaskState());
            assertEquals(0, workerStatus.getRunningRemainingPrefillTokens());
            assertEquals(1, workerStatus.getInTransitAndWaitingTaskCount());
            assertEquals(64_000, workerStatus.getInTransitAndWaitingUncachedTokens());
        }

        @Test
        @DisplayName("Missing RUNNING progress falls back to full uncached work")
        void missingRunningProgress_shouldUseUncachedTokens() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            localTask.setInputLength(64_000);
            localTask.setPredictedPrefixLength(16_000);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setInputLength(64_000);
            runningTask.setPrefixLength(16_000);
            runningTask.setPrefixLengthValid(true);
            workerStatus.updateTaskStates(Map.of(), Map.of(REQUEST_ID, runningTask), Map.of());

            TaskInfo updated = workerStatus.getLocalTaskMap().get(REQUEST_ID);
            assertEquals(-1, updated.getRemainingPrefillTokens());
            assertEquals(48_000, workerStatus.getRunningRemainingPrefillTokens());
        }

        @Test
        @DisplayName("Task in waiting then in running on next call should be RUNNING and report waiting-to-running latency once")
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

            var runningUpdateResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());

            assertEquals(TaskStateEnum.RUNNING, workerStatus.getLocalTaskMap().get(REQUEST_ID).getTaskState());
            assertEquals(1, runningUpdateResult.waitingToRunningObservedLatenciesMs().size());
            assertTrue(runningUpdateResult.waitingToRunningObservedLatenciesMs().getFirst() >= 0);

            var repeatedRunningUpdateResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());
            assertTrue(repeatedRunningUpdateResult.waitingToRunningObservedLatenciesMs().isEmpty());
        }

        @Test
        @DisplayName("Engine-observed waiting-to-running latency reported once from engine timestamps")
        void taskWithEngineTimestamps_shouldReportEngineObservedLatencyOnce() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setWaitingEnteredTimeMs(1_000L);
            runningTask.setRunningEnteredTimeMs(1_250L);
            Map<String, TaskInfo> runningTaskInfo = new HashMap<>();
            runningTaskInfo.put(String.valueOf(REQUEST_ID), runningTask);

            var firstResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());
            assertEquals(1, firstResult.engineWaitingToRunningLatenciesMs().size());
            assertEquals(250L, firstResult.engineWaitingToRunningLatenciesMs().getFirst());

            var secondResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());
            assertTrue(secondResult.engineWaitingToRunningLatenciesMs().isEmpty());
        }

        @Test
        @DisplayName("Engine-observed received-to-waiting latency reported once from engine timestamps")
        void taskWithEngineTimestamps_shouldReportEngineReceivedToWaitingLatencyOnce() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo waitingTask = new TaskInfo();
            waitingTask.setRequestId(REQUEST_ID);
            waitingTask.setRequestReceivedTimeMs(1_000L);
            waitingTask.setWaitingEnteredTimeMs(1_080L);
            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), waitingTask);

            var firstResult =
                    workerStatus.updateTaskStates(waitingTaskInfo, new HashMap<>(), new HashMap<>());
            assertEquals(1, firstResult.engineReceivedToWaitingLatenciesMs().size());
            assertEquals(80L, firstResult.engineReceivedToWaitingLatenciesMs().getFirst());

            var secondResult =
                    workerStatus.updateTaskStates(waitingTaskInfo, new HashMap<>(), new HashMap<>());
            assertTrue(secondResult.engineReceivedToWaitingLatenciesMs().isEmpty());
        }

        @Test
        @DisplayName("First running observation reports engine received-to-waiting latency")
        void taskFirstObservedRunning_shouldReportEngineReceivedToWaitingLatency() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setRequestReceivedTimeMs(1_000L);
            runningTask.setWaitingEnteredTimeMs(1_080L);
            runningTask.setRunningEnteredTimeMs(1_250L);
            Map<String, TaskInfo> runningTaskInfo = new HashMap<>();
            runningTaskInfo.put(String.valueOf(REQUEST_ID), runningTask);

            var updateResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());

            assertEquals(1, updateResult.engineReceivedToWaitingLatenciesMs().size());
            assertEquals(80L, updateResult.engineReceivedToWaitingLatenciesMs().getFirst());
        }

        @Test
        @DisplayName("Running snapshot reports received-to-waiting after zero-timestamp pending confirmation")
        void taskPendingThenRunning_shouldReportEngineReceivedToWaitingLatency() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo pendingTask = new TaskInfo();
            pendingTask.setRequestId(REQUEST_ID);
            Map<String, TaskInfo> pendingTaskInfo = new HashMap<>();
            pendingTaskInfo.put(String.valueOf(REQUEST_ID), pendingTask);
            workerStatus.updateTaskStates(pendingTaskInfo, new HashMap<>(), new HashMap<>());

            TaskInfo runningTask = new TaskInfo();
            runningTask.setRequestId(REQUEST_ID);
            runningTask.setRequestReceivedTimeMs(1_000L);
            runningTask.setWaitingEnteredTimeMs(1_080L);
            runningTask.setRunningEnteredTimeMs(1_250L);
            Map<String, TaskInfo> runningTaskInfo = new HashMap<>();
            runningTaskInfo.put(String.valueOf(REQUEST_ID), runningTask);

            var updateResult =
                    workerStatus.updateTaskStates(new HashMap<>(), runningTaskInfo, new HashMap<>());

            assertEquals(1, updateResult.engineReceivedToWaitingLatenciesMs().size());
            assertEquals(80L, updateResult.engineReceivedToWaitingLatenciesMs().getFirst());
        }

        @Test
        @DisplayName("Waiting snapshot reports received-to-waiting after zero-timestamp pending confirmation")
        void taskPendingThenWaiting_shouldReportEngineReceivedToWaitingLatency() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo pendingTask = new TaskInfo();
            pendingTask.setRequestId(REQUEST_ID);
            Map<String, TaskInfo> pendingTaskInfo = new HashMap<>();
            pendingTaskInfo.put(String.valueOf(REQUEST_ID), pendingTask);
            workerStatus.updateTaskStates(pendingTaskInfo, new HashMap<>(), new HashMap<>());

            TaskInfo waitingTask = new TaskInfo();
            waitingTask.setRequestId(REQUEST_ID);
            waitingTask.setRequestReceivedTimeMs(1_000L);
            waitingTask.setWaitingEnteredTimeMs(1_080L);
            Map<String, TaskInfo> waitingTaskInfo = new HashMap<>();
            waitingTaskInfo.put(String.valueOf(REQUEST_ID), waitingTask);

            var updateResult =
                    workerStatus.updateTaskStates(waitingTaskInfo, new HashMap<>(), new HashMap<>());

            assertEquals(1, updateResult.engineReceivedToWaitingLatenciesMs().size());
            assertEquals(80L, updateResult.engineReceivedToWaitingLatenciesMs().getFirst());
        }

        @Test
        @DisplayName("First finished observation reports engine transition latencies")
        void taskFirstObservedFinished_shouldReportEngineTransitionLatencies() {
            TaskInfo localTask = new TaskInfo();
            localTask.setRequestId(REQUEST_ID);
            workerStatus.putLocalTask(REQUEST_ID, localTask);

            TaskInfo finishedTask = new TaskInfo();
            finishedTask.setRequestId(REQUEST_ID);
            finishedTask.setRequestReceivedTimeMs(1_000L);
            finishedTask.setWaitingEnteredTimeMs(1_080L);
            finishedTask.setRunningEnteredTimeMs(1_250L);
            Map<String, TaskInfo> finishedTaskInfo = new HashMap<>();
            finishedTaskInfo.put(String.valueOf(REQUEST_ID), finishedTask);

            var updateResult =
                    workerStatus.updateTaskStates(new HashMap<>(), new HashMap<>(), finishedTaskInfo);

            assertEquals(1, updateResult.engineReceivedToWaitingLatenciesMs().size());
            assertEquals(80L, updateResult.engineReceivedToWaitingLatenciesMs().getFirst());
            assertEquals(1, updateResult.engineWaitingToRunningLatenciesMs().size());
            assertEquals(170L, updateResult.engineWaitingToRunningLatenciesMs().getFirst());
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
