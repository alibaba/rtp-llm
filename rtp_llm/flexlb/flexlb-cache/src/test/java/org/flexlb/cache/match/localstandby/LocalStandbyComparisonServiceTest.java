package org.flexlb.cache.match.localstandby;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies feedback correlation and distinguishes unreported samples from zero hits.
 */
class LocalStandbyComparisonServiceTest {
    private static final String ADDRESS = "127.0.0.1:8080";
    private static final LocalStandbyComparisonService.Dispatch DISPATCH =
            new LocalStandbyComparisonService.Dispatch(41, RoleType.PREFILL, ADDRESS, 7, 11);

    @Test
    void onlyOneValidObservationForTheExactDispatchIsReported() {
        var events = new ArrayList<LocalStandbyComparisonService.Comparison>();
        var comparison = new LocalStandbyComparisonService(events::add);
        var prediction = comparison.register(DISPATCH, "group", 48, 16, 2, 48L);
        assertNull(comparison.register(DISPATCH, "group", 48, 16, 3, 48L));
        WorkerStatus worker = worker(7);
        comparison.observe(worker, observation(worker, 11, false));
        comparison.observe(worker, observation(worker, 12, true));
        WorkerStatus replacement = worker(8);
        comparison.observe(replacement, observation(replacement, 11, true));

        assertTrue(events.isEmpty());
        comparison.complete(DISPATCH, prediction, 1);
        assertEquals(1, events.size());
        assertNull(events.getFirst().actualHitTokens());

        comparison.observe(worker, observation(worker, 11, true));
        comparison.observe(worker, observation(worker, 11, true));
        assertEquals(2, events.size());
        var feedback = events.getLast();
        assertEquals("cache_hit_comparison", feedback.event());
        assertEquals(0L, feedback.actualHitTokens());
        assertEquals(32L, feedback.kvcmPrediction().localPredictionTokens());
        assertEquals(48L, feedback.kvcmPrediction().globalPredictionTokens());
        assertEquals(16L, feedback.localStandbyPrediction().predictedHitTokens());
    }

    @Test
    void anEarlyActualSampleCanCompleteWithoutAShadowPrediction() {
        var events = new ArrayList<LocalStandbyComparisonService.Comparison>();
        var comparison = new LocalStandbyComparisonService(events::add);
        var prediction = comparison.register(DISPATCH, null, 48, 16, 0, null);
        WorkerStatus worker = worker(7);
        comparison.observe(worker, observation(worker, 11, true));
        assertTrue(events.isEmpty());
        comparison.complete(DISPATCH, prediction, null);
        assertEquals(List.of("cache_hit_comparison"), events.stream()
                .map(LocalStandbyComparisonService.Comparison::event).toList());
        assertNull(events.getFirst().localStandbyPrediction());
        assertEquals(-1L, events.getFirst().kvcmPrediction().globalPredictionTokens());
    }

    private static WorkerStatus worker(long generation) {
        WorkerStatus worker = mock(WorkerStatus.class);
        when(worker.getIpPort()).thenReturn(ADDRESS);
        when(worker.getGenerationId()).thenReturn(generation);
        when(worker.isActiveGeneration()).thenReturn(true);
        return worker;
    }

    private static WorkerStatus.StatusObservation observation(WorkerStatus worker, long batchId, boolean valid) {
        var task = mock(WorkerStatus.TaskObservation.class);
        when(task.requestId()).thenReturn(41L);
        when(task.batchId()).thenReturn(batchId);
        when(task.inputLength()).thenReturn(48L);
        when(task.prefixLengthValid()).thenReturn(valid);
        var observation = mock(WorkerStatus.StatusObservation.class);
        when(observation.owner()).thenReturn(worker);
        when(observation.role()).thenReturn(RoleType.PREFILL);
        when(observation.finishedTasks()).thenReturn(Map.of("41", task));
        return observation;
    }
}
