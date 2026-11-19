package org.flexlb.dao.master;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PrefillTtftWindowTest {
    private final WorkerStatus worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "g", "127.0.0.1", 1, 2, "s");
    private final PrefillTtftWindow window = new PrefillTtftWindow();

    @Test
    void deduplicatesActiveThenFinishedAndRepeatedPolls() {
        poll(1000, 0, Map.of(), Map.of());
        TaskInfo task = task("a", 1100, 1300, 0);
        poll(1400, 0, Map.of("a", task), Map.of());
        poll(1500, 1, Map.of(), Map.of("a", task));
        poll(1600, 1, Map.of(), Map.of("a", task));
        var result = window.read(1100, 1600, 1000, 1000);
        assertEquals(1, result.count());
        assertEquals(200, result.totalMs());
    }

    @Test
    void includesSlowRequestCrossingWindowBoundaryButExcludesPriorMembership() {
        poll(1000, 0, Map.of(), Map.of());
        poll(2400, 2, Map.of(), Map.of("a", task("a", 1100, 2300, 0), "b", task("b", 900, 2200, 0)));
        var result = window.read(2000, 2400, 1000, 1000);
        assertEquals(1, result.count());
        assertEquals(1200, result.totalMs());
    }

    @Test
    void idleWorkerHasNoSamplesAndLongPollGapWithoutLostRecordsIsValid() {
        poll(1000, 0, Map.of(), Map.of());
        poll(5000, 0, Map.of(), Map.of());
        assertEquals(0, window.read(1100, 5000, 1000, 1000).count());
    }

    @Test
    void missingFirstTokenNeverBecomesZeroEvenWhenErrorCodeDefaultsToZero() {
        poll(1000, 0, Map.of(), Map.of());
        poll(1500, 1, Map.of(), Map.of("e", task("e", 1100, 0, 4)));
        assertEquals(0, window.read(1100, 1500, 1000, 1000).count());
        poll(1600, 2, Map.of(), Map.of("bad", task("bad", 1100, 0, 0)));
        assertEquals(0, window.read(1100, 1600, 1000, 1000).count());
    }

    @Test
    void finishedCursorGapInvalidatesWindowAndRecoversAfterFreshWindow() {
        poll(1000, 0, Map.of(), Map.of());
        poll(1500, 3, Map.of(), Map.of("a", task("a", 1100, 1400, 0)));
        assertThrows(IllegalStateException.class, () -> window.read(1100, 1500, 1000, 1000));
        poll(2000, 4, Map.of(), Map.of("b", task("b", 1700, 1900, 0)));
        assertEquals(1, window.read(1600, 2000, 1000, 1600).count());
    }

    @Test
    void cursorRegressionRequiresGenerationReplacement() {
        poll(1000, 10, Map.of(), Map.of());
        poll(1500, 0, Map.of(), Map.of());
        poll(3000, 1, Map.of(), Map.of("a", task("a", 2100, 2200, 0)));
        assertThrows(IllegalStateException.class, () -> window.read(2000, 3000, 1000, 2000));
    }

    @Test
    void staleOrUncoveredWindowIsRejected() {
        poll(1000, 0, Map.of(), Map.of());
        assertThrows(IllegalStateException.class, () -> window.read(900, 1000, 1000, 0));
        assertThrows(IllegalStateException.class, () -> window.read(1000, 3000, 1000, 0));
    }

    @Test
    void invalidDurationIsRejectedRatherThanSubtractedFromMean() {
        poll(1000, 0, Map.of(), Map.of());
        poll(1500, 1, Map.of(), Map.of("a", task("a", 1400, 1300, 0)));
        assertThrows(IllegalStateException.class, () -> window.read(1100, 1500, 1000, 1000));
    }

    @Test
    void capacityOverflowHoldsInsteadOfUsingTruncatedSamplePopulation() {
        poll(1000, 0, Map.of(), Map.of());
        Map<String, TaskInfo> active = new HashMap<>();
        for (int i = 0; i < 100_001; i++) {
            String id = "request-" + i;
            active.put(id, task(id, 3000, 4000, 0));
        }
        poll(5000, 0, active, Map.of());
        assertThrows(IllegalStateException.class, () -> window.read(2000, 5000, 1000, 1000));
    }

    private void poll(long now, long finishedVersion, Map<String, TaskInfo> active, Map<String, TaskInfo> finished) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setStatusVersion(now);
        response.setLatestFinishedVersion(finishedVersion);
        response.setRunningTaskInfo(active);
        response.setFinishedTaskInfo(finished);
        window.collect(worker.freezeStatusResponse(response), now);
    }

    private static TaskInfo task(String id, long received, long first, long error) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(id);
        task.setRequestReceivedTimeMs(received);
        task.setFirstTokenTimeMs(first);
        task.setErrorCode(error);
        return task;
    }
}
