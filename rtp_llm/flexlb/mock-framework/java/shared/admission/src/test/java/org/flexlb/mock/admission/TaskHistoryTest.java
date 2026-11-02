package org.flexlb.mock.admission;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.logging.*;
import org.flexlb.mock.engine.Scheduler;
import org.junit.jupiter.api.Test;

class TaskHistoryTest {
    private static Scheduler.Task task(long id) {
        return new Scheduler.Task(id, 0, 1, 0, 0, Scheduler.ReportedPhase.RECEIVED, 0, 0, 0, -1, 0, "");
    }
    @Test void laggingReadersKeepCurrentStateAndReceiveExplicitGapEvidence() {
        var history = new TaskHistory();
        for (int i = 1; i <= TaskHistory.LIMIT + 1; i++)
            history.add(task(i));
        var warnings = new ArrayList<String>();
        var logger = Logger.getLogger(TaskHistory.class.getName());
        var handler = new Handler() {
            public void publish(LogRecord record) { warnings.add(record.getMessage()); }
            public void flush() {}
            public void close() {}
        };
        logger.addHandler(handler);
        try {
            var active = List.of(task(2000));
            var snapshot = history.snapshot(0, active, 1, 0);
            assertEquals(active, snapshot.tasks()); assertEquals(1, snapshot.waiting());
            assertEquals(1024, snapshot.finished().size()); assertEquals(2, snapshot.finished().getFirst().requestId());
            assertEquals(1025, snapshot.finishedVersion());
            assertEquals(List.of("finished_history_gap requested_version=0 dropped_through=1 latest_version=1025 retained=1024"), warnings);
            assertTrue(history.snapshot(snapshot.finishedVersion(), active, 1, 0).finished().isEmpty());
            assertEquals(1024, history.snapshot(1, active, 1, 0).finished().size());
            assertEquals(1, warnings.size());
        } finally { logger.removeHandler(handler); }
    }

    @Test void anAheadCursorNeverMovesBackwardsIncludingAnEmptyEngine() {
        var history = new TaskHistory();
        assertEquals(99, history.snapshot(99, List.of(), 0, 0).finishedVersion());
        history.add(task(1));
        var snapshot = history.snapshot(99, List.of(), 0, 0);
        assertEquals(99, snapshot.finishedVersion()); assertTrue(snapshot.finished().isEmpty());
    }
}
