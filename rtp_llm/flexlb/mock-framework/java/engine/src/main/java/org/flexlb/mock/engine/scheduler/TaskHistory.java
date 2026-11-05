package org.flexlb.mock.engine.scheduler;

import java.util.*;
import java.util.logging.Logger;
import org.flexlb.mock.engine.scheduler.EngineScheduler.Task;

/** Bounded native finished-version log shared by P and D; does not own requests or KV. */
public final class TaskHistory {
    private static final Logger LOG = Logger.getLogger(TaskHistory.class.getName());
    public static final int LIMIT = 1024;
    private record Entry(long version, Task task) {}
    private final ArrayDeque<Entry> finished = new ArrayDeque<>();
    private long version, droppedThrough;

    public synchronized void add(Task task) {
        finished.add(new Entry(++version, task));
        if (finished.size() > LIMIT) droppedThrough = finished.removeFirst().version();
    }
    public synchronized EngineScheduler.Status snapshot(long latest, List<Task> tasks, int waiting, int running) {
        if (droppedThrough > 0 && latest < droppedThrough)
            // Native status remains a full current snapshot even when its finished delta has a gap.
            LOG.warning("finished_history_gap requested_version=" + latest + " dropped_through=" + droppedThrough
                + " latest_version=" + version + " retained=" + finished.size());
        return new EngineScheduler.Status(List.copyOf(tasks), finished.stream().filter(e -> e.version() > latest).map(Entry::task).toList(), Math.max(latest, version), waiting, running);
    }
}
