package org.flexlb.mock.engine.scheduler;

import java.util.*;
import java.util.logging.Logger;
import org.flexlb.mock.engine.scheduler.EngineScheduler.Task;

/**
 * Bounded native finished-version log shared by P and D; does not own requests or KV.
 */
public final class TaskHistory {
    private static final Logger LOG = Logger.getLogger(TaskHistory.class.getName());

    public static final int LIMIT = 1024;

    private final ArrayDeque<FinishedTask> finished = new ArrayDeque<>();

    private long version, droppedThrough, nextWarning;

    private String pendingWarning;

    public synchronized void add(Task task) {
        finished.add(new FinishedTask(++version, task));
        if (finished.size() > LIMIT) droppedThrough = finished.removeFirst().version();
    }

    public synchronized EngineScheduler.Status snapshot(long latest, List<Task> tasks, int waiting, int running) {
        long now = System.nanoTime();
        if (droppedThrough > 0 && latest < droppedThrough && now >= nextWarning) {
            pendingWarning = "finished_history_gap requested_version=" + latest + " dropped_through=" + droppedThrough
                + " latest_version=" + version + " retained=" + finished.size();
            nextWarning = now + java.util.concurrent.TimeUnit.SECONDS.toNanos(1);
        }
        return new EngineScheduler.Status(List.copyOf(tasks), finished.stream().filter(e -> e.version() > latest).map(FinishedTask::task).toList(), Math.max(latest, version), waiting, running);
    }

    /** RPC calls this after releasing execution/history locks; slow handlers cannot block scheduling. */
    public void reportGaps() {
        String warning;
        synchronized (this) { warning = pendingWarning; pendingWarning = null; }
        if (warning != null) LOG.warning(warning);
    }

    private record FinishedTask(long version, Task task) {}

}
