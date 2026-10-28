package org.flexlb.mock.engine.rtp;

import java.util.logging.Logger;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB;

/** Bounded native finished-version log shared by P and D; does not own requests or KV. */
final class TaskHistory {
    private static final Logger LOG = Logger.getLogger(TaskHistory.class.getName());
    static final int LIMIT = 1024;
    record Snapshot(List<TaskInfoPB> tasks, List<TaskInfoPB> finished, long finishedVersion, int waiting, int running) {}
    private record Entry(long version, TaskInfoPB task) {}
    private final ArrayDeque<Entry> finished = new ArrayDeque<>();
    private long version, droppedThrough;

    synchronized void add(TaskInfoPB task) {
        finished.add(new Entry(++version, task));
        if (finished.size() > LIMIT) droppedThrough = finished.removeFirst().version();
    }
    synchronized Snapshot snapshot(long latest, List<TaskInfoPB> tasks, int waiting, int running) {
        if (droppedThrough > 0 && latest < droppedThrough)
            // Native status remains a full current snapshot even when its finished delta has a gap.
            LOG.warning("finished_history_gap requested_version=" + latest + " dropped_through=" + droppedThrough
                + " latest_version=" + version + " retained=" + finished.size());
        return new Snapshot(List.copyOf(tasks), finished.stream().filter(e -> e.version() > latest).map(Entry::task).toList(), Math.max(latest, version), waiting, running);
    }
}
