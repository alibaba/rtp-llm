package org.flexlb.mock.engine.rtp;

import io.grpc.Status;
import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.TaskInfoPB;

/** Bounded native finished-version log shared by P and D; does not own requests or KV. */
final class TaskHistory {
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
            throw Status.OUT_OF_RANGE.withDescription("Finished task history no longer covers requested version").asRuntimeException();
        return new Snapshot(List.copyOf(tasks), finished.stream().filter(e -> e.version() > latest).map(Entry::task).toList(), version, waiting, running);
    }
}
