package org.flexlb.mock.engine;

import java.util.Collection;

/** Counts requests, regardless of how a scheduler stores groups or batches. */
public record QueueSnapshot(int waiting, int running) {
    public static QueueSnapshot of(Collection<RequestState.Snapshot> requests) {
        int waiting = 0, running = 0;
        for (var request : requests) {
            if (request.terminating()) continue;
            if (request.phase() == RequestState.Phase.FORWARD) running++;
            else waiting++;
        }
        return new QueueSnapshot(waiting, running);
    }
}
