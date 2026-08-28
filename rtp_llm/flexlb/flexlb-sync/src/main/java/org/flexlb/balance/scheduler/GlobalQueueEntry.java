package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/** One request identity retained by the model-wide ordered queue. */
final class GlobalQueueEntry {

    final BalanceContext context;
    final CompletableFuture<Response> future;
    final int priority;
    volatile boolean removed;
    volatile PlacementKey blockedKey;
    volatile WorkerEndpoint blockedEndpoint;
    GlobalQueueEntry previous;
    GlobalQueueEntry next;
    boolean linked;

    GlobalQueueEntry(
            BalanceContext context,
            CompletableFuture<Response> future,
            int priority) {
        this.context = context;
        this.future = future;
        this.priority = priority;
    }
}
