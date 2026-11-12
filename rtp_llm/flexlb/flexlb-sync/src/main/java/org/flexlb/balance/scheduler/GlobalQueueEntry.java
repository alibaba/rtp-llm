package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/** One request identity retained by the model-wide ordered queue. */
final class GlobalQueueEntry {

    final BalanceContext context;
    final CompletableFuture<Response> future;
    final int priority;
    final String routingGroup;
    /** Arrival marker for shedding controllers; captured once at offer time. */
    final long queuedAtNanos = System.nanoTime();
    long sequence;
    volatile boolean removed;
    /** Latest fleet-side work estimate used by eviction decisions. */
    volatile PrefillQueueShedding.Estimate sheddingEstimate;
    /** Whether the arrival has not yet been classified with the controller. */
    volatile boolean classifyArrival = true;
    /** Set once planning proves the request needs no cached prefix. */
    volatile boolean knownColdRequest;
    /** Latest blocked placement key observed while waiting in queue. */
    volatile PlacementKey lastBlocker;
    GlobalQueueEntry previous;
    GlobalQueueEntry next;
    boolean linked;

    GlobalQueueEntry(
            BalanceContext context,
            CompletableFuture<Response> future,
            int priority) {
        this(context, future, priority, null);
    }

    GlobalQueueEntry(BalanceContext context, CompletableFuture<Response> future,
            int priority, String routingGroup) {
        this.context = context;
        this.future = future;
        this.priority = priority;
        this.routingGroup = routingGroup;
    }
}
