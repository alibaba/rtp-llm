package org.flexlb.balance.dp;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * One request waiting in a {@link DispatchBatcher}'s queue.
 * <p>
 * The future is the same one returned to {@link org.flexlb.service.RouteService};
 * completion happens either from the planner (per-request failure), the dispatcher
 * (Master.Enqueue ack), or RouteService (cancel).
 */
public record QueuedRequest(
        BalanceContext ctx,
        CompletableFuture<Response> future,
        long enqueuedAtMicros,
        int computeTokenLength,
        long sloDeadlineMicros,
        int bucketIndex) {

    public static QueuedRequest of(BalanceContext ctx, CompletableFuture<Response> future) {
        return new QueuedRequest(ctx, future, System.nanoTime() / 1000, 0, Long.MAX_VALUE, 0);
    }

    public static QueuedRequest of(BalanceContext ctx, CompletableFuture<Response> future,
                                   int computeTokenLength, long sloDeadlineMicros, int bucketIndex) {
        return new QueuedRequest(ctx, future, System.nanoTime() / 1000,
                computeTokenLength, sloDeadlineMicros, bucketIndex);
    }

    public long requestId() {
        return ctx == null ? -1 : ctx.getRequestId();
    }

    public long waitMicros() {
        return System.nanoTime() / 1000 - enqueuedAtMicros;
    }
}
