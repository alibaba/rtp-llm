package org.flexlb.balance.dp;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.concurrent.CompletableFuture;

/**
 * One placed request inside a {@link DispatchBatch} (prefill + decode workers
 * already chosen by {@link DispatchPlanner}). Once Master.Enqueue is acked, the
 * scheduler completes {@link #future()} so the upstream {@code Mono} created
 * in {@link org.flexlb.service.RouteService} resumes and the HTTP caller
 * receives a routing decision.
 */
public record PendingRequest(
        BalanceContext ctx,
        ServerStatus prefill,
        ServerStatus decode,
        CompletableFuture<Response> future,
        long enqueuedAtMicros) {

    public static PendingRequest of(BalanceContext ctx,
                                    ServerStatus prefill,
                                    ServerStatus decode,
                                    CompletableFuture<Response> future) {
        return new PendingRequest(ctx, prefill, decode, future, System.nanoTime() / 1000);
    }

    /** Wall-clock wait so the batcher can force-flush a single starving request. */
    public long waitMicros() {
        return System.nanoTime() / 1000 - enqueuedAtMicros;
    }

    public long requestId() {
        return ctx == null ? -1 : ctx.getRequestId();
    }
}
