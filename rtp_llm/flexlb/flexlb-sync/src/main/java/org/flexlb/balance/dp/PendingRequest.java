package org.flexlb.balance.dp;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.concurrent.CompletableFuture;

/**
 * Single request waiting in a {@link PrefillQueue} for batch flush.
 * <p>
 * Once the batch is built and Master.Enqueue returns Ack, the scheduler completes
 * {@link #future()} with the assembled {@link Response} so the upstream {@code Mono}
 * (created in {@link org.flexlb.service.RouteService}) resumes and the HTTP caller
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

    /** Wall-clock wait so PrefillQueue can force-flush a single starving request. */
    public long waitMicros() {
        return System.nanoTime() / 1000 - enqueuedAtMicros;
    }

    public long requestId() {
        return ctx == null ? -1 : ctx.getRequestId();
    }
}
