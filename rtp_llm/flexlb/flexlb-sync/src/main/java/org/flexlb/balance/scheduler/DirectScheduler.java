package org.flexlb.balance.scheduler;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.util.Logger;

import java.util.concurrent.CompletableFuture;

/**
 * Scheduler for DIRECT mode: route → complete, no queueing, no batching.
 *
 * <p>Composes a {@link Router} for worker selection and completes the result
 * future synchronously with the routing response. Requests are admitted
 * through the base-class pending registry (duplicate detection while a
 * previous submission is pending; lifecycle ends at terminal completion).
 *
 * <p>Duplicate request IDs (previous submission still pending) are logged
 * and processed untracked — the direct path never rejects a request for
 * tracking reasons, preserving the pre-registration behavior.
 *
 * <p>Also serves as the base class for {@link QueueScheduler}, which reuses
 * {@link #routeAndComplete} for the dequeue-then-route worker path and
 * overrides {@link #onRouteResult} to add retry semantics.
 */
public class DirectScheduler extends AbstractScheduler {

    protected final Router router;

    public DirectScheduler(Router router, FlexlbMetricHelper metricHelper) {
        super(metricHelper);
        this.router = router;
    }

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        if (!register(ctx, future)) {
            Logger.warn("Duplicate request_id: {}, processing untracked", ctx.getRequestId());
        }
        routeAndComplete(ctx, future);
        return future;
    }

    /**
     * Route the request through the composed {@link Router} and settle the
     * given future with the outcome. Routing exceptions complete the future
     * exceptionally.
     *
     * <p>Shared by the DIRECT submit path and the QUEUE worker-consume path
     * ({@link QueueScheduler}); the completion policy is the
     * {@link #onRouteResult} template hook.
     */
    protected void routeAndComplete(BalanceContext ctx, CompletableFuture<Response> future) {
        try {
            RouteResult result = router.route(ctx);
            Response response = result.toResponse();
            onRouteResult(ctx, future, response);
        } catch (Exception e) {
            Logger.error("Failed to route request id: {}", ctx.getRequestId(), e);
            future.completeExceptionally(e);
        }
    }

    /**
     * Completion policy for a routing result: complete the future directly.
     * {@link QueueScheduler} overrides this to intercept retryable failures
     * and re-queue instead of completing.
     */
    protected void onRouteResult(BalanceContext ctx, CompletableFuture<Response> future,
                                 Response response) {
        future.complete(response);
    }
}
