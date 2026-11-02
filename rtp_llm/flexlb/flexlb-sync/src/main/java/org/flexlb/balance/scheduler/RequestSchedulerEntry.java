package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/** Request admission, result futures and cancellation around the scheduling decisions. */
public class RequestSchedulerEntry {
    private final RequestScheduler scheduler;
    private final RequestLifecycle lifecycle;
    private final RequestWorkerSelector router;
    private final boolean queue;

    public RequestSchedulerEntry(
            RequestScheduler scheduler,
            RequestLifecycle lifecycle,
            RequestWorkerSelector router,
            boolean queue) {
        this.scheduler = Objects.requireNonNull(scheduler);
        this.lifecycle = Objects.requireNonNull(lifecycle);
        this.router = Objects.requireNonNull(router);
        this.queue = queue;
    }

    public CompletableFuture<Response> submit(RequestContext context) {
        return submit(context, () -> {});
    }

    public CompletableFuture<Response> submit(RequestContext context, Runnable onRegistered) {
        Objects.requireNonNull(onRegistered, "onRegistered");
        try {
            CompletableFuture<Response> special =
                    tryCompleteVitRequest(context, lifecycle.requests, router);
            if (special != null) {
                return special;
            }
            if (context != null && context.data().getConfig().isQueue() != queue) {
                return CompletableFuture.completedFuture(
                        Response.buildErrorResponse(
                                StrategyErrorType.DISPATCH_FAILED,
                                "request configuration does not match scheduler mode"));
            }
            CompletableFuture<Response> future =
                    lifecycle.register(
                            context,
                            onRegistered,
                            queue
                                    ? StrategyErrorType.RESOURCE_EXHAUSTED
                                    : StrategyErrorType.BATCH_SLO_EXPIRED);
            if (!future.isDone()) {
                scheduler.onRequest(context);
                if (!queue) scheduler.schedule(System.nanoTime());
            } else if (context != null
                    && context.data().getScheduler() == lifecycle
                    && context.data().getFuture() == future) {
                lifecycle.closeRequest(context, null);
            }
            return future;
        } catch (RuntimeException failure) {
            Response response =
                    Response.buildErrorResponse(
                            StrategyErrorType.DISPATCH_FAILED, failure.getMessage());
            if (context != null && context.data().getScheduler() == lifecycle) {
                lifecycle.publish(context, response, false);
                return context.data().getFuture();
            }
            return CompletableFuture.completedFuture(response);
        }
    }

    public RequestState cancel(long requestId, long batchId, CancelReason reason) {
        return lifecycle.cancel(requestId, batchId, reason);
    }

    /** Completes ViT-only or stale-preselection requests; null leaves mode-specific registration to the caller. */
    static CompletableFuture<Response> tryCompleteVitRequest(
            RequestContext context, RequestRepository requests, RequestWorkerSelector router) {
        if (context == null || context.data().getRequest() == null) {
            return null;
        }
        var request = context.data().getRequest();
        if (request.isVitRouteOnly()) {
            var result = CompletableFuture.completedFuture(router.routeVit(context));
            context.data().setFuture(result);
            return result;
        }
        if (request.getSelectedVit() != null
                && requests.getRequestState(context.data().getRequestId(), 0L) == null
                && !router.selectedVitIsValid(context)) {
            return CompletableFuture.completedFuture(
                    Response.error(StrategyErrorType.VIT_ROUTE_STALE));
        }
        return null;
    }
}
