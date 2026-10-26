package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.concurrent.CompletableFuture;

/** Shared ViT fast paths before either placement mode registers a request. */
final class RequestSchedulerEntry {
    private RequestSchedulerEntry() { }

    /** Completes ViT-only or stale-preselection requests; null leaves mode-specific registration to the caller. */
    static CompletableFuture<Response> tryCompleteVitRequest(
            RequestContext context, RequestRepository requests, RequestWorkerSelector router) {
        if (context == null || context.data().getRequest() == null) { return null; }
        var request = context.data().getRequest();
        if (request.isVitRouteOnly()) {
            var result = CompletableFuture.completedFuture(router.routeVit(context));
            context.data().setFuture(result);
            return result;
        }
        if (request.getSelectedVit() != null
                && requests.getRequestState(context.data().getRequestId(), 0L) == null
                && !router.selectedVitIsValid(context)) {
            return CompletableFuture.completedFuture(Response.error(StrategyErrorType.VIT_ROUTE_STALE));
        }
        return null;
    }

}
