package org.flexlb.balance.request;

import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.CancelReason;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestPhase;
import org.flexlb.balance.schedule.RequestRepository;
import org.flexlb.balance.schedule.RequestState;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/**
 * Routes local requests through ViT selection or request scheduling and records successful cache feedback.
 * Protocol conversion and transport delivery remain with the API.
 */
@Component
public class RequestRoutingService {
    private final AbstractRequestScheduler scheduler;
    private final RequestRepository requests;
    private final RequestRouteSelector selector;
    private final CacheAwareService cacheService;

    public RequestRoutingService(
            AbstractRequestScheduler scheduler,
            RequestRepository requests,
            RequestRouteSelector selector,
            CacheAwareService cacheService) {
        this.scheduler = scheduler;
        this.requests = requests;
        this.selector = selector;
        this.cacheService = cacheService;
    }

    public CompletableFuture<Response> submit(RequestContext ctx, Runnable onRegistered) {
        var request = ctx.getRequest();
        CompletableFuture<Response> result;
        try {
            if (request.isVitRouteOnly()) {
                result = CompletableFuture.completedFuture(selector.routeVit(ctx));
                ctx.setFuture(result);
            } else if (request.getSelectedVit() != null
                    && requests.getRequestState(ctx.getRequestId(), ctx.getPhase()) == null
                    && !selector.selectedVitIsValid(ctx)) {
                result =
                        CompletableFuture.completedFuture(
                                Response.error(StrategyErrorType.VIT_ROUTE_STALE));
            } else {
                result = scheduler.submit(ctx, onRegistered);
            }
        } catch (RuntimeException failure) {
            result =
                    CompletableFuture.completedFuture(
                            Response.buildErrorResponse(
                                    StrategyErrorType.DISPATCH_FAILED, failure.getMessage()));
        }
        return result.thenApply(
                response -> {
                    if (response.isSuccess()) {
                        try {
                            cacheService.updateFromRoutedRequest(
                                    request.getRequestId(),
                                    request.getBlockSize(),
                                    request.getBlockCacheKeys(),
                                    response.getServerStatus());
                        } catch (RuntimeException failure) {
                            Logger.warn("Routed cache metadata update failed", failure);
                        }
                    }
                    return response;
                });
    }

    public RequestState cancel(String id, CancelReason reason, RequestPhase phase) {
        return scheduler.cancel(id, reason, phase);
    }
}
