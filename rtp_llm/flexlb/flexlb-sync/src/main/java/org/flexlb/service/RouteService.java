package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final RoutingQueueReporter routingQueueReporter;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        RoutingQueueReporter routingQueueReporter) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.routingQueueReporter = routingQueueReporter;
    }

    /**
     * Routes a request to appropriate workers and owns cancellation of the returned subscription.
     *
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public Mono<Response> route(BalanceContext balanceContext) {
        return manageCancellation(routeInternal(balanceContext), balanceContext);
    }

    /**
     * Routes a request and maps its result while owning cancellation of the complete caller chain.
     *
     * @param balanceContext Load balancing context
     * @param responseHandler Maps the routing result to the caller response
     * @param <T> Caller response type
     * @return Mapped routing result
     */
    public <T> Mono<T> route(BalanceContext balanceContext,
                             Function<Response, Mono<T>> responseHandler) {
        return manageCancellation(routeInternal(balanceContext).flatMap(responseHandler), balanceContext);
    }

    private Mono<Response> routeInternal(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        Mono<Response> resultMono;
        if (flexlbConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // Use async queuing mechanism
        } else {
            resultMono = router.route(balanceContext);  // Direct routing without queuing
        }

        return resultMono.doOnSuccess(balanceContext::setResponse);
    }

    private <T> Mono<T> manageCancellation(Mono<T> routeMono, BalanceContext balanceContext) {
        return routeMono.doOnCancel(() -> cancel(balanceContext));
    }

    /**
     * Cancels a request and releases any routed workers once.
     *
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        if (!balanceContext.tryCancel()) {
            return;
        }
        routingQueueReporter.reportRoutingCancelled();
        Logger.info(
                "Routing cancelled, requestId={}, queueing={}",
                balanceContext.getRequestId(),
                flexlbConfig.isEnableQueueing());
        router.rollBack(balanceContext, balanceContext.getResponse());
        if (flexlbConfig.isEnableQueueing()) {
            CompletableFuture<Response> future = balanceContext.getFuture();
            if (future != null) {
                future.completeExceptionally(new CancellationException("Request cancelled by client"));
            }
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }
}
