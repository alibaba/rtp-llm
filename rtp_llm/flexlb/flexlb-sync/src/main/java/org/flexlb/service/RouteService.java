package org.flexlb.service;

import java.util.concurrent.CancellationException;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
    }

    /**
     * Route request to appropriate workers
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public Mono<Response> route(BalanceContext balanceContext) {
        balanceContext.getSpan().addEvent("start selectEngineWorker");
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(whaleMasterConfig);

        Mono<Response> resultMono;
        if (whaleMasterConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // Use async queuing mechanism
        } else {
            resultMono = Mono.fromCallable(() -> router.route(balanceContext));  // Direct routing without queuing
        }

        return resultMono.doOnSuccess(result -> {
            balanceContext.setResponse(result);
            balanceContext.getSpan().addEvent("finish selectEngineWorker");
        });
    }

    /**
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        WhaleMasterConfig whaleMasterConfig = configService.loadBalanceConfig();
        if (whaleMasterConfig.isEnableQueueing()) {
            balanceContext.cancel();
            balanceContext.getFuture().completeExceptionally(new CancellationException("Request cancelled by client"));
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }
}
