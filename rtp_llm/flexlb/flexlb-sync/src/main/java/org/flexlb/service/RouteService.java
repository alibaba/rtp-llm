package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.Response;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final DefaultRouter router;
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
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        Mono<Response> resultMono;
        if (flexlbConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // Use async queuing mechanism
        } else {
            // Direct routing: keep the worker-map scan off the caller's Netty event loop, the same
            // way batchSchedule does. The queue path above already hands off to a worker thread.
            resultMono = Mono.fromCallable(() -> router.route(balanceContext))
                    .subscribeOn(Schedulers.parallel());
        }

        return resultMono.doOnSuccess(result -> {
            balanceContext.setResponse(result);
        });
    }

    /**
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        if (flexlbConfig.isEnableQueueing()) {
            balanceContext.cancel();
            CompletableFuture<Response> future = balanceContext.getFuture();
            if (future != null) {
                future.completeExceptionally(new CancellationException("Request cancelled by client"));
            }
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }

    /**
     * Batch dispatch for single-role deployments. Bypasses the request queue and the
     * {@code localTaskMap} bookkeeping; reconciliation and lost-task detection are not
     * available on this path. Multi-role deployments must use {@link #route} per request.
     */
    public Mono<BatchScheduleResponse> batchSchedule(BatchScheduleRequest batchScheduleRequest) {
        // router.batchSchedule scans the worker map and allocates the target list; run it on a
        // worker thread so it never executes on the caller's Netty event loop (the dispatcher's
        // in-JVM call and the master's HTTP handler both subscribe from event-loop threads).
        return Mono.fromCallable(() -> router.batchSchedule(batchScheduleRequest))
                .subscribeOn(Schedulers.parallel());
    }
}
