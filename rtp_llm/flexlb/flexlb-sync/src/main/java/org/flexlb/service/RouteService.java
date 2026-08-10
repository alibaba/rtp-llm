package org.flexlb.service;

import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
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
    private final Router router;
    private final QueueManager queueManager;

    public RouteService(ConfigService configService,
                        Router defaultScheduler,
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
            // Direct routing runs inline on the subscribing thread so a client cancel cannot
            // interleave with the worker reservation taken inside router.route().
            resultMono = Mono.fromCallable(() -> router.route(balanceContext));
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
