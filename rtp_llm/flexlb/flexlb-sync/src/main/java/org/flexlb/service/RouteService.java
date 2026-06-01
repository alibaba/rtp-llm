package org.flexlb.service;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final FlexlbBatchScheduler flexlbBatchScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        @Autowired(required = false) FlexlbBatchScheduler flexlbBatchScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.flexlbBatchScheduler = flexlbBatchScheduler;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
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
        if (shouldUseFlexlbBatch(balanceContext, flexlbConfig)) {
            CompletableFuture<Response> future = flexlbBatchScheduler.submit(balanceContext);
            balanceContext.setFuture(future);
            resultMono = Mono.fromFuture(future);
        } else if (flexlbConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // Use async queuing mechanism
        } else {
            resultMono = Mono.fromCallable(() -> router.route(balanceContext));  // Direct routing without queuing
        }

        return resultMono.doOnSuccess(result -> {
            balanceContext.setResponse(result);
            if (result != null && result.isSuccess()) {
                recentCacheKeyTraceReporter.report(balanceContext);
            }
        });
    }

    /**
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.cancel();
        if (flexlbConfig.isEnableQueueing()) {
            CompletableFuture<Response> future = balanceContext.getFuture();
            if (future != null) {
                future.completeExceptionally(new CancellationException("Request cancelled by client"));
            }
        }
        if (flexlbBatchScheduler != null && balanceContext.getRequest() != null) {
            flexlbBatchScheduler.cancel(balanceContext.getRequest().getRequestId());
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }

    public void cancelByRequestId(long requestId) {
        if (flexlbBatchScheduler != null) {
            flexlbBatchScheduler.cancel(requestId);
        }
    }

    boolean shouldUseFlexlbBatch(BalanceContext ctx, FlexlbConfig config) {
        if (flexlbBatchScheduler == null || config == null || !config.isFlexlbBatchEnabled()) {
            return false;
        }
        Request request = ctx.getRequest();
        return request != null
                && request.getMaxNewTokens() > 1
                && request.getNumBeams() <= 1
                && !request.isForceDisableSpRun()
                && request.getGenerateInputPbB64() != null
                && !request.getGenerateInputPbB64().isEmpty();
    }
}
