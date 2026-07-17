package org.flexlb.service;

import java.util.concurrent.CompletableFuture;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final FlexlbBatchScheduler flexlbBatchScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;
    private final EndpointRegistry endpointRegistry;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        @Lazy @Autowired(required = false) FlexlbBatchScheduler flexlbBatchScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter,
                        EndpointRegistry endpointRegistry) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.flexlbBatchScheduler = flexlbBatchScheduler;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
        this.endpointRegistry = endpointRegistry;
    }

    /**
     * Route request to appropriate workers
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public CompletableFuture<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        // Resolve AUTO to actual schedule mode so downstream components
        // (e.g., CostBasedDecodeStrategy) can make mode-aware decisions.
        ScheduleModeEnum mode = balanceContext.getScheduleMode();
        if (mode == ScheduleModeEnum.AUTO) {
            if (shouldUseFlexlbBatch(balanceContext, flexlbConfig)) {
                mode = ScheduleModeEnum.BATCH;
            } else if (flexlbConfig.isEnableQueueing()) {
                mode = ScheduleModeEnum.QUEUE;
            } else {
                mode = ScheduleModeEnum.DIRECT;
            }
            balanceContext.setScheduleMode(mode);
        }

        CompletableFuture<Response> resultFuture;
        if (shouldUseFlexlbBatch(balanceContext, flexlbConfig)) {
            resultFuture = flexlbBatchScheduler.submit(balanceContext);
            balanceContext.setFuture(resultFuture);
        } else if (mode == ScheduleModeEnum.QUEUE || flexlbConfig.isEnableQueueing()) {
            resultFuture = queueManager.tryRouteAsync(balanceContext).toFuture();  // Use async queuing mechanism
        } else {
            // Direct routing without queuing
            try {
                resultFuture = CompletableFuture.completedFuture(router.route(balanceContext));
            } catch (Exception e) {
                resultFuture = CompletableFuture.failedFuture(e);
            }
        }

        return resultFuture.whenComplete((result, throwable) -> {
            if (throwable != null) {
                return;
            }
            balanceContext.setResponse(result);
            if (result != null && result.isSuccess()) {
                recentCacheKeyTraceReporter.report(balanceContext);
            }
        });
    }

    boolean shouldUseFlexlbBatch(BalanceContext ctx, FlexlbConfig config) {
        if (flexlbBatchScheduler == null || config == null) {
            return false;
        }
        ScheduleModeEnum mode = ctx.getScheduleMode();
        if (mode == ScheduleModeEnum.BATCH) {
            return true;
        }
        if (mode == ScheduleModeEnum.DIRECT) {
            return false;
        }
        if (mode == ScheduleModeEnum.QUEUE) {
            return false;
        }
        // AUTO: use batch when config enables it and request characteristics match
        if (!config.isFlexlbBatchEnabled()) {
            return false;
        }
        Request request = ctx.getRequest();
        return request != null
                && request.getMaxNewTokens() > 1
                && request.getNumBeams() <= 1
                && !request.isForceDisableSpRun()
                && ctx.getGenerateInputPbBytes() != null
                && ctx.getGenerateInputPbBytes().length > 0;
    }
}
