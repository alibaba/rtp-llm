package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final PriorityScheduler priorityScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        PriorityScheduler priorityScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.priorityScheduler = priorityScheduler;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
    }

    /**
     * Route request to appropriate workers based on the deployment-level schedule mode.
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public CompletableFuture<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        ScheduleModeEnum mode = flexlbConfig.getDefaultScheduleModeEnum();
        balanceContext.setScheduleMode(mode);

        CompletableFuture<Response> resultFuture = switch (mode) {
            case BATCH -> routeBatch(balanceContext);
            case QUEUE -> flexlbConfig.usesRouteDecisionDelivery()
                    ? submitScheduled(balanceContext)
                    : queueManager.tryRouteAsync(balanceContext).toFuture();
            case DIRECT -> routeDirect(balanceContext);
        };

        // Observe the scheduler-owned future without replacing it with a
        // dependent stage. Returning the exact source preserves external
        // cancel propagation and keeps one publication owner end to end.
        resultFuture.whenComplete((result, throwable) -> {
            if (throwable != null) {
                return;
            }
            try {
                balanceContext.setResponse(result);
                if (result != null && result.isSuccess()) {
                    recentCacheKeyTraceReporter.report(balanceContext);
                }
            } catch (RuntimeException completionSideEffectFailure) {
                Logger.warn("Route completion side effect failed: request_id={}",
                        balanceContext.getRequestId(), completionSideEffectFailure);
            }
        });
        return resultFuture;
    }

    /**
     * BATCH retains its established compatibility behavior: an unavailable
     * scheduler or missing serialized generate input falls back to DIRECT.
     */
    private CompletableFuture<Response> routeBatch(BalanceContext balanceContext) {
        if (priorityScheduler == null || !hasValidGenerateInput(balanceContext)) {
            Logger.debug("BATCH mode cannot process this request, falling back to DIRECT");
            balanceContext.setScheduleMode(ScheduleModeEnum.DIRECT);
            return routeDirect(balanceContext);
        }
        return submitScheduled(balanceContext);
    }

    /**
     * Submit to the common scheduler. Route-decision requests intentionally do
     * not require generate_input: Master selects endpoints but the frontend
     * remains responsible for sending the original request to the engine.
     */
    private CompletableFuture<Response> submitScheduled(BalanceContext balanceContext) {
        if (priorityScheduler == null) {
            return CompletableFuture.failedFuture(new IllegalStateException(
                    "PriorityScheduler is required for the configured scheduling path"));
        }
        CompletableFuture<Response> resultFuture = priorityScheduler.submit(balanceContext);
        balanceContext.setFuture(resultFuture);
        return resultFuture;
    }

    private CompletableFuture<Response> routeDirect(BalanceContext balanceContext) {
        try {
            return CompletableFuture.completedFuture(router.route(balanceContext));
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    private boolean hasValidGenerateInput(BalanceContext ctx) {
        byte[] bytes = ctx.getGenerateInputPbBytes();
        return bytes != null && bytes.length > 0;
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return priorityScheduler == null ? null
                : priorityScheduler.getRequestState(requestId, expectedBatchId);
    }

    /**
     * Cancel one scheduler-owned request generation.
     *
     * <p>The scheduler is the only lifecycle and resource owner.  Keeping the
     * reducer there gives BATCH enqueue and QUEUE route-decision delivery the
     * same idempotency and generation-fencing semantics.</p>
     */
    public RequestLifecycleSnapshot cancelRequest(long requestId,
                                                   long expectedBatchId,
                                                   CancelReason reason) {
        return priorityScheduler == null ? null
                : priorityScheduler.cancelRequest(requestId, expectedBatchId, reason);
    }
}
