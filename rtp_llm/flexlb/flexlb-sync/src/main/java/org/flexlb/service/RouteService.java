package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

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
                        FlexlbBatchScheduler flexlbBatchScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.flexlbBatchScheduler = flexlbBatchScheduler;
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

        CompletableFuture<Response> resultFuture;
        switch (mode) {
            case BATCH -> {
                if (flexlbBatchScheduler == null) {
                    Logger.error("BATCH mode selected without a batch scheduler");
                    resultFuture = CompletableFuture.completedFuture(
                            Response.error(StrategyErrorType.BATCH_DISPATCH_FAILED));
                } else if (!hasValidGenerateInput(balanceContext)) {
                    resultFuture = CompletableFuture.completedFuture(
                            Response.error(StrategyErrorType.INVALID_REQUEST));
                } else {
                    resultFuture = flexlbBatchScheduler.submit(balanceContext);
                    balanceContext.setFuture(resultFuture);
                }
            }
            case QUEUE -> {
                resultFuture = queueManager.tryRouteAsync(balanceContext).toFuture();
            }
            case DIRECT -> {
                try {
                    resultFuture = CompletableFuture.completedFuture(router.route(balanceContext));
                } catch (Exception e) {
                    resultFuture = CompletableFuture.failedFuture(e);
                }
            }
            default -> {
                try {
                    resultFuture = CompletableFuture.completedFuture(router.route(balanceContext));
                } catch (Exception e) {
                    resultFuture = CompletableFuture.failedFuture(e);
                }
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

    private boolean hasValidGenerateInput(BalanceContext ctx) {
        byte[] bytes = ctx.getGenerateInputPbBytes();
        return bytes != null && bytes.length > 0;
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return flexlbBatchScheduler == null ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
    }
}
