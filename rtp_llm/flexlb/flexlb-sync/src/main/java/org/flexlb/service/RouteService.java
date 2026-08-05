package org.flexlb.service;

import org.flexlb.balance.scheduler.CancelHandler;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.CancelReasonPB;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService implements CancelHandler {

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
                if (flexlbBatchScheduler == null || !hasValidGenerateInput(balanceContext)) {
                    Logger.warn("BATCH mode cannot process this request, falling back to DIRECT");
                    balanceContext.setScheduleMode(ScheduleModeEnum.DIRECT);
                    try {
                        resultFuture = CompletableFuture.completedFuture(router.route(balanceContext));
                    } catch (Exception e) {
                        resultFuture = CompletableFuture.failedFuture(e);
                    }
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

    /**
     * {@link CancelHandler} entry — cancel an inflight request through the
     * scheduler cancel chain (local terminal handling + engine cancel RPC).
     */
    @Override
    public void cancel(long requestId, CancelReasonPB reason) {
        if (flexlbBatchScheduler != null) {
            flexlbBatchScheduler.cancelRequest(requestId, reason);
        }
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return flexlbBatchScheduler == null ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
    }
}
