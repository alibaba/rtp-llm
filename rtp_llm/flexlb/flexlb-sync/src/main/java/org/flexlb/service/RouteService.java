package org.flexlb.service;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final FlexlbBatchScheduler flexlbBatchScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    public RouteService(
            ConfigService configService,
            Router router,
            QueueManager queueManager,
            FlexlbBatchScheduler flexlbBatchScheduler,
            RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.configService = configService;
        this.router = router;
        this.queueManager = queueManager;
        this.flexlbBatchScheduler = flexlbBatchScheduler;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
    }

    /**
     * Route request to appropriate workers based on the deployment-level schedule mode.
     *
     * @param balanceContext load-balancing context
     * @return asynchronous routing result
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
                    Logger.debug("BATCH mode cannot process this request, falling back to DIRECT");
                    balanceContext.setScheduleMode(ScheduleModeEnum.DIRECT);
                    resultFuture = routeDirect(balanceContext);
                } else {
                    resultFuture = flexlbBatchScheduler.submit(balanceContext);
                    balanceContext.setFuture(resultFuture);
                }
            }
            case QUEUE -> resultFuture = queueManager.tryRouteAsync(balanceContext).toFuture();
            case DIRECT -> resultFuture = routeDirect(balanceContext);
            default -> resultFuture = routeDirect(balanceContext);
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

    private CompletableFuture<Response> routeDirect(BalanceContext context) {
        try {
            return CompletableFuture.completedFuture(router.route(context));
        } catch (Exception error) {
            return CompletableFuture.failedFuture(error);
        }
    }

    private boolean hasValidGenerateInput(BalanceContext context) {
        byte[] bytes = context.getGenerateInputPbBytes();
        return bytes != null && bytes.length > 0;
    }

    public RequestLifecycleSnapshot getRequestState(long requestId, long expectedBatchId) {
        return flexlbBatchScheduler == null
                ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
    }

    /**
     * Resolve a whole dispatcher chunk atomically for a single-role deployment.
     * This intentionally bypasses the normal request queue and its per-request lifecycle.
     */
    public Mono<BatchScheduleResponse> batchSchedule(BatchScheduleRequest request) {
        return Mono.fromCallable(() -> router.batchSchedule(request))
                .subscribeOn(Schedulers.parallel());
    }
}
