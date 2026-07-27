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
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.flexlb.util.Logger;
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
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        cancel(balanceContext, CancelReason.CLIENT_CANCELLED);
    }

    public void cancel(BalanceContext balanceContext, CancelReason reason) {
        balanceContext.cancel();
        ScheduleModeEnum mode = balanceContext.getScheduleMode();
        switch (mode) {
            case BATCH -> {
                if (flexlbBatchScheduler != null && balanceContext.getRequest() != null) {
                    flexlbBatchScheduler.cancel(balanceContext.getRequest().getRequestId(), reason, 0);
                }
            }
            case QUEUE -> queueManager.cancel(balanceContext);
            case DIRECT -> {
                // DIRECT path has no dedicated manager class; inline the release logic.
                Runnable releaseCallback = balanceContext.getDecodeReleaseCallback();
                if (releaseCallback != null) {
                    releaseCallback.run();
                } else {
                    long rid = balanceContext.getRequestId();
                    for (DecodeEndpoint ep : endpointRegistry.getDecodeEndpoints().values()) {
                        ep.release(rid);
                    }
                }
            }
            default -> Logger.warn("Unknown schedule mode {} in cancel, no-op", mode);
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }

    public RequestLifecycleSnapshot cancelByRequestId(long requestId,
                                                      CancelReason reason,
                                                      long expectedBatchId) {
        RequestLifecycleSnapshot snapshot = null;
        if (flexlbBatchScheduler != null) {
            snapshot = flexlbBatchScheduler.cancel(requestId, reason, expectedBatchId);
        }
        if (snapshot == null) {
            // Not in BATCH inflight — likely a DIRECT/QUEUE request whose decode
            // KV reservation is not tracked by the batch scheduler. Release it
            // via QueueManager, which brute-force iterates all decode endpoints.
            queueManager.cancelByRequestId(requestId);
        }
        return snapshot;
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return flexlbBatchScheduler == null ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
    }
}
