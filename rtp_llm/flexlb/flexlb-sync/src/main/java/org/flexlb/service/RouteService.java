package org.flexlb.service;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

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
    public CompletableFuture<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        // Resolve AUTO to actual schedule mode so downstream components
        // (e.g., CostBasedDecodeStrategy) can make mode-aware decisions
        // such as skipping KV reserve for DIRECT/QUEUE paths.
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

    /**
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        cancel(balanceContext, CancelReason.CLIENT_CANCELLED);
    }

    public void cancel(BalanceContext balanceContext, CancelReason reason) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.cancel();
        if (flexlbConfig.isEnableQueueing() || balanceContext.getScheduleMode() == ScheduleModeEnum.QUEUE) {
            CompletableFuture<Response> future = balanceContext.getFuture();
            if (future != null) {
                future.completeExceptionally(new CancellationException("Request cancelled by client"));
            }
        }
        if (flexlbBatchScheduler != null && balanceContext.getRequest() != null) {
            flexlbBatchScheduler.cancel(balanceContext.getRequest().getRequestId(), reason, 0);
        }
        // Note: DIRECT and QUEUE paths skip de.reserve() in CostBasedDecodeStrategy,
        // so there is no decode KV reservation to release on cancel.
        // BATCH path's reservation is cleaned by flexlbBatchScheduler.cancel() above,
        // which calls rollbackOnce() → de.release() for the decode endpoint.
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }

    public RequestLifecycleSnapshot cancelByRequestId(long requestId,
                                                      CancelReason reason,
                                                      long expectedBatchId) {
        // Only BATCH path has decode KV reservations tracked by flexlbBatchScheduler.
        // DIRECT and QUEUE paths skip reserve, so cancel is a no-op for them.
        if (flexlbBatchScheduler != null) {
            return flexlbBatchScheduler.cancel(requestId, reason, expectedBatchId);
        }
        return null;
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return flexlbBatchScheduler == null ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
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
