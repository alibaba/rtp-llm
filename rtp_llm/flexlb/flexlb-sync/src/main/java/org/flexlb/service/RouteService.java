package org.flexlb.service;

import com.google.protobuf.ByteString;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final ConfigService configService;
    private final DefaultRouter router;
    private final RequestScheduler requestScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        RequestScheduler requestScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.requestScheduler = requestScheduler;
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

        CompletableFuture<Response> resultFuture;
        if (flexlbConfig.isDirect()) {
            resultFuture = routeDirect(balanceContext);
        } else {
            resultFuture = routeScheduled(balanceContext);
        }

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

    private CompletableFuture<Response> routeScheduled(BalanceContext balanceContext) {
        if (requestScheduler == null) {
            return CompletableFuture.failedFuture(new IllegalStateException(
                    "RequestScheduler is required for the configured scheduling path"));
        }
        if (balanceContext.getConfig().getDispatcher().requiresGenerateInput()
                && !hasValidGenerateInput(balanceContext)) {
            Logger.warn("{} dispatcher rejected request without serialized generate input: request_id={}",
                    balanceContext.getConfig().getDispatcher().typeName(),
                    balanceContext.getRequestId());
            return CompletableFuture.completedFuture(
                    Response.error(StrategyErrorType.BATCH_BUILD_FAILED));
        }
        return submitScheduled(balanceContext);
    }

    /**
     * Submit to the common scheduler. Route-decision requests intentionally do
     * not require generate_input: Master selects endpoints but the frontend
     * remains responsible for sending the original request to the engine.
     */
    private CompletableFuture<Response> submitScheduled(BalanceContext balanceContext) {
        if (requestScheduler == null) {
            return CompletableFuture.failedFuture(new IllegalStateException(
                    "RequestScheduler is required for the configured scheduling path"));
        }
        CompletableFuture<Response> resultFuture = requestScheduler.submit(balanceContext);
        balanceContext.setFuture(resultFuture);
        return resultFuture;
    }

    private CompletableFuture<Response> routeDirect(BalanceContext balanceContext) {
        try {
            if (balanceContext.requestExpired(System.currentTimeMillis())) {
                return CompletableFuture.completedFuture(
                        Response.error(StrategyErrorType.BATCH_SLO_EXPIRED));
            }
            return CompletableFuture.completedFuture(
                    router.routeDirect(balanceContext));
        } catch (Exception e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    private boolean hasValidGenerateInput(BalanceContext ctx) {
        ByteString generateInput = ctx.getGenerateInputPb();
        return generateInput != null && !generateInput.isEmpty();
    }

    public RequestState getRequestState(long requestId,
                                                    long expectedBatchId) {
        return requestScheduler == null ? null
                : requestScheduler.getRequestState(requestId, expectedBatchId);
    }

    /**
     * Cancel one scheduler-owned request generation.
     *
     * <p>The scheduler is the only lifecycle and resource owner.  Keeping the
     * reducer there gives BATCH enqueue and QUEUE route-decision delivery the
     * same idempotency and generation-fencing semantics.</p>
     */
    public RequestState cancelRequest(long requestId,
                                                   long expectedBatchId,
                                                   CancelReason reason) {
        return requestScheduler == null ? null
                : requestScheduler.cancelRequest(requestId, expectedBatchId, reason);
    }
}
