package org.flexlb.service;

import com.google.protobuf.ByteString;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

@Component
public class RouteService {

    private final RequestScheduler requestScheduler;
    private final RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    public RouteService(RequestScheduler requestScheduler,
                        RecentCacheKeyTraceReporter recentCacheKeyTraceReporter) {
        this.requestScheduler = requestScheduler;
        this.recentCacheKeyTraceReporter = recentCacheKeyTraceReporter;
    }

    /**
     * Route request to appropriate workers based on the deployment-level schedule mode.
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public CompletableFuture<Response> route(BalanceContext balanceContext) {
        CompletableFuture<Response> resultFuture = routeScheduled(balanceContext);

        // Observe the scheduler-owned future without replacing it with a
        // dependent stage. Returning the exact source preserves external
        // cancel propagation and keeps one publication owner end to end.
        resultFuture.whenComplete((result, throwable) -> {
            if (throwable != null) {
                return;
            }
            try {
                balanceContext.setResponse(result);
                if (result != null && result.isSuccess()
                        && !balanceContext.getRequest().isVitOnly()) {
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
        if (!balanceContext.getRequest().isVitOnly()
                && balanceContext.getConfig().getDispatcher().requiresGenerateInput()
                && !hasValidGenerateInput(balanceContext)) {
            Logger.warn("{} dispatcher rejected request without serialized generate input: request_id={}",
                    balanceContext.getConfig().getDispatcher().typeName(),
                    balanceContext.getRequestId());
            return CompletableFuture.completedFuture(
                    Response.buildErrorResponse(StrategyErrorType.INVALID_REQUEST,
                            "missing serialized generate_input for batch dispatch"));
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
