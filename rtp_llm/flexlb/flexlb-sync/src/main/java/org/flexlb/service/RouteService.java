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

        // Prefill seq_len admission gate: reject oversized requests before they
        // enter any scheduling/batch queue (fail-fast) — an oversized prompt can
        // OOM-crash the prefill engine, so it must never be dispatched.
        Response rejected = checkSeqLenLimit(balanceContext, flexlbConfig);
        if (rejected != null) {
            balanceContext.setResponse(rejected);
            return CompletableFuture.completedFuture(rejected);
        }

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
     * Prefill seq_len admission check. Returns a non-retryable INVALID_REQUEST
     * error response when the request's seq_len exceeds the configured
     * maxPrefillSeqLen, or {@code null} when the request is admitted.
     * A limit of 0 (default) disables the check.
     */
    private Response checkSeqLenLimit(BalanceContext ctx, FlexlbConfig config) {
        long maxSeqLen = config.getMaxPrefillSeqLen();
        if (maxSeqLen <= 0 || ctx.getRequest() == null) {
            return null;
        }
        long seqLen = ctx.getRequest().getSeqLen();
        if (seqLen <= maxSeqLen) {
            return null;
        }
        String message = "SEQ_LEN_EXCEEDED: seq_len=" + seqLen
                + " exceeds max_prefill_seq_len=" + maxSeqLen;
        Logger.warn("reject oversized prefill request: request_id={} seq_len={} max_prefill_seq_len={}",
                ctx.getRequestId(), seqLen, maxSeqLen);
        Response response = Response.error(StrategyErrorType.INVALID_REQUEST);
        response.setErrorMessage(StrategyErrorType.INVALID_REQUEST.buildErrorMessage(message));
        return response;
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        return flexlbBatchScheduler == null ? null
                : flexlbBatchScheduler.getRequestState(requestId, expectedBatchId);
    }
}
