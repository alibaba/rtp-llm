package org.flexlb.service;

import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.DpBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

/**
 * High-level routing entry. Selects between three downstream paths based on config:
 *
 * <ol>
 *   <li><b>DP batching</b> ({@code dpBalanceEnabled=true} + per-request gate ok):
 *       hand off to {@link DpBatchScheduler} which globally batches requests across the
 *       same Prefill cluster, assigns dp_rank, and async-Enqueues to Prefill. The HTTP
 *       caller's Mono completes with {@code enqueued_by_master=true} so frontend
 *       switches to {@code Decode.FetchResponse}. See V1-α plan.</li>
 *   <li><b>Queueing</b> ({@code enableQueueing=true}): enqueue into {@link QueueManager}
 *       and let {@code RequestScheduler} workers consume + route via {@link Router}.</li>
 *   <li><b>Direct</b>: synchronous {@link Router#route} on the calling thread.</li>
 * </ol>
 *
 * <p>Cancellation: when DP batching is in effect, {@link #cancel} additionally cascades
 * a Cancel RPC to the Prefill + Decode workers via {@link DpBatchScheduler#cancel}.
 */
@Component
public class RouteService {

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final DpBatchScheduler dpBatchScheduler;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        @Autowired(required = false) DpBatchScheduler dpBatchScheduler) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.dpBatchScheduler = dpBatchScheduler;
    }

    /**
     * Route request to appropriate workers
     * @param balanceContext Load balancing context
     * @return Routing result
     */
    public Mono<Response> route(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        balanceContext.setConfig(flexlbConfig);

        Mono<Response> resultMono;
        if (shouldUseDpBatch(balanceContext, flexlbConfig)) {
            // DP batch path: DpBatchScheduler returns a CompletableFuture that completes
            // when the request's batch is flushed and Master.Enqueue is acked.
            CompletableFuture<Response> f = dpBatchScheduler.submit(balanceContext);
            balanceContext.setFuture(f);  // keep parity with queue path so cancel() can find it
            resultMono = Mono.fromFuture(f);
        } else if (flexlbConfig.isEnableQueueing()) {
            resultMono = queueManager.tryRouteAsync(balanceContext);  // Use async queuing mechanism
        } else {
            resultMono = Mono.fromCallable(() -> router.route(balanceContext));  // Direct routing without queuing
        }

        return resultMono.doOnSuccess(result -> {
            balanceContext.setResponse(result);
        });
    }

    /**
     * Cancel a specified request
     * @param balanceContext Load balancing context
     */
    public void cancel(BalanceContext balanceContext) {
        FlexlbConfig flexlbConfig = configService.loadBalanceConfig();
        if (flexlbConfig.isEnableQueueing()) {
            balanceContext.cancel();
            CompletableFuture<Response> future = balanceContext.getFuture();
            if (future != null) {
                future.completeExceptionally(new CancellationException("Request cancelled by client"));
            }
        }
        // Cascade cancel to Prefill + Decode if the request was Master-enqueued.
        // Safe to call unconditionally: DpBatchScheduler.cancel is a no-op on unknown ids.
        if (dpBatchScheduler != null && balanceContext.getRequest() != null) {
            dpBatchScheduler.cancel(balanceContext.getRequest().getRequestId());
        }
        balanceContext.setSuccess(false);
        balanceContext.setErrorMessage("request cancelled");
    }

    /**
     * Cancel by request_id without an existing {@link BalanceContext}.
     * <p>
     * Called from HTTP {@code POST /rtp_llm/cancel} when frontend explicitly cancels
     * an in-flight request after Master has Enqueued it (the request's original HTTP
     * Mono has already returned with {@code enqueued_by_master=true}).
     */
    public void cancelByRequestId(long requestId) {
        if (dpBatchScheduler != null) {
            dpBatchScheduler.cancel(requestId);
        }
    }

    /**
     * V1-α gate: only route through DpBatchScheduler when:
     * <ul>
     *   <li>config flag is on,</li>
     *   <li>scheduler bean is wired (defensive: optional injection above means
     *       early-stage builds can omit it without crashing),</li>
     *   <li>request will produce more than one token AND not use beam search AND
     *       hasn't disabled SP — the same precondition under which pd_separation
     *       activates in the engine. SP/beam paths still rely on the legacy
     *       direct or queued router.</li>
     * </ul>
     */
    boolean shouldUseDpBatch(BalanceContext ctx, FlexlbConfig cfg) {
        if (dpBatchScheduler == null) return false;
        if (!cfg.isDpBalanceEnabled()) return false;
        Request req = ctx.getRequest();
        if (req == null) return false;
        if (req.getMaxNewTokens() <= 1) return false;     // pd_separation gating
        if (req.getNumBeams() > 1) return false;           // pd_separation gating
        if (req.isForceDisableSpRun()) return false;       // explicit SP opt-out
        return true;
    }
}
