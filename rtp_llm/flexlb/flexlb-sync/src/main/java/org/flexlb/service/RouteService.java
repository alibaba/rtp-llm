package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.DpBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

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

    private static final Logger logger = LoggerFactory.getLogger(RouteService.class);

    private final ConfigService configService;
    private final Router router;
    private final QueueManager queueManager;
    private final DpBatchScheduler dpBatchScheduler;
    private final EngineWorkerStatus engineWorkerStatus;

    public RouteService(ConfigService configService,
                        DefaultRouter defaultScheduler,
                        QueueManager queueManager,
                        EngineWorkerStatus engineWorkerStatus,
                        @Autowired(required = false) DpBatchScheduler dpBatchScheduler) {
        this.configService = configService;
        this.router = defaultScheduler;
        this.queueManager = queueManager;
        this.engineWorkerStatus = engineWorkerStatus;
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
     * Gate for DpBatchScheduler lane. All five conditions must hold:
     * <ul>
     *   <li>scheduler bean is wired (optional injection — early-stage builds may omit it),</li>
     *   <li>{@code dpBalanceEnabled} config flag is on,</li>
     *   <li>request will produce more than one token AND not use beam search AND
     *       hasn't disabled SP — the same precondition under which pd_separation
     *       activates in the engine; SP/beam paths still take the legacy router,</li>
     *   <li>at least one alive Prefill worker exists. Multi-rank workers route
     *       to {@code GlobalPrefillBatcher}; single-rank ({@code dp_size == 1})
     *       workers route to {@code SloBudgetBatcher} (FIFO + SLO-budget batching).</li>
     * </ul>
     */
    boolean shouldUseDpBatch(BalanceContext ctx, FlexlbConfig cfg) {
        boolean schedulerOk = dpBatchScheduler != null;
        boolean cfgOk = cfg.isDpBalanceEnabled();
        Request req = ctx.getRequest();
        boolean reqOk = req != null;
        int maxNewTokens = reqOk ? req.getMaxNewTokens() : -1;
        int numBeams = reqOk ? req.getNumBeams() : -1;
        boolean forceDisableSp = reqOk && req.isForceDisableSpRun();
        boolean prefillWorkerOk = hasAlivePrefillWorker();
        boolean decision = schedulerOk && cfgOk && reqOk
                && maxNewTokens > 1 && numBeams <= 1 && !forceDisableSp && prefillWorkerOk;
        logger.warn("dp-batch gate decision={} scheduler={} cfgDpBalance={} req={} maxNewTokens={} numBeams={} forceDisableSp={} prefillWorker={}",
                decision, schedulerOk, cfgOk, reqOk, maxNewTokens, numBeams, forceDisableSp, prefillWorkerOk);
        return decision;
    }

    private boolean hasAlivePrefillWorker() {
        Map<String, WorkerStatus> prefillWorkers =
                engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (prefillWorkers == null || prefillWorkers.isEmpty()) {
            logger.warn("dp-batch gate prefillWorkers={}",
                    prefillWorkers == null ? "null" : "empty");
            return false;
        }
        boolean found = false;
        for (WorkerStatus w : prefillWorkers.values()) {
            if (w != null && w.isAlive()) {
                found = true;
                break;
            }
        }
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, WorkerStatus> e : prefillWorkers.entrySet()) {
            WorkerStatus w = e.getValue();
            sb.append("[").append(e.getKey()).append(":dpSize=")
              .append(w == null ? "null" : w.getDpSize()).append(",alive=")
              .append(w == null ? "null" : w.isAlive()).append("]");
        }
        logger.warn("dp-batch gate prefillWorkers count={} found={} workers={}",
                prefillWorkers.size(), found, sb);
        return found;
    }
}
