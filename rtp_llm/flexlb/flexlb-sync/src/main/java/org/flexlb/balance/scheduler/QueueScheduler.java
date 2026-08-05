package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.Logger;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

/**
 * Scheduler for QUEUE mode: enqueue → worker dequeues → route → complete.
 *
 * <p>Extends {@link DirectScheduler} and reuses its route-and-complete logic
 * for the worker consume path, overriding {@link #onRouteResult} to intercept
 * retryable failures (re-queued at the head via
 * {@link QueueingComponent#requeueHead} with a configurable retry cap).
 *
 * <p>The queueing capability is fully encapsulated in the composed
 * {@link QueueingComponent} (bounded deque + permit-gated worker pool);
 * lifecycle follows this scheduler ({@link #start()} / {@link #shutdown()}).
 *
 * <p>Per-request flow in {@link #submit}:
 * <ol>
 *   <li>Create the raw worker future ({@code ctx.getFuture()}) that worker
 *       threads complete</li>
 *   <li>Wrap it in a reactive pipeline: generate-timeout, queue-exception
 *       mapping, and route-execution-time reporting</li>
 *   <li>Register the pipeline-derived future in the global
 *       {@link InflightStore} so that timeout-driven terminal transitions are
 *       captured (cancel completes the raw future exceptionally through the
 *       item, which propagates through the pipeline)</li>
 *   <li>Enqueue; a full queue completes the request with
 *       {@link StrategyErrorType#QUEUE_FULL}</li>
 * </ol>
 */
public class QueueScheduler extends DirectScheduler {

    private final RoutingQueueReporter queueReporter;
    private final QueueingComponent queueing;

    public QueueScheduler(Router router,
                          ConfigService configService,
                          RoutingQueueReporter queueReporter,
                          DynamicWorkerManager dynamicWorkerManager,
                          InflightStore globalStore,
                          FlexlbMetricHelper metricHelper) {
        super(router, globalStore, metricHelper);
        this.queueReporter = queueReporter;
        this.queueing = new QueueingComponent(
                configService, queueReporter, dynamicWorkerManager, this::consume);
    }

    // ==================== Lifecycle ====================

    /** Start the queue consumer worker pool. */
    public void start() {
        queueing.start();
    }

    public void shutdown() {
        queueing.shutdown();
    }

    // ==================== Submit path ====================

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        // Raw future completed by worker threads (or the queue-full fast path).
        CompletableFuture<Response> workerFuture = new CompletableFuture<>();
        ctx.setFuture(workerFuture);

        // Reactive pipeline built before enqueue so no completion can be missed:
        // generate-timeout, queue exception mapping, route execution reporting.
        CompletableFuture<Response> resultFuture = Mono.fromFuture(workerFuture)
                .timeout(Duration.ofMillis(ctx.getRequest().getGenerateTimeout()))
                .onErrorResume(e -> handleQueueException(ctx, e))
                .doFinally(signalType -> {
                    if (ctx.getDequeueTime() > 0) {
                        long routeExecutionTimeMs = System.currentTimeMillis() - ctx.getDequeueTime();
                        queueReporter.reportRouteExecutionMetric(routeExecutionTimeMs);
                    }
                })
                .toFuture();

        // Track the pipeline-derived future so timeout / cancel transitions
        // are observed as terminal states. Duplicates proceed untracked —
        // queue semantics never reject a request for tracking reasons.
        InflightItem existing = register(ctx, resultFuture);
        if (existing != null) {
            Logger.warn("Duplicate request_id: {}, processing untracked", ctx.getRequestId());
        }

        if (!queueing.enqueue(ctx)) {
            workerFuture.complete(Response.error(StrategyErrorType.QUEUE_FULL));
        }
        return resultFuture;
    }

    // ==================== Consume path (worker threads) ====================

    /**
     * Dequeued-request handler: reuse the parent's route-and-complete logic
     * against the raw worker future. Retry interception happens in
     * {@link #onRouteResult}.
     */
    private void consume(BalanceContext ctx) {
        routeAndComplete(ctx, ctx.getFuture());
    }

    /**
     * Cancel cascade: best-effort removal of the request from the routing
     * queue so a cancelled request never keeps occupying a queue slot. The
     * cancel already settled the pipeline future via the item's CAS; the raw
     * worker future stays incomplete, so the worker-loop settled-skip is the
     * second line of defence if the removal races a concurrent dequeue.
     */
    @Override
    public void onCancel(InflightItem item) {
        queueing.removeIfQueued(item.ctx());
    }

    /**
     * Retry-aware completion policy: retryable routing failures are re-queued
     * at the head (bounded by {@code maxRetryCount}; {@code <= 0} means
     * unlimited); everything else completes the future.
     */
    @Override
    protected void onRouteResult(BalanceContext ctx, CompletableFuture<Response> future,
                                 Response response) {
        int maxRetry = ctx.getConfig() != null ? ctx.getConfig().getMaxRetryCount() : 0;
        boolean retryAllowed = maxRetry <= 0 || ctx.getRetryCount() < maxRetry;
        if (!response.isSuccess() && shouldRetry(response) && retryAllowed) {
            ctx.incrementRetryCount();
            Logger.warn("Route failed for request id:{}, error: {}, retry count: {}",
                    ctx.getRequestId(),
                    response.getCode(),
                    ctx.getRetryCount());
            queueReporter.reportRoutingFailureQps(response.getCode());

            queueing.requeueHead(ctx);
        } else {
            if (!response.isSuccess() && !retryAllowed) {
                Logger.warn("Max retry count ({}) exceeded for request id:{}, completing with error",
                        maxRetry, ctx.getRequestId());
            }
            future.complete(response);
            queueReporter.reportRoutingSuccessQps(ctx.getRetryCount());
        }
    }

    /**
     * Determine if a request should be retried based on error code range:
     * 8000-8999 are retryable (transient failures, resource unavailable);
     * 4000-4999 are non-retryable (invalid request, queue full, persistent errors).
     */
    private static boolean shouldRetry(Response response) {
        return StrategyErrorType.isRetryableCode(response.getCode());
    }

    // ==================== Exception mapping (reactive pipeline) ====================

    private Mono<Response> handleQueueException(BalanceContext ctx, Throwable e) {
        // Handle ExecutionException wrapper (consistent with synchronous version)
        Throwable cause = e instanceof ExecutionException ? e.getCause() : e;
        if (cause instanceof TimeoutException) {
            queueing.remove(ctx);
            queueReporter.reportTimeout();
            long waitTimeMs = System.currentTimeMillis() - ctx.getEnqueueTime();
            Logger.warn("Request timeout in queue for id: {}, wait time: {}ms", ctx.getRequestId(), waitTimeMs);
            return Mono.just(Response.error(StrategyErrorType.QUEUE_TIMEOUT));
        } else if (cause instanceof InterruptedException) {
            queueing.remove(ctx);
            Thread.currentThread().interrupt();
            Logger.error("Request interrupted while waiting in queue for id: {}", ctx.getRequestId());
            return Mono.just(Response.error(StrategyErrorType.QUEUE_TIMEOUT));
        }
        // Other exceptions: log and return NO_AVAILABLE_WORKER (consistent with synchronous version)
        Logger.error("Request execution failed error: {}", e);
        return Mono.just(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
    }

    // ==================== Diagnostics ====================

    /** Current routing queue length (requests waiting for a worker). */
    public int queueSize() {
        return queueing.queueSize();
    }

    /** Dump the queued requests to a JSON snapshot file. */
    public QueueSnapshotResponse snapshotQueue() {
        return queueing.snapshot();
    }

    /**
     * Report QUEUE-specific metrics: the queue length both as the unified
     * path-tagged inflight gauge and the legacy routing-queue-length gauge.
     */
    @Override
    public void reportMetrics() {
        metricHelper.reportInflightSize("PREFILL", "scheduler", queueing.queueSize());
        queueing.reportQueueSize();
    }
}
