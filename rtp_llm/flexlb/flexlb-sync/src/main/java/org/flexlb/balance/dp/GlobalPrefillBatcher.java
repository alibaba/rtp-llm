package org.flexlb.balance.dp;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Per-model global queue for DP batching.
 *
 * <h3>Why global instead of per-group</h3>
 * Per-group queues only fill efficiently at high QPS (each group needs
 * {@code ≥ dpSize} requests inside one window). At low/medium QPS, a per-group
 * queue partial-flushes at every window, sending many tiny batches. A single
 * global queue accumulates across all groups; once {@code dpSize} requests have
 * arrived (regardless of which group they will eventually land on), one full
 * batch is built and shipped to a {@link GroupSelector}-chosen group. The next
 * batch goes to the next group via the cursor.
 *
 * <h3>Triggers</h3>
 * <ol>
 *   <li><b>Size:</b> {@code queue.size() >= dpSize} — flush immediately on the
 *       offer thread.</li>
 *   <li><b>Window:</b> if no size flush armed, schedule a one-shot timer to
 *       flush after {@code batchWindowMs}.</li>
 *   <li><b>Per-request timeout:</b> the head request having waited longer than
 *       {@code requestTimeoutMs} forces an immediate flush regardless of
 *       window — protects starvation under low QPS.</li>
 * </ol>
 *
 * <h3>Concurrency</h3>
 * {@code offer} / {@code flushOnTimeout} are serialized on {@code this}; the
 * drain loop is short and non-blocking. Assembly + dispatch run OUTSIDE the
 * lock so concurrent {@code offer} calls are not blocked by gRPC.
 *
 * <h3>Cancellation</h3>
 * {@link #cancelInQueue} is best-effort: requests still queued are removed in
 * O(N). A request that was already drained but not yet registered with
 * {@code InflightBatchRegistry} is in a small race window; {@code RouteService}
 * always completes the future exceptionally first, so a successful dispatch in
 * that window is wasted work but not a correctness bug.
 */
public class GlobalPrefillBatcher {

    private final String model;
    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final DispatchPlanner planner;
    private final Consumer<PrefillBatch> dispatchCallback;
    private final ScheduledExecutorService timerExecutor;

    private final LinkedBlockingDeque<QueuedRequest> queue = new LinkedBlockingDeque<>();

    /** One-shot window timer; null when none armed (queue empty or just flushed). */
    private volatile ScheduledFuture<?> windowTimer;

    public GlobalPrefillBatcher(String model,
                                ConfigService configService,
                                EngineWorkerStatus engineWorkerStatus,
                                DispatchPlanner planner,
                                Consumer<PrefillBatch> dispatchCallback,
                                ScheduledExecutorService timerExecutor) {
        this.model = model;
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.planner = planner;
        this.dispatchCallback = dispatchCallback;
        this.timerExecutor = timerExecutor;
    }

    public void offer(QueuedRequest req) {
        FlexlbConfig cfg = configService.loadBalanceConfig();
        int dpSize = currentDpSize(cfg);
        long batchWindowMs = cfg.getDpBatchWindowMs();
        long requestTimeoutMs = cfg.getDpBatchTimeoutMs();

        List<QueuedRequest> drained = null;
        synchronized (this) {
            queue.add(req);
            if (headWaitedTooLong(requestTimeoutMs)) {
                drained = drainLocked(dpSize, batchWindowMs);
            } else if (queue.size() >= dpSize) {
                drained = drainLocked(dpSize, batchWindowMs);
            } else if (windowTimer == null) {
                windowTimer = timerExecutor.schedule(
                        this::flushOnTimeout, batchWindowMs, TimeUnit.MILLISECONDS);
            }
        }
        if (drained != null) {
            planAndDispatch(drained, dpSize, cfg);
        }
    }

    /**
     * Best-effort removal of a queued request. Returns {@code true} if the entry
     * was found and removed; {@code false} if the request had already been
     * drained (it may be mid-planning or already enqueued).
     *
     * <p>On removal, the request's future is completed exceptionally with a
     * {@link CancellationException} so the upstream {@code Mono} never hangs.
     * Double-completion (when {@code RouteService.cancel} also fails the
     * future) is harmless — {@code completeExceptionally} returns false on a
     * future that is already done.
     */
    public boolean cancelInQueue(long requestId) {
        Iterator<QueuedRequest> it = queue.iterator();
        while (it.hasNext()) {
            QueuedRequest qr = it.next();
            if (qr.requestId() == requestId) {
                it.remove();
                qr.future().completeExceptionally(
                        new CancellationException("Cancelled while queued in DP batcher"));
                return true;
            }
        }
        return false;
    }

    /** Test/observability hook. */
    public int queueSize() {
        return queue.size();
    }

    private void flushOnTimeout() {
        FlexlbConfig cfg = configService.loadBalanceConfig();
        int dpSize = currentDpSize(cfg);
        long batchWindowMs = cfg.getDpBatchWindowMs();

        List<QueuedRequest> drained = null;
        synchronized (this) {
            windowTimer = null;
            if (!queue.isEmpty()) {
                drained = drainLocked(dpSize, batchWindowMs);
            }
        }
        if (drained != null) {
            planAndDispatch(drained, dpSize, cfg);
        }
    }

    /** Caller MUST hold {@code synchronized (this)}. */
    private List<QueuedRequest> drainLocked(int dpSize, long batchWindowMs) {
        cancelTimerLocked();
        List<QueuedRequest> chunk = new ArrayList<>(dpSize);
        queue.drainTo(chunk, dpSize);
        if (chunk.isEmpty()) {
            return null;
        }
        // Residual requests (when one drain doesn't empty the queue) re-arm the
        // window timer for the next batch.
        if (!queue.isEmpty()) {
            windowTimer = timerExecutor.schedule(
                    this::flushOnTimeout, batchWindowMs, TimeUnit.MILLISECONDS);
        }
        return chunk;
    }

    private void cancelTimerLocked() {
        ScheduledFuture<?> t = windowTimer;
        if (t != null) {
            t.cancel(false);
            windowTimer = null;
        }
    }

    private boolean headWaitedTooLong(long requestTimeoutMs) {
        QueuedRequest head = queue.peek();
        return head != null && head.waitMicros() > requestTimeoutMs * 1000L;
    }

    private void planAndDispatch(List<QueuedRequest> drained, int dpSize, FlexlbConfig cfg) {
        DispatchPlan result;
        try {
            result = planner.plan(drained, new DispatchContext(model, dpSize, cfg));
        } catch (Throwable t) {
            Logger.error("DispatchPlanner threw on {} requests; failing all", drained.size(), t);
            failAll(drained, t);
            return;
        }

        for (FailedRequest fr : result.failures()) {
            fr.request().future().complete(failureResponse(fr.reason(), fr.message()));
        }
        for (PrefillBatch batch : result.batches()) {
            try {
                dispatchCallback.accept(batch);
            } catch (Throwable t) {
                Logger.error("dispatch callback threw on batch of {}; failing all in batch",
                        batch.size(), t);
                for (PendingRequest pr : batch.requests()) {
                    pr.future().completeExceptionally(t);
                }
            }
        }
    }

    private static void failAll(List<QueuedRequest> chunk, Throwable cause) {
        for (QueuedRequest qr : chunk) {
            qr.future().completeExceptionally(cause);
        }
    }

    private static Response failureResponse(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }

    private int currentDpSize(FlexlbConfig cfg) {
        int configured = cfg.getDpBatchSizeMax();
        if (configured > 0) {
            return configured;
        }
        Map<String, WorkerStatus> workers = engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (workers == null || workers.isEmpty()) {
            return 1;
        }
        for (WorkerStatus w : workers.values()) {
            if (w != null && w.getDpSize() > 1) {
                return (int) w.getDpSize();
            }
        }
        return 1;
    }
}
