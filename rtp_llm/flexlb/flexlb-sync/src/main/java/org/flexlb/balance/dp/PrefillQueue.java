package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Per-(model, prefill cluster, dp_group) batching buffer.
 *
 * <p>One queue collects requests targeting one Prefill worker (one DP group). When
 * any of these triggers fires, the queue drains a batch and hands it to the registered
 * flush callback (typically {@code DpBatchScheduler::flushBatch}):
 *
 * <ol>
 *   <li><b>Size:</b> queue length reaches {@code batchSizeMax} (typically = dp_size,
 *       so the batch fills exactly one DP cycle).</li>
 *   <li><b>Window:</b> the batch window timer (set when the first request arrives at
 *       an empty queue) elapses. Defends against medium-QPS where requests arrive
 *       slower than dp_size per window.</li>
 *   <li><b>Single-request timeout:</b> any pending request waiting longer than
 *       {@code requestTimeoutMs} forces an immediate flush regardless of the window.
 *       Defends low-QPS scenarios where the head request would otherwise be permanently
 *       stalled by the DP barrier.</li>
 * </ol>
 *
 * <h3>Thread safety</h3>
 * The {@code offer} / {@code flushOnSize} / {@code flushOnTimeout} entry points are
 * all synchronized on {@code this}. The drain is short and non-blocking, so contention
 * stays bounded. The flush callback is invoked outside the lock to avoid blocking
 * concurrent {@code offer} calls during gRPC dispatch.
 *
 * <h3>Bounded backlog</h3>
 * If multiple {@code drainAndFlush()} calls leave residual requests (e.g., a burst
 * of 10 requests with batchSizeMax=4), the residue fires another window timer and
 * flushes on the next trigger. No work is lost.
 */
public class PrefillQueue {

    private final ServerStatus prefillTarget;
    private final Consumer<PrefillBatch> onFlush;
    private final ScheduledExecutorService timerExecutor;
    private final LinkedBlockingDeque<PendingRequest> deque = new LinkedBlockingDeque<>();

    private final long batchWindowMs;
    private final long requestTimeoutMs;
    private final int batchSizeMax;
    private final int dpSize;

    /** Window timer; null when no in-flight timer (queue empty or just flushed). */
    private volatile ScheduledFuture<?> windowTimer;

    public PrefillQueue(ServerStatus prefillTarget,
                        int dpSize,
                        int batchSizeMax,
                        long batchWindowMs,
                        long requestTimeoutMs,
                        ScheduledExecutorService timerExecutor,
                        Consumer<PrefillBatch> onFlush) {
        if (dpSize <= 0) {
            throw new IllegalArgumentException("dpSize must be > 0, got " + dpSize);
        }
        if (batchSizeMax <= 0) {
            throw new IllegalArgumentException("batchSizeMax must be > 0, got " + batchSizeMax);
        }
        this.prefillTarget = prefillTarget;
        this.dpSize = dpSize;
        this.batchSizeMax = batchSizeMax;
        this.batchWindowMs = batchWindowMs;
        this.requestTimeoutMs = requestTimeoutMs;
        this.timerExecutor = timerExecutor;
        this.onFlush = onFlush;
    }

    /** Add a request and possibly trigger a size-based flush. */
    public void offer(PendingRequest req) {
        PrefillBatch toFlush = null;
        synchronized (this) {
            deque.add(req);
            // Single-request timeout takes precedence: if the head has been waiting
            // longer than requestTimeoutMs, drain whatever is pending now.
            if (headWaitedTooLong()) {
                toFlush = drainLocked();
            } else if (deque.size() >= batchSizeMax) {
                toFlush = drainLocked();
            } else if (windowTimer == null) {
                windowTimer = timerExecutor.schedule(
                        this::flushOnTimeout, batchWindowMs, TimeUnit.MILLISECONDS);
            }
        }
        // Dispatch outside the lock to avoid blocking concurrent offer() during gRPC.
        if (toFlush != null) {
            dispatch(toFlush);
        }
    }

    /** Window-timer callback. */
    private void flushOnTimeout() {
        PrefillBatch toFlush = null;
        synchronized (this) {
            windowTimer = null;
            if (!deque.isEmpty()) {
                toFlush = drainLocked();
            }
        }
        if (toFlush != null) {
            dispatch(toFlush);
        }
    }

    /**
     * Cancel current window timer, drain up to batchSizeMax, build a {@link PrefillBatch}.
     * If anything remains in the deque, restart the window timer for the next batch.
     * Caller MUST hold {@code synchronized (this)}.
     */
    private PrefillBatch drainLocked() {
        cancelTimerLocked();
        List<PendingRequest> taken = new ArrayList<>(batchSizeMax);
        deque.drainTo(taken, batchSizeMax);
        if (taken.isEmpty()) {
            return null;
        }
        // Residual requests beyond batchSizeMax stay; arm timer so they get flushed.
        if (!deque.isEmpty()) {
            windowTimer = timerExecutor.schedule(
                    this::flushOnTimeout, batchWindowMs, TimeUnit.MILLISECONDS);
        }
        return new PrefillBatch(prefillTarget, taken, dpSize);
    }

    private void cancelTimerLocked() {
        ScheduledFuture<?> t = windowTimer;
        if (t != null) {
            t.cancel(false);
            windowTimer = null;
        }
    }

    private boolean headWaitedTooLong() {
        PendingRequest head = deque.peek();
        return head != null && head.waitMicros() > requestTimeoutMs * 1000L;
    }

    private void dispatch(PrefillBatch batch) {
        try {
            onFlush.accept(batch);
        } catch (Throwable t) {
            // Don't let a bad flush callback poison future offers; report via futures.
            Logger.error("PrefillQueue flush callback threw, failing batch of {} requests", batch.size(), t);
            for (PendingRequest r : batch.requests()) {
                r.future().completeExceptionally(t);
            }
        }
    }

    /** Test/observability hook. */
    public int size() {
        return deque.size();
    }
}
