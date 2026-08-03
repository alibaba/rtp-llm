package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.FlexlbMetricHelper;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A single inference request currently inflight (dispatched to an engine,
 * awaiting ACK or response).
 *
 * <p>Core v2 design: <b>binding-as-state</b> — the presence/absence of
 * {@link #prefillEp}, {@link #decodeEp}, and {@link #batch} references
 * implicitly expresses the progress phase, eliminating the need for a
 * separate state machine. A single {@link #terminated} AtomicBoolean
 * provides CAS-guarded idempotent terminal transition.
 *
 * <p>Thread-safety: the {@code terminated} AtomicBoolean ensures
 * {@link #terminate(TerminalReason)} succeeds at most once. All binding
 * fields are {@code volatile} so that other threads (e.g. the cancel path
 * via {@link InflightStore}) observe the latest endpoint/batch references
 * when performing cleanup.
 */
public final class InflightItem implements InflightEntry {

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final AbstractScheduler scheduler;
    private final String requestId;

    // ---- binding-as-state (volatile: set after construction) ----

    /** null = not routed, non-null = EP reserved. */
    volatile PrefillEndpoint prefillEp;

    /** null = not routed, non-null = EP reserved. */
    volatile DecodeEndpoint decodeEp;

    /** null = not in a batch, non-null = in a batch awaiting dispatch. */
    volatile Batch batch;

    // ---- single terminal flag ----

    /** CAS flag — {@code true} once the item reaches a terminal state. */
    final AtomicBoolean terminated = new AtomicBoolean(false);

    /** The reason for the terminal transition. Set after CAS succeeds. */
    volatile TerminalReason terminalReason;

    /** Wall-clock timestamp (ms) when the terminal state was entered. */
    volatile long terminalTime;

    // ---- dispatch tracking ----

    /** Set by {@link #ack(long)} — engine-assigned batch ID for stale ACK detection. */
    volatile long batchId;

    /** Set by {@link #ack(long)} — wall-clock ms when ACK was received. */
    volatile long dispatchedAtMs;

    // ---- progress timestamps (metrics/debug only, not control flow) ----

    volatile long enqueueTime;
    volatile long dispatchTime;
    volatile long ackTime;

    /** Flag set by {@link Batch#markItemFailed(InflightItem)} when batch is already dispatched. */
    volatile boolean failedInBatch;

    /**
     * Optional unified metric helper for terminal-state reporting.
     * Null-safe: if not set (null), no metrics are reported.
     */
    private volatile FlexlbMetricHelper metricHelper;

    // ---- constructor ----

    public InflightItem(BalanceContext ctx,
                        CompletableFuture<Response> future,
                        AbstractScheduler scheduler) {
        this.ctx = ctx;
        this.future = future;
        this.scheduler = scheduler;
        this.requestId = String.valueOf(ctx.getRequestId());
    }

    // ---- accessors ----

    public BalanceContext ctx() {
        return ctx;
    }

    public CompletableFuture<Response> future() {
        return future;
    }

    public AbstractScheduler scheduler() {
        return scheduler;
    }

    public PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    public DecodeEndpoint decodeEp() {
        return decodeEp;
    }

    public Batch batch() {
        return batch;
    }

    /** String-form request identifier used as the {@link InflightStore} key. */
    public String requestId() {
        return requestId;
    }

    public TerminalReason terminalReason() {
        return terminalReason;
    }

    public long batchId() {
        return batchId;
    }

    public long dispatchedAtMs() {
        return dispatchedAtMs;
    }

    // ---- binding setters (called by scheduler / queue logic after construction) ----

    public void setPrefillEp(PrefillEndpoint ep) {
        this.prefillEp = ep;
    }

    public void setDecodeEp(DecodeEndpoint ep) {
        this.decodeEp = ep;
    }

    void setBatch(Batch batch) {
        this.batch = batch;
    }

    /**
     * Set the optional unified metric helper for terminal-state reporting.
     * If not set (null), no unified metrics are reported on terminal transition.
     */
    public void setMetricHelper(FlexlbMetricHelper helper) {
        this.metricHelper = helper;
    }

    // ---- terminal state management ----

    /**
     * Atomically transition this item to a terminal state.
     *
     * <p>CAS-guarded: only the first caller wins. On success, releases EP-level
     * resources (Phase 2), notifies the batch, and completes the future
     * exceptionally with {@link TerminalReason#toException()}.
     *
     * @return {@code true} if this call won the CAS (first terminal transition),
     *         {@code false} if the item was already terminal
     */
    public boolean terminate(TerminalReason reason) {
        return terminate(reason, null);
    }

    /**
     * Overloaded terminate that preserves the original cause in the future completion.
     *
     * @param reason the terminal reason
     * @param cause  optional cause for the exceptional completion (null → use reason.toException())
     * @return {@code true} if this call won the CAS
     */
    public boolean terminate(TerminalReason reason, Throwable cause) {
        if (!terminated.compareAndSet(false, true)) return false;
        this.terminalReason = reason;
        this.terminalTime = System.currentTimeMillis();
        if (prefillEp != null) prefillEp.release(ctx.getRequestId());
        if (decodeEp != null) decodeEp.release(ctx.getRequestId());
        if (batch != null) {
            if (batch.isDispatched()) {
                batch.markItemFailed(this);
            } else {
                batch.removeItem(this);
            }
        }
        reportTerminalMetric(reason);
        future.completeExceptionally(cause != null ? cause : reason.toException());
        return true;
    }

    /**
     * Complete the request normally with the given response.
     *
     * <p>Shares the same {@code terminated} CAS flag as {@link #terminate(TerminalReason)},
     * so only one terminal transition (success or failure) can take effect.
     * On success, releases EP-level resources (Phase 2), removes the item from
     * its batch, and completes the future normally.
     */
    public void complete(Response response) {
        if (!terminated.compareAndSet(false, true)) return;
        this.terminalReason = response.isSuccess() ? TerminalReason.COMPLETED : TerminalReason.FAILED;
        this.terminalTime = System.currentTimeMillis();
        if (prefillEp != null) prefillEp.release(ctx.getRequestId());
        if (decodeEp != null) decodeEp.release(ctx.getRequestId());
        if (batch != null) {
            batch.removeItem(this);
        }
        reportTerminalMetric(this.terminalReason);
        future.complete(response);
    }

    /**
     * Cancel the request.
     *
     * @return {@code true} if this call won the CAS, {@code false} if already terminal (tombstone)
     */
    public boolean cancel() {
        return terminate(TerminalReason.CANCELLED);
    }

    /**
     * Fail the request with the given cause.
     *
     * @return {@code true} if this call won the CAS
     */
    public boolean fail(Throwable cause) {
        return terminate(TerminalReason.FAILED, cause);
    }

    /**
     * Time out the request.
     *
     * @return {@code true} if this call won the CAS
     */
    public boolean timeout() {
        return terminate(TerminalReason.TIMED_OUT);
    }

    // ---- ACK ----

    /**
     * Record the engine-assigned batch ID and ACK timestamp.
     *
     * @param batchId the engine-assigned batch ID (for stale ACK detection)
     */
    public void ack(long batchId) {
        this.batchId = batchId;
        this.dispatchedAtMs = System.currentTimeMillis();
    }

    // ---- batch-level failure marking ----

    /**
     * Called by {@link Batch#markItemFailed(InflightItem)} when the batch has
     * already been dispatched and a single item fails. The item cannot be
     * removed from a sent batch; instead it is marked so that when the batch
     * response arrives, the item is handled as failed.
     */
    void markFailedInBatch() {
        this.failedInBatch = true;
    }

    public boolean isFailedInBatch() {
        return failedInBatch;
    }

    // ---- terminal metric reporting ----

    /**
     * Report the terminal transition via the unified metric helper (if set).
     * Called exactly once, inside the CAS-guarded {@link #terminate} or {@link #complete}.
     */
    private void reportTerminalMetric(TerminalReason reason) {
        FlexlbMetricHelper helper = this.metricHelper;
        if (helper == null) {
            return;
        }
        String role = resolveRole();
        String engineIp = resolveEngineIp();
        helper.reportTerminal(reason, role, engineIp, null);
    }

    /**
     * Resolve the engine role from the available endpoint references.
     * Prefers the prefill endpoint, then decode. Falls back to "UNKNOWN"
     * when neither is set (e.g. items not yet routed).
     */
    private String resolveRole() {
        if (prefillEp != null) {
            return "PREFILL";
        }
        if (decodeEp != null) {
            return "DECODE";
        }
        return "UNKNOWN";
    }

    /**
     * Resolve the engine IP from the available endpoint references.
     * Falls back to "unknown" when neither endpoint is set.
     */
    private String resolveEngineIp() {
        if (prefillEp != null) {
            return prefillEp.getIp();
        }
        if (decodeEp != null) {
            return decodeEp.getIp();
        }
        return "unknown";
    }

    // ---- queries used by InflightStore TTL eviction ----

    public boolean isTerminated() {
        return terminated.get();
    }

    public long getTerminalTime() {
        return terminalTime;
    }
}
