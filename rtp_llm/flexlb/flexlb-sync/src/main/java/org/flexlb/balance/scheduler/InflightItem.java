package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.monitor.FlexlbMetricHelper;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A single inference request currently inflight (dispatched to an engine,
 * awaiting ACK or response).
 *
 * <p>Core v2 design: <b>binding-as-state</b> — the presence/absence of
 * {@link #prefillEp}, {@link #decodeEp}, and {@link #batch} references
 * implicitly expresses the progress phase, eliminating the need for a
 * separate state machine. A single {@link #state} AtomicReference
 * provides CAS-guarded idempotent terminal transition.
 *
 * <p>Thread-safety: the {@code state} AtomicReference ensures
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

    /** Wall-clock timestamp (ms) when this item was created. Used for TTL expiry check. */
    final long createdAtMs = System.currentTimeMillis();

    // ---- binding-as-state (volatile: set after construction) ----

    /** null = not routed, non-null = EP reserved. */
    volatile PrefillEndpoint prefillEp;

    /** null = not routed, non-null = EP reserved. */
    volatile DecodeEndpoint decodeEp;

    /** null = not in a batch, non-null = in a batch awaiting dispatch. */
    volatile Batch batch;

    // ---- single atomic state ----

    /**
     * CAS-guarded lifecycle state. Transitions from {@link InflightState#RUNNING}
     * to a terminal state ({@link InflightState#COMPLETED},
     * {@link InflightState#FAILED}, {@link InflightState#TIMED_OUT})
     * exactly once — the first caller wins.
     */
    private final AtomicReference<InflightState> state = new AtomicReference<>(InflightState.RUNNING);

    /** Wall-clock timestamp (ms) when the terminal state was entered. */
    volatile long terminalTime;

    /**
     * Optional unified metric helper for terminal-state reporting.
     * Null-safe: if not set (null), no metrics are reported.
     */
    private volatile FlexlbMetricHelper metricHelper;

    /**
     * Optional callback invoked exactly once when this item enters a terminal
     * state. Used by {@link InflightStore#putIfAbsent} to maintain the
     * active-item counter.
     *
     * <p>Exactly-once guarantee: both the CAS-guarded {@link #transitionTo}
     * and the registration-compensation path ({@link #takeOnTerminal})
     * atomically claim the callback via {@code getAndSet(null)} — only one
     * of the two paths can observe a non-null reference and run it.
     */
    private final AtomicReference<Runnable> onTerminal = new AtomicReference<>();

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

    /** Returns the current lifecycle state (atomic read). */
    public InflightState state() {
        return state.get();
    }

    /** Wall-clock timestamp (ms) when this item was created. */
    public long createdAtMs() {
        return createdAtMs;
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

    /**
     * Set the terminal-transition callback. Invoked exactly once when this
     * item becomes terminal — either inside the CAS-guarded
     * {@link #transitionTo}, or via the compensation path
     * ({@link #takeOnTerminal}) if the item was already terminal when the
     * callback got registered.
     */
    public void setOnTerminal(Runnable callback) {
        this.onTerminal.set(callback);
    }

    /**
     * Atomically claim the terminal callback (getAndSet(null)). Used by the
     * {@link InflightStore#putIfAbsent} compensation path: if the item turned
     * terminal between registration and callback wiring, the caller claims
     * and runs the callback itself. Because both this path and
     * {@link #transitionTo} claim via getAndSet(null), the callback runs
     * exactly once.
     *
     * @return the callback if not yet claimed, or {@code null}
     */
    Runnable takeOnTerminal() {
        return onTerminal.getAndSet(null);
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
        if (!transitionTo(toInflightState(reason))) return false;
        if (prefillEp != null) prefillEp.release(ctx.getRequestId());
        if (decodeEp != null) decodeEp.release(ctx.getRequestId());
        if (batch != null) {
            batch.removeItem(this);
        }
        reportTerminalMetric(reason);
        future.completeExceptionally(cause != null ? cause : reason.toException());
        return true;
    }

    /**
     * Complete the request normally with the given response.
     *
     * <p>Shares the same {@code state} CAS as {@link #terminate(TerminalReason)},
     * so only one terminal transition (success or failure) can take effect.
     * On success, releases EP-level resources (Phase 2), removes the item from
     * its batch, and completes the future normally.
     */
    public void complete(Response response) {
        InflightState targetState = response.isSuccess() ? InflightState.COMPLETED : InflightState.FAILED;
        if (!transitionTo(targetState)) return;
        if (prefillEp != null) prefillEp.release(ctx.getRequestId());
        if (decodeEp != null) decodeEp.release(ctx.getRequestId());
        if (batch != null) {
            batch.removeItem(this);
        }
        TerminalReason reason = response.isSuccess() ? TerminalReason.COMPLETED : TerminalReason.FAILED;
        reportTerminalMetric(reason);
        future.complete(response);
    }

    /**
     * Atomically transition from any non-terminal state to the target terminal state.
     * Loop-CAS pattern: reads current state, returns {@code false} if already
     * terminal, otherwise attempts CAS. Preserves "first caller wins" semantics.
     */
    private boolean transitionTo(InflightState targetState) {
        while (true) {
            InflightState current = state.get();
            if (current.isTerminal()) {
                return false;
            }
            if (state.compareAndSet(current, targetState)) {
                this.terminalTime = System.currentTimeMillis();
                // Atomically claim the callback so it runs exactly once even if
                // the compensation path in InflightStore#putIfAbsent races here.
                Runnable cb = onTerminal.getAndSet(null);
                if (cb != null) {
                    cb.run();
                }
                return true;
            }
        }
    }

    /** Map a {@link TerminalReason} to the corresponding {@link InflightState}. */
    private static InflightState toInflightState(TerminalReason reason) {
        return switch (reason) {
            case COMPLETED -> InflightState.COMPLETED;
            case TIMED_OUT -> InflightState.TIMED_OUT;
            case FAILED, CANCELLED -> InflightState.FAILED;
        };
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
        return state.get().isTerminal();
    }

    public long getTerminalTime() {
        return terminalTime;
    }
}
