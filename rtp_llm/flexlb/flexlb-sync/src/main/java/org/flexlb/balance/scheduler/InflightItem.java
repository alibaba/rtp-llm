package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.sync.shadow.StateShadowBridge;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A single inference request currently inflight (dispatched to an engine,
 * awaiting ACK or response).
 *
 * <p>Core v2 design: <b>binding-as-state</b> — the presence/absence of
 * {@link #prefillEp} and {@link #decodeEp} references
 * implicitly expresses the progress phase, eliminating the need for a
 * separate state machine. A single {@link #state} AtomicReference
 * provides CAS-guarded idempotent terminal transition.
 *
 * <p>Thread-safety: the {@code state} AtomicReference ensures
 * {@link #complete(Response, InflightState)} succeeds at most once. All binding
 * fields are {@code volatile} so that other threads (e.g. the cancel path
 * via {@link InflightStore}) observe the latest endpoint references
 * when performing cleanup.
 */
public final class InflightItem {

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

    // ---- single atomic state ----

    /**
     * CAS-guarded lifecycle state. Transitions from {@link InflightState#RUNNING}
     * to a terminal state ({@link InflightState#COMPLETED},
     * {@link InflightState#FAILED}, {@link InflightState#CANCELLED},
     * {@link InflightState#TIMED_OUT}) exactly once — the first caller wins.
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

    /**
     * Package-private on purpose: the future's completion authority belongs to
     * this item's terminal methods ({@link #complete(Response, InflightState)},
     * {@link #terminate(TerminalReason, Throwable)}, {@link #fail},
     * {@link #timeout}). External code should query
     * {@link #isTerminated()} / {@link #state()} instead.
     */
    CompletableFuture<Response> future() {
        return future;
    }

    public AbstractScheduler scheduler() {
        return scheduler;
    }

    /**
     * Invoke the owning scheduler's {@link AbstractScheduler#onCancel}
     * hook if this item has a scheduler. Called from {@code RouteService.cancel}
     * after the cancel CAS wins, to release path-specific resources (e.g.
     * a queue slot). Public so that {@code RouteService} (in another package)
     * can trigger the cascade; the actual {@code onCancel} is protected and
     * accessible from here because {@code InflightItem} is in the same package.
     *
     * <p>Best-effort: the request may already have left the scheduler's
     * structures, so the hook must be idempotent.
     */
    public void fireOnCancel() {
        AbstractScheduler s = scheduler;
        if (s != null) {
            s.onCancel(this);
        }
    }

    public PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    public DecodeEndpoint decodeEp() {
        return decodeEp;
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
     * Convenience: terminate with a reason (no cause).
     *
     * @return {@code true} if this call won the CAS
     */
    public boolean terminate(TerminalReason reason) {
        return terminate(reason, null);
    }

    /**
     * Convenience terminate that preserves the original cause's message in the
     * error {@link Response}. Internally builds an error response and delegates
     * to {@link #complete(Response, InflightState)}.
     *
     * @param reason the terminal reason
     * @param cause  optional cause whose message is used in the error response
     * @return {@code true} if this call won the CAS
     */
    public boolean terminate(TerminalReason reason, Throwable cause) {
        InflightState targetState = toInflightState(reason);
        String message = cause != null ? cause.getMessage() : reason.toException().getMessage();
        Response errorResp = Response.error(toStrategyErrorType(reason), message);
        return complete(errorResp, targetState);
    }

    /**
     * Convenience: complete with a response, deriving the target state from
     * {@code response.isSuccess()} (success → COMPLETED, failure → FAILED).
     */
    public void complete(Response response) {
        boolean success = response.isSuccess();
        complete(response, success ? InflightState.COMPLETED : InflightState.FAILED);
    }

    /**
     * Unified CAS-guarded settle path: transition to the target terminal state,
     * release EP-level resources, report the terminal metric, and complete the
     * future with the given response (never {@code completeExceptionally}).
     *
     * <p>All terminal paths funnel through this method:
     * <ul>
     *   <li>success → {@code complete(successResp, COMPLETED)}</li>
     *   <li>failure → {@code complete(Response.error(FAILED, msg), FAILED)}</li>
     *   <li>timeout → {@code complete(Response.error(INFLIGHT_TTL_EXPIRED, msg), TIMED_OUT)}</li>
     *   <li>cancel  → {@code complete(Response.error(CANCELLED, msg), CANCELLED)}</li>
     * </ul>
     *
     * @return {@code true} if this call won the CAS (first terminal transition),
     *         {@code false} if the item was already terminal
     */
    public boolean complete(Response response, InflightState targetState) {
        if (!transitionTo(targetState)) return false;
        if (prefillEp != null) prefillEp.release(ctx.getRequestId());
        if (decodeEp != null) decodeEp.release(ctx.getRequestId());
        reportTerminalMetric(toTerminalReason(targetState));
        future.complete(response);
        return true;
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
            case FAILED -> InflightState.FAILED;
            case CANCELLED -> InflightState.CANCELLED;
        };
    }

    /** Map an {@link InflightState} back to the corresponding {@link TerminalReason}. */
    private static TerminalReason toTerminalReason(InflightState state) {
        return switch (state) {
            case COMPLETED -> TerminalReason.COMPLETED;
            case FAILED -> TerminalReason.FAILED;
            case CANCELLED -> TerminalReason.CANCELLED;
            case TIMED_OUT -> TerminalReason.TIMED_OUT;
            case RUNNING -> throw new IllegalArgumentException("RUNNING is not a terminal state");
        };
    }

    /** Map a {@link TerminalReason} to the corresponding {@link StrategyErrorType}. */
    private static StrategyErrorType toStrategyErrorType(TerminalReason reason) {
        return switch (reason) {
            case FAILED -> StrategyErrorType.WORKER_EXECUTION_FAILED;
            case TIMED_OUT -> StrategyErrorType.INFLIGHT_TTL_EXPIRED;
            case CANCELLED -> StrategyErrorType.CANCELLED;
            case COMPLETED -> throw new IllegalArgumentException("COMPLETED is not an error");
        };
    }

    /**
     * Convenience: fail the request with the given cause.
     *
     * @return {@code true} if this call won the CAS
     */
    public boolean fail(Throwable cause) {
        return terminate(TerminalReason.FAILED, cause);
    }

    /**
     * Convenience: time out the request.
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
     *
     * <p>G3（终态结算换权）开启时不设 helper（见 AbstractScheduler#register）——
     * metric 改由 ledger settle 出口经 {@link #terminalMetricContext()} 生产。
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
     * 终态 metric 上下文（G3 载荷）：旧四值终态原因（监控值域连续）+ 路由解析的
     * 角色与引擎地址。调用时 item 必已终态（whenComplete 回调在 complete/fail 之后）。
     */
    public StateShadowBridge.TerminalMetricContext terminalMetricContext() {
        return new StateShadowBridge.TerminalMetricContext(
                toTerminalReason(state.get()), resolveRole(), resolveEngineIp());
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
