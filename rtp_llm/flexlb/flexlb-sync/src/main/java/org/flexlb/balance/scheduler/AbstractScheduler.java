package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.sync.shadow.StateShadowBridge;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base class for all scheduler implementations.
 *
 * <p>Owns the shared submission registry: {@link #register} atomically
 * admits a request (duplicate request IDs rejected while the previous
 * submission is still pending) and wires the terminal transition onto the
 * result future; {@link #cancelIfPending} cancels a pending submission from
 * {@code RouteService} (completing the future with a CANCELLED error and
 * triggering the {@link #onLocalTerminal} cleanup hook).
 *
 * <p>Terminal handling is delegated by the state-ledger switch:
 * <ul>
 *   <li>ledger enabled — terminal settlement and the terminal metric go
 *       through the ledger's single authoritative exit
 *       ({@link StateShadowBridge#onOldTerminalAuthority});</li>
 *   <li>ledger disabled (degraded mode, legacy global store removed) — the
 *       terminal metric is reported directly here with the legacy
 *       four-value {@link TerminalReason} vocabulary.</li>
 * </ul>
 */
public abstract class AbstractScheduler implements DiagnosticsProvider {

    protected final FlexlbMetricHelper metricHelper;

    /** 状态账本门面：开关关时为 {@link StateShadowBridge#DISABLED} no-op 单例。 */
    protected final StateShadowBridge shadowBridge;

    /** 提交中请求登记表（requestId → 终态前的上下文/结果 future/端点引用快照）。 */
    private final ConcurrentHashMap<Long, PendingSubmission> pendingSubmissions = new ConcurrentHashMap<>();

    protected AbstractScheduler(FlexlbMetricHelper metricHelper) {
        this(metricHelper, StateShadowBridge.DISABLED);
    }

    protected AbstractScheduler(FlexlbMetricHelper metricHelper, StateShadowBridge shadowBridge) {
        this.metricHelper = metricHelper;
        this.shadowBridge = shadowBridge == null ? StateShadowBridge.DISABLED : shadowBridge;
    }

    /**
     * Submit a request for scheduling and dispatch.
     *
     * @param ctx the request context carrying the {@link BalanceContext}
     * @return a future that will be completed with the routing {@link Response}
     */
    public abstract CompletableFuture<Response> submit(BalanceContext ctx);

    /**
     * Atomically register the submission and wire the terminal transition
     * onto the given result future.
     *
     * <p>Registration is keyed by request ID in the per-scheduler pending
     * table ({@link ConcurrentHashMap#putIfAbsent}) — a duplicate submission
     * (previous submission with the same ID still pending) is rejected
     * without a check-then-act window. The entry is removed when the future
     * reaches any terminal completion.
     *
     * <p>On terminal completion the state ledger is notified through the
     * bridge: settlement-authority mode funnels into the ledger's single
     * terminal exit (metric produced exactly once per request there);
     * ledger-disabled mode falls back to direct legacy metric reporting.
     * Resource release at the endpoint level is owned by the dispatch path
     * (e.g. the BatchItem safety net) — both are idempotent.
     *
     * @param ctx    the request context
     * @param future the result future whose completion drives the terminal state
     * @return {@code true} if the submission was newly registered,
     *         {@code false} when the request ID duplicates a pending submission
     */
    protected boolean register(BalanceContext ctx, CompletableFuture<Response> future) {
        long requestId = ctx.getRequestId();
        PendingSubmission fresh = new PendingSubmission(ctx, future);
        if (pendingSubmissions.putIfAbsent(requestId, fresh) != null) {
            return false;
        }
        future.whenComplete((response, throwable) -> {
            pendingSubmissions.remove(requestId);
            String terminalState = oldTerminalState(response, throwable);
            if (shadowBridge.isSettleAuthority()) {
                // 结算换权：ledger settle 单出口（COMPLETED 挂起等引擎终局；
                // FAILED/TIMED_OUT/CANCELLED 双侧主动 settle）。metric 由 settle
                // 出口生产——每请求恰好一次。客户端 future 已完成，行为不变。
                shadowBridge.onOldTerminalAuthority(requestId, terminalState,
                        fresh.terminalMetricContext(reasonOf(terminalState)));
            } else if (shadowBridge.isEnabled()) {
                // 影子对账（未换权的观察模式）：旧路径终态记录进 diff 窗口。
                shadowBridge.onOldTerminal(requestId, terminalState);
            } else if (metricHelper != null) {
                // 账本关（退化模式）：终态 metric 直报，保持旧四值口径连续。
                metricHelper.reportTerminal(reasonOf(terminalState),
                        fresh.role(), fresh.engineIp(), null);
            }
        });
        return true;
    }

    /**
     * Resolve the terminal state name from a future completion, mirroring
     * the legacy inflight lifecycle vocabulary: success → COMPLETED,
     * CANCELLED error code → CANCELLED, inflight-TTL error code → TIMED_OUT,
     * everything else (exceptional completion included) → FAILED.
     */
    static String oldTerminalState(Response response, Throwable throwable) {
        if (throwable != null || response == null) {
            return "FAILED";
        }
        if (response.isSuccess()) {
            return "COMPLETED";
        }
        if (response.getCode() == StrategyErrorType.CANCELLED.getErrorCode()) {
            return "CANCELLED";
        }
        if (response.getCode() == StrategyErrorType.INFLIGHT_TTL_EXPIRED.getErrorCode()) {
            return "TIMED_OUT";
        }
        return "FAILED";
    }

    /** 终态名 → 旧四值终态原因（监控值域与旧口径连续）。 */
    static TerminalReason reasonOf(String terminalState) {
        return switch (terminalState) {
            case "COMPLETED" -> TerminalReason.COMPLETED;
            case "CANCELLED" -> TerminalReason.CANCELLED;
            case "TIMED_OUT" -> TerminalReason.TIMED_OUT;
            default -> TerminalReason.FAILED;
        };
    }

    /**
     * Cancel a still-pending submission (local cancel): unregister it and
     * complete its future with a CANCELLED error response — first completion
     * wins, so a submission that already reached any terminal state returns
     * {@code false}. On success the owning scheduler's
     * {@link #onLocalTerminal} hook releases path-specific resources (e.g.
     * a queue slot).
     */
    public boolean cancelIfPending(long requestId) {
        PendingSubmission pending = pendingSubmissions.remove(requestId);
        if (pending == null) {
            return false;
        }
        boolean completed = pending.future().complete(
                Response.error(StrategyErrorType.CANCELLED, "cancelled"));
        if (completed) {
            onLocalTerminal(pending.ctx());
        }
        return completed;
    }

    /**
     * Backfill endpoint references onto the pending submission after a
     * successful route. Serves only the terminal metric's role / engineIp
     * resolution (monitoring vocabulary continuity); resource release is
     * owned by the dispatch path (BatchItem safety net) — both idempotent.
     */
    protected void bindPendingEndpoints(long requestId, PrefillEndpoint prefillEp, DecodeEndpoint decodeEp) {
        PendingSubmission pending = pendingSubmissions.get(requestId);
        if (pending != null) {
            pending.bind(prefillEp, decodeEp);
        }
    }

    /** 登记表当前大小（诊断/监控汇总用）。 */
    public int pendingCount() {
        return pendingSubmissions.size();
    }

    /**
     * Hook invoked after a local cancel won the future completion, letting
     * the owning scheduler release path-specific resources (e.g. a queue
     * slot) with best-effort semantics — the request may already have left
     * the scheduler's structures.
     *
     * <p>Default implementation is a no-op.
     */
    protected void onLocalTerminal(BalanceContext ctx) {
        // default: no path-specific terminal cleanup
    }

    /**
     * Report path-specific metrics for this scheduler.
     *
     * <p>Default implementation is a no-op. Subclasses override this to
     * report their own metrics (e.g. inflight size, queue length).
     * Called periodically by {@code RouteService.triggerSchedulerMetrics()}.
     */
    public void reportMetrics() {
        // default: no scheduler-specific metrics
    }

    /**
     * Start any background resources owned by this scheduler (e.g. worker
     * pool, queue consumer). Called once at startup by
     * {@code RouteService.start()}.
     *
     * <p>Default implementation is a no-op. Subclasses with background
     * resources override this (e.g. {@link QueueScheduler} starts its
     * {@code QueueingComponent} worker pool).
     */
    public void start() {
        // default: no background resources to start
    }

    /**
     * Shut down any background resources owned by this scheduler.
     * Called once at shutdown by {@code RouteService.shutdown()}.
     *
     * <p>Default implementation is a no-op. Subclasses with background
     * resources override this (e.g. {@link QueueScheduler} shuts down its
     * {@code QueueingComponent} worker pool).
     */
    public void shutdown() {
        // default: no background resources to shut down
    }

    /**
     * {@inheritDoc}
     *
     * <p>Default implementation returns an empty map. Subclasses with
     * diagnostics to report (e.g. queue length) override this.
     */
    @Override
    public Map<String, Object> getDiagnostics() {
        return Map.of();
    }

    /**
     * A submission registered in the pending table: holds the context and
     * result future from admission until terminal completion. Endpoint
     * references are backfilled after a successful route (terminal metric's
     * role / engineIp resolution — mirroring the legacy inflight binding
     * semantics for monitoring only; resource release stays with the
     * dispatch path).
     */
    static final class PendingSubmission {

        private final BalanceContext ctx;
        private final CompletableFuture<Response> future;

        private volatile PrefillEndpoint prefillEp;
        private volatile DecodeEndpoint decodeEp;

        PendingSubmission(BalanceContext ctx, CompletableFuture<Response> future) {
            this.ctx = ctx;
            this.future = future;
        }

        BalanceContext ctx() {
            return ctx;
        }

        CompletableFuture<Response> future() {
            return future;
        }

        void bind(PrefillEndpoint prefillEp, DecodeEndpoint decodeEp) {
            if (prefillEp != null) {
                this.prefillEp = prefillEp;
            }
            if (decodeEp != null) {
                this.decodeEp = decodeEp;
            }
        }

        /** Resolve the engine role from the available endpoint references. */
        String role() {
            if (prefillEp != null) {
                return "PREFILL";
            }
            if (decodeEp != null) {
                return "DECODE";
            }
            return "UNKNOWN";
        }

        /** Resolve the engine IP from the available endpoint references. */
        String engineIp() {
            if (prefillEp != null) {
                return prefillEp.getIp();
            }
            if (decodeEp != null) {
                return decodeEp.getIp();
            }
            return "unknown";
        }

        /** 终态 metric 上下文（结算换权载荷：旧四值终态原因 + 路由解析的角色与引擎地址）。 */
        StateShadowBridge.TerminalMetricContext terminalMetricContext(TerminalReason reason) {
            return new StateShadowBridge.TerminalMetricContext(reason, role(), engineIp());
        }
    }
}
