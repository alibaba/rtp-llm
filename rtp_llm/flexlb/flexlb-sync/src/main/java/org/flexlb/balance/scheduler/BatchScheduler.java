package org.flexlb.balance.scheduler;

import org.flexlb.autotpm.PreemptResult;
import org.flexlb.autotpm.PriorityPressureController;
import org.flexlb.autotpm.PriorityRegistry;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.util.Logger;

import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/**
 * Scheduler for BATCH mode — the request admission front door for FlexLB
 * disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission: capacity gate and atomic duplicate detection via
 *       the global {@link InflightStore} (base-class {@code register})</li>
 *   <li>Routing: pick prefill/decode workers through the {@link Router}</li>
 *   <li>Hand-off: build a {@link BatchItem} and offer it to the target
 *       {@link PrefillEndpoint}'s {@link WorkerBatcher}</li>
 * </ul>
 *
 * <p>Everything past the hand-off lives on the endpoint side: batching
 * ({@link WorkerBatcher}), commit + gRPC dispatch
 * ({@link PrefillEndpoint#submitBatch}), and per-item settlement
 * ({@link BatchItem} terminal transitions). The scheduler keeps only a
 * {@code whenComplete} safety net that releases EP resources on any
 * non-success completion (TTL expiry, cancel); both operations are
 * idempotent, so overlapping with the endpoint-side paths is safe.
 *
 * <p>TTL expiry of a batch item follows the unified inflight semantics:
 * {@link StrategyErrorType#INFLIGHT_TTL_EXPIRED} (see
 * {@link InflightItem#timeoutWithError()}); batch dispatch timeouts keep
 * {@code BATCH_SLO_EXPIRED} inside {@link BatchItem}.
 */
public class BatchScheduler extends AbstractScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchSchedulerReporter reporter;

    /**
     * Auto-TPM requestId → priority registry. The engine does not report
     * priority, so victim selection reads this master-side record. Entries
     * live from InflightStore registration to future settlement.
     */
    private final PriorityRegistry priorityRegistry = new PriorityRegistry();

    /**
     * Auto-TPM running-decode preemption orchestrator (D6). Optional wiring
     * via {@link #setPressureController}: null (default) means the preempt
     * branch is a no-op — behavior is identical to the pre-Stage-3 scheduler.
     */
    private volatile PriorityPressureController pressureController;

    public BatchScheduler(ConfigService configService,
                          Router router,
                          EndpointRegistry endpointRegistry,
                          BatchSchedulerReporter reporter,
                          InflightStore globalStore,
                          FlexlbMetricHelper metricHelper) {
        super(globalStore, metricHelper);
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.reporter = reporter;
    }

    // ==================== Request submission ====================

    @Override
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            if (ctx == null || ctx.getRequest() == null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST, null);
                return future;
            }

            // BATCH-only admission gate: count only items this scheduler
            // registered — DIRECT/QUEUE traffic must not consume the budget.
            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && globalStore.activeCount(this) >= maxInflight) {
                completeError(future, StrategyErrorType.QUEUE_FULL, null);
                return future;
            }

            // Auto-TPM D11 (task40, scope widened per owner sign-off): with the
            // switch on, ANY request — including the 0 sentinel (D12) — whose
            // SLO deadline (startTime + resolveSloMs(seqLen)) has already passed
            // is rejected right at the admission gate (8400) — remaining
            // budget ≤ 0 is not schedulable work. Switch off keeps the pre-D11
            // parity. The sentinel still skips every other Auto-TPM mechanism
            // (registry, yield, preemption, priority metrics).
            var d11Cfg = configService.loadBalanceConfig();
            if (d11Cfg.isAutoTpmEnabled()) {
                long sloMs = d11Cfg.resolveSloMs(
                        ctx.getRequest() != null ? ctx.getRequest().getSeqLen() : 0);
                if (sloMs > 0) {
                    long deadlineMs = ctx.getStartTime() + sloMs;
                    long nowMs = System.currentTimeMillis();
                    if (deadlineMs <= nowMs) {
                        Logger.info("FlexLB auto-tpm slo deadline exceeded, reject "
                                        + "request_id={} priority={} deadline_ms={} now_ms={}",
                                ctx.getRequestId(), ctx.getPriority(), deadlineMs, nowMs);
                        completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER,
                                "slo deadline exceeded: deadline_ms=" + deadlineMs
                                        + " now_ms=" + nowMs);
                        return future;
                    }
                }
            }

            // Atomic registration — duplicate request IDs (active or tombstone
            // within TTL) are rejected here without a check-then-act window.
            InflightItem existing = register(ctx, future);
            if (existing != null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }

            // Auto-TPM priority bookkeeping: register right after winning the
            // InflightStore registration; every terminal path settles the
            // future (exactly-once), so whenComplete reliably drops the entry.
            // D12: the 0 sentinel (no priority carried) is never registered —
            // such requests must never enter the victim candidate snapshot.
            long requestId = ctx.getRequestId();
            if (ctx.getPriority() > 0) {
                priorityRegistry.register(requestId, ctx.getPriority());
                future.whenComplete((response, throwable) -> priorityRegistry.remove(requestId));
            }

            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                // Auto-TPM Phase 2 (D6): no capacity for this priority —
                // try preempting one strictly lower-priority RUNNING decode
                // request, then re-route once onto the confirmed-freed
                // capacity (reusing the full reserve+BatchItem+offer path).
                routeResponse = tryPreemptAndReroute(ctx, routeResponse);
            }
            if (routeResponse == null || !routeResponse.isSuccess()) {
                if (routeResponse != null) {
                    future.complete(routeResponse);
                } else {
                    completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER, null);
                }
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollback(routeResponse);
                completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                return future;
            }

            String prefillIpPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(prefillIpPort);
            if (prefillEp == null) {
                rollback(routeResponse);
                completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                return future;
            }

            DecodeEndpoint decodeEp = null;
            if (decode != null) {
                String decodeIpPort = decode.getServerIp() + ":" + decode.getHttpPort();
                decodeEp = endpointRegistry.getDecode(decodeIpPort);
            }

            // Backfill EP references on the InflightItem so that TTL/cancel
            // terminal transitions can directly release EP-level resources
            // (review A1). Without this, InflightItem.terminate() always sees
            // null prefillEp/decodeEp and EP release relies solely on the
            // whenComplete safety net below.
            //
            // decodeEp.release(requestId) works immediately (always keyed by
            // requestId). prefillEp.release(requestId) is a no-op for batch
            // entries (keyed by batchId) but works for non-batch entries; the
            // whenComplete safety net covers the batch prefill case via
            // BatchItem.removeFromPrefillBatch(). Both paths are idempotent.
            InflightItem inflightItem = globalStore.get(String.valueOf(ctx.getRequestId()));
            if (inflightItem != null) {
                inflightItem.setPrefillEp(prefillEp);
                if (decodeEp != null) {
                    inflightItem.setDecodeEp(decodeEp);
                }
            }

            BatchItem item = new BatchItem(ctx, future, routeResponse,
                    BatchItem.copyOf(prefill), BatchItem.copyOf(decode),
                    prefillEp, decodeEp, System.currentTimeMillis());

            // Safety net: any non-success completion (dispatch failure, timeout,
            // TTL expiry, cancel) releases the decode reservation and repacks the
            // prefill batch. Both operations are idempotent (CAS / computeIfPresent),
            // so overlapping with the endpoint-side terminal paths is safe.
            //
            // D10 metrics piggyback on the same settlement hook (gated on
            // AUTO_TPM_ENABLED for off-state parity; the helper centrally
            // drops the 0 sentinel): success emits the scheduler-side TTFT
            // approximation (submit arrival → engine enqueue ACK), failure
            // emits deadline_miss.count when the item was cleared on a
            // queue-deadline path.
            boolean autoTpmMetricsOn = configService.loadBalanceConfig().isAutoTpmEnabled();
            future.whenComplete((response, throwable) -> {
                if (throwable != null || response == null || !response.isSuccess()) {
                    item.rollbackOnce();
                    item.removeFromPrefillBatch();
                    if (autoTpmMetricsOn && item.deadlineMissed()) {
                        metricHelper.reportAutoTpmDeadlineMiss(item.priority());
                    }
                } else if (autoTpmMetricsOn) {
                    metricHelper.reportAutoTpmTtft(item.priority(),
                            System.currentTimeMillis() - ctx.getStartTime());
                }
            });

            WorkerBatcher batcher = prefillEp.getBatcher();
            ctx.setRouteSubmittedNanos(System.nanoTime());
            Logger.debug("FlexLB batch_submit request_id={} priority={} worker={}",
                    ctx.getRequestId(), ctx.getPriority(), prefillEp.getIp());
            batcher.offer(item);

            // Report route+submit time: from schedule() entry (ctx.startTime) to batcher offer completion
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    prefillEp.getIp(),
                    System.currentTimeMillis() - ctx.getStartTime());
        } catch (Throwable t) {
            Logger.error("BatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + t.getMessage());
        }
        return future;
    }

    // ==================== Internal: resource rollback (pre-BatchItem paths) ====================

    /**
     * Auto-TPM Phase 2 preempt branch (D6). Invoked only when routing failed;
     * returns the original failed response unless a preemption is confirmed
     * and a single re-route succeeds.
     *
     * <p>Preconditions for even attempting: a controller is wired, and the
     * failure is a capacity-exhaustion outcome (null response,
     * NO_AVAILABLE_WORKER or NO_DECODE_WORKER) — other route errors (invalid
     * request, no prefill worker, ...) are not preemption-solvable.
     *
     * <p>On confirmed preemption the freed capacity is claimed by re-running
     * the router once, reusing the full reserve+BatchItem+offer path. The
     * controller only returns after the victim's release has been verified,
     * so the freed slot is visible to the router (never optimistic; the
     * re-route may still pick another endpoint that freed up meanwhile,
     * which is equally correct).
     */
    private Response tryPreemptAndReroute(BalanceContext ctx, Response failed) {
        PriorityPressureController controller = pressureController;
        if (controller == null) {
            return failed;
        }
        if (failed != null
                && failed.getCode() != StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode()
                && failed.getCode() != StrategyErrorType.NO_DECODE_WORKER.getErrorCode()) {
            return failed;
        }
        Optional<PreemptResult> preempted = controller.tryPreempt(ctx);
        if (preempted.isEmpty()) {
            return failed;
        }
        PreemptResult result = preempted.get();
        Logger.info("FlexLB auto-tpm preempt_reroute request_id={} priority={} "
                        + "victim_request_id={} victim_priority={} freed_endpoint={}",
                ctx.getRequestId(), ctx.getPriority(),
                result.victimRequestId(), result.victimPriority(), result.endpoint());
        Response rerouted = router.route(ctx);
        return rerouted != null ? rerouted : failed;
    }

    /**
     * Rollback using route response — used only in submit() early-return paths
     * where BatchItem has not been created yet.
     */
    private void rollback(Response routeResponse) {
        if (routeResponse == null || routeResponse.getServerStatus() == null) {
            return;
        }
        for (ServerStatus serverStatus : routeResponse.getServerStatus()) {
            rollback(serverStatus);
        }
    }

    private void rollback(ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        if (serverStatus.getRole() == RoleType.DECODE) {
            String ipPort = serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
            DecodeEndpoint ep = endpointRegistry.getDecode(ipPort);
            if (ep != null) {
                ep.release(serverStatus.getRequestId());
            }
        }
    }

    private static void completeError(CompletableFuture<Response> future,
                                      StrategyErrorType errorType,
                                      String message) {
        if (future.isDone()) {
            return;
        }
        Response errorResp = Response.error(errorType);
        errorResp.setErrorMessage(message == null ? errorType.getErrorMsg() : message);
        future.complete(errorResp);
    }

    // ==================== Internal: static utilities ====================

    private static ServerStatus findServer(Response response, RoleType roleType) {
        if (response.getServerStatus() == null) {
            return null;
        }
        for (ServerStatus serverStatus : response.getServerStatus()) {
            if (serverStatus != null && roleType == serverStatus.getRole()) {
                return serverStatus;
            }
        }
        return null;
    }

    // ==================== Metrics ====================

    /** Auto-TPM priority registry accessor (preemption orchestration wiring). */
    public PriorityRegistry priorityRegistry() {
        return priorityRegistry;
    }

    /** Wire the Auto-TPM preemption orchestrator; null keeps the branch a no-op. */
    public void setPressureController(PriorityPressureController controller) {
        this.pressureController = controller;
    }

    /**
     * Report BATCH-specific metrics: the number of active (non-terminal)
     * requests registered by this scheduler in the global inflight store —
     * the same count gating admission via {@code flexlbBatchMaxInflight}.
     */
    @Override
    public void reportMetrics() {
        metricHelper.reportInflightSize("PREFILL", "scheduler", globalStore.activeCount(this));
    }
}
