package org.flexlb.balance.scheduler;

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
            long requestId = ctx.getRequestId();
            priorityRegistry.register(requestId, ctx.getPriority());
            future.whenComplete((response, throwable) -> priorityRegistry.remove(requestId));

            Response routeResponse = router.route(ctx);
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
            future.whenComplete((response, throwable) -> {
                if (throwable != null || response == null || !response.isSuccess()) {
                    item.rollbackOnce();
                    item.removeFromPrefillBatch();
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
