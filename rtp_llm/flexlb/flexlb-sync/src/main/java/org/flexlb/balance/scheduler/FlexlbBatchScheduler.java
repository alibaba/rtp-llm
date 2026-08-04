package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * Coordinates batch scheduling for FlexLB disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission and routing</li>
 *   <li>Inflight lifecycle management via the global {@link InflightStore}
 *       ({@link InflightItem} registration, TTL cleanup)</li>
 *   <li>Batch assembly coordination — commits to PrefillEndpoint,
 *       delegates gRPC dispatch to {@link DefaultBatchDispatcher}</li>
 *   <li>Resource rollback on failure or completion</li>
 * </ul>
 *
 * <p>State model: each submitted request is registered atomically in the
 * {@link InflightStore} as an {@link InflightItem} (EP references null — the
 * scheduler owns EP lifecycle through the {@link BatchItem}). Duplicate
 * detection and tombstone retention are handled by the store; dispatch-level
 * tracking (assigned batch ID, dispatch timestamp, rollback flag) lives on
 * the {@link BatchItem} itself.
 *
 * <p>The actual gRPC dispatch (build protobuf, send, parse response) is
 * delegated to {@link DefaultBatchDispatcher}. Per-item results come back
 * through the {@link DefaultBatchDispatcher.DispatchCallbacks} method
 * references assembled in {@link #flushItems}; batcher decisions arrive via
 * the {@code onExpired}/{@code onBatchReady}/{@code onOfferFailure} method
 * references wired into each {@code WorkerBatcher} by the EndpointRegistry.
 */
@Component
public class FlexlbBatchScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final DefaultBatchDispatcher dispatcher;
    private final BatchSchedulerReporter reporter;
    private final InflightStore globalStore;
    private final BatchIdGenerator batchIdGenerator;

    @Autowired
    public FlexlbBatchScheduler(ConfigService configService,
                                Router router,
                                EndpointRegistry endpointRegistry,
                                DefaultBatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter,
                                InflightStore globalStore,
                                Environment environment) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.dispatcher = dispatcher;
        this.reporter = reporter;
        this.globalStore = globalStore;
        // Initialize Snowflake batch ID generator with master identity
        this.batchIdGenerator = new BatchIdGenerator(detectLocalIp(), detectPort(environment));
    }

    private static String detectLocalIp() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (UnknownHostException e) {
            Logger.warn("Failed to detect local IP, using 127.0.0.1 as fallback", e);
            return "127.0.0.1";
        }
    }

    private static int detectPort(Environment environment) {
        String portStr = environment == null ? null : environment.getProperty("server.port");
        if (portStr == null) {
            portStr = System.getProperty("server.port", "7001");
        }
        try {
            return Integer.parseInt(portStr);
        } catch (NumberFormatException e) {
            return 7001;
        }
    }

    // ==================== Request submission ====================

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            if (ctx == null || ctx.getRequest() == null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST, null);
                return future;
            }

            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && globalStore.activeCount() >= maxInflight) {
                completeError(future, StrategyErrorType.QUEUE_FULL, null);
                return future;
            }

            // Atomic registration — duplicate request IDs (active or tombstone
            // within TTL) are rejected here without a check-then-act window.
            String requestId = String.valueOf(ctx.getRequestId());
            InflightItem placeholder = new InflightItem(ctx, future, null);
            InflightItem existing = globalStore.putIfAbsent(requestId, placeholder);
            if (existing != null) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }

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

            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode),
                    prefillEp, decodeEp, System.currentTimeMillis());

            // Safety net: any non-success completion (dispatch failure, timeout,
            // TTL expiry, cancel) releases the decode reservation and repacks the
            // prefill batch. Both operations are idempotent (CAS / computeIfPresent),
            // so overlapping with the explicit callback paths is safe.
            future.whenComplete((response, throwable) -> {
                if (throwable != null || response == null || !response.isSuccess()) {
                    rollbackOnce(item);
                    removeFromPrefillBatch(item);
                }
            });

            WorkerBatcher batcher = prefillEp.getBatcher();
            ctx.setRouteSubmittedNanos(System.nanoTime());
            batcher.offer(item);

            // Report route+submit time: from schedule() entry (ctx.startTime) to batcher offer completion
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    prefillEp.getIp(),
                    System.currentTimeMillis() - ctx.getStartTime());
        } catch (Throwable t) {
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + t.getMessage());
        }
        return future;
    }

    public int getInflightSize() {
        return globalStore.activeCount();
    }

    // ==================== Inflight TTL cleanup ====================

    /**
     * TTL safety net for requests that never reached a terminal state (e.g.
     * a lost ACK). Completes the future with a TTL-expiry response, then
     * CAS-transitions the item to TIMED_OUT (tombstone). BATCH placeholders
     * (registered by this scheduler with a null scheduler reference) expire
     * with BATCH_SLO_EXPIRED; QUEUE/DIRECT items (scheduler-bound, registered
     * via plain put) expire with INFLIGHT_TTL_EXPIRED to keep batch error
     * semantics clean. Resource rollback for BATCH items runs through the
     * whenComplete hook attached in {@link #submit}. Terminal tombstones are
     * evicted by the {@link InflightStore} evictor.
     */
    @Scheduled(fixedRate = 60000L)
    public void cleanupInflight() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        long now = System.currentTimeMillis();
        globalStore.forEach((requestId, item) -> {
            if (item.state() == InflightState.RUNNING && now - item.createdAtMs() > ttlMs) {
                StrategyErrorType errorType = item.scheduler() == null
                        ? StrategyErrorType.BATCH_SLO_EXPIRED
                        : StrategyErrorType.INFLIGHT_TTL_EXPIRED;
                completeError(item.future(), errorType, "inflight TTL expired");
                if (item.timeout()) {
                    Logger.warn("FlexLB inflight TTL expired: request_id={}", requestId);
                }
            }
        });
    }

    // ==================== Batcher callbacks (wired into WorkerBatcher as method references) ====================

    public void onExpired(BatchItem head) {
        if (head.future().isDone()) {
            return;
        }
        rollbackOnce(head);
        removeFromPrefillBatch(head);
        completeError(head.future(), StrategyErrorType.BATCH_SLO_EXPIRED,
                "batch SLO expired before dispatch");
    }

    public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
        flushItems(items, meta);
    }

    public void onOfferFailure(BatchItem item, Throwable error) {
        if (item.future().isDone()) {
            return;
        }
        rollbackOnce(item);
        completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED,
                "Batcher offer failed: " + error.getMessage());
    }

    // ==================== Dispatch pipeline ====================

    /**
     * Commit batch to PrefillEndpoint, then delegate to
     * {@link DefaultBatchDispatcher} for asynchronous gRPC dispatch.
     * <p>
     * The heavy gRPC I/O is handled asynchronously by the dispatcher's own thread pool.
     */
    private void flushItems(List<BatchItem> items, DispatchMeta meta) {
        String reason = meta.reason();
        PrefillEndpoint prefillEp = items.get(0).prefillEp();

        // A timeout or prior failure may finish an item while it is still queued.
        List<BatchItem> active = items.stream()
                .filter(item -> !item.future().isDone())
                .toList();

        if (active.isEmpty()) {
            return;
        }

        // [SYNC] Assign batch ID and commit only active items to endpoint
        long predMs = 0;
        long batchId = batchIdGenerator.nextBatchId();
        List<BatchItem> dispatchable = new ArrayList<>(active.size());
        for (BatchItem item : active) {
            if (item.future().isDone()) {
                continue;
            }
            item.setAssignedBatchId(batchId);
            dispatchable.add(item);
        }

        if (dispatchable.isEmpty()) {
            return;
        }
        if (prefillEp != null) {
            PrefillTimePredictor predictor = prefillEp.getPredictor();
            predMs = (long) predictor.predictBatchMs(dispatchable);
            prefillEp.commitBatch(batchId, predMs, dispatchable);
        }

        // [ASYNC] Delegate gRPC dispatch — dispatcher owns its own thread pool
        long waitMs = System.currentTimeMillis() - items.get(0).enqueuedAtMs();
        reporter.reportBatchWaitTimeMs(
                RoleType.PREFILL.name(), prefillEp != null ? prefillEp.getIp() : "", waitMs);

        // Record dispatch timestamp for dispatch-to-ACK latency metric
        for (BatchItem item : dispatchable) {
            item.setDispatchedAtMs(System.currentTimeMillis());
            item.ctx().setBatchDispatchedNanos(System.nanoTime());
        }

        dispatcher.dispatch(dispatchable, prefillEp, batchId, predMs, reason,
                new DefaultBatchDispatcher.DispatchCallbacks(
                        this::onSuccess, this::onFailure, this::onTimeout));
    }

    // ==================== Dispatch result callbacks (passed as DispatchCallbacks) ====================

    public void onSuccess(BatchItem item, long batchId) {
        if (item.assignedBatchId() != batchId) {
            Logger.warn("Ignoring stale EnqueueBatch ACK request_id={} batch_id={}",
                    item.requestId(), batchId);
            return;
        }
        if (item.future().isDone()) {
            return;
        }

        // Record ACK timestamp for ack_to_response_time_ms metric (reported in FlexlbServiceImpl.completeSchedule)
        item.ctx().setAckAtMs(System.currentTimeMillis());
        item.ctx().setAckAtNanos(System.nanoTime());

        if (item.dispatchedAtMs() > 0) {
            PrefillEndpoint ep = item.prefillEp();
            reporter.reportDispatchAckTimeMs(
                    RoleType.PREFILL.name(),
                    ep != null ? ep.getIp() : "",
                    System.currentTimeMillis() - item.dispatchedAtMs());
        }

        completeSuccess(item);
        Logger.debug("FlexLB batch enqueued request {} in batch_id={}",
                item.requestId(), batchId);
    }

    private void completeSuccess(BatchItem item) {
        if (item.future().isDone()) {
            return;
        }
        Response success = copyResponse(item.routeResponse());
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(true);
        success.setQueueLength(globalStore.activeCount());
        item.future().complete(success);
    }

    public void onFailure(BatchItem item, Throwable error) {
        if (item.future().isDone()) {
            return;
        }
        rollbackOnce(item);
        removeFromPrefillBatch(item);
        completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED, error.getMessage());
    }

    public void onTimeout(BatchItem item, Throwable error) {
        if (item.future().isDone()) {
            return;
        }
        rollbackOnce(item);
        removeFromPrefillBatch(item);
        completeError(item.future(), StrategyErrorType.BATCH_SLO_EXPIRED,
                "EnqueueBatch deadline exceeded: " + error.getMessage());
    }

    // ==================== Internal: resource rollback ====================

    /** Release the decode reservation at most once (CAS-guarded on the item). */
    private void rollbackOnce(BatchItem item) {
        if (item.rolledBack.compareAndSet(false, true)) {
            rollback(item);
        }
    }

    /** Rollback using endpoint references already held by the item (no registry lookup). */
    private void rollback(BatchItem item) {
        DecodeEndpoint decodeEp = item.decodeEp();
        if (decodeEp != null && item.decode() != null) {
            decodeEp.release(item.decode().getRequestId());
        }
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

    /**
     * Remove a failed or timed-out request from its prefill batch entry.
     * Uses {@link PrefillEndpoint#repackBatch} which:
     * <ul>
     *   <li>Single-request batch → removes the entire entry (batch becomes empty)</li>
     *   <li>Multi-request batch → keeps survivors, removes only this request</li>
     *   <li>Batch already removed (calibrate or releaseBatch ran first) → no-op</li>
     * </ul>
     * Safe to call multiple times (idempotent via ConcurrentHashMap.computeIfPresent).
     */
    private void removeFromPrefillBatch(BatchItem item) {
        long batchId = item.assignedBatchId();
        if (batchId <= 0) {
            return;
        }
        PrefillEndpoint prefillEp = item.prefillEp();
        if (prefillEp != null) {
            prefillEp.repackBatch(batchId, Set.of(item.requestId()));
            Logger.info("FlexLB remove from prefill batch: request_id={} batch_id={} engine={}",
                    item.requestId(), batchId, prefillEp.getIp());
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

    private static Response copyResponse(Response src) {
        Response response = new Response();
        response.setServerStatus(copyServerList(src.getServerStatus()));
        response.setSuccess(src.isSuccess());
        response.setCode(src.getCode());
        response.setErrorMessage(src.getErrorMessage());
        response.setRealMasterHost(src.getRealMasterHost());
        response.setQueueLength(src.getQueueLength());
        response.setEnqueuedByMaster(src.isEnqueuedByMaster());
        return response;
    }

    private static List<ServerStatus> copyServerList(List<ServerStatus> src) {
        if (src == null) {
            return null;
        }
        List<ServerStatus> result = new ArrayList<>(src.size());
        for (ServerStatus serverStatus : src) {
            result.add(copyOf(serverStatus));
        }
        return result;
    }

    private static ServerStatus copyOf(ServerStatus src) {
        if (src == null) {
            return null;
        }
        ServerStatus status = new ServerStatus();
        status.setRole(src.getRole());
        status.setServerIp(src.getServerIp());
        status.setHttpPort(src.getHttpPort());
        status.setGrpcPort(src.getGrpcPort());
        status.setDpRank(src.getDpRank());
        status.setPrefillTime(src.getPrefillTime());
        status.setGroup(src.getGroup());
        status.setDebugInfo(copyOf(src.getDebugInfo()));
        status.setRequestId(src.getRequestId());
        status.setSuccess(src.isSuccess());
        status.setCode(src.getCode());
        status.setMessage(src.getMessage());
        return status;
    }

    private static DebugInfo copyOf(DebugInfo src) {
        if (src == null) {
            return null;
        }
        DebugInfo info = new DebugInfo();
        info.setRunningBatchSize(src.getRunningBatchSize());
        info.setQueueSize(src.getQueueSize());
        info.setWaitingTimeMs(src.getWaitingTimeMs());
        info.setAvailableKvCacheLen(src.getAvailableKvCacheLen());
        info.setEstimateTtftMs(src.getEstimateTtftMs());
        info.setEstimateTpotMs(src.getEstimateTpotMs());
        info.setHitCacheLen(src.getHitCacheLen());
        return info;
    }

    // ==================== Lifecycle ====================

    @PreDestroy
    public void shutdown() {
        endpointRegistry.close();
    }
}
