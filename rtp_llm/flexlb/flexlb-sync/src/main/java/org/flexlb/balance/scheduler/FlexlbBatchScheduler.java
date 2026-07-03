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
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Lazy;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Coordinates batch scheduling for FlexLB disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission and routing (submit, cancel)</li>
 *   <li>Inflight lifecycle management (inflight map, TTL cleanup)</li>
 *   <li>Batch assembly coordination — commits to PrefillEndpoint,
 *       filters cancelled items, delegates gRPC dispatch to {@link BatchDispatcher}</li>
 *   <li>Resource rollback on failure or completion</li>
 * </ul>
 *
 * <p>The actual gRPC dispatch (build protobuf, send, parse response) is
 * delegated to {@link BatchDispatcher}. Per-item results come back through
 * {@link DispatchCallback} which this class implements.
 */
@Component
public class FlexlbBatchScheduler implements BatchDecisionHandler, DispatchCallback {

    public final ConfigService configService;
    private final Router router;
    final EngineGrpcClient grpcClient;
    final EngineWorkerStatus engineWorkerStatus;
    final EndpointRegistry endpointRegistry;
    final BatchDispatcher dispatcher;
    final BatchSchedulerReporter reporter;
    final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    final AtomicLong batchIdGenerator = new AtomicLong(0);
    private final InflightEvictor<Long, InflightEntry> inflightEvictor
            = new InflightEvictor<>(inflight, entry -> {
                synchronized (entry) {
                    rollbackOnce(entry);
                    repackPrefillBatch(entry);
                    completeCancelled(entry);
                }
            });

    public FlexlbBatchScheduler(ConfigService configService,
                                @Lazy Router router,
                                EngineGrpcClient grpcClient,
                                EngineWorkerStatus engineWorkerStatus,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter) {
        this.configService = configService;
        this.router = router;
        this.grpcClient = grpcClient;
        this.engineWorkerStatus = engineWorkerStatus;
        this.endpointRegistry = endpointRegistry;
        this.dispatcher = dispatcher;
        this.reporter = reporter;
    }

    // ==================== Request submission ====================

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            if (ctx == null || ctx.getRequest() == null) {
                future.complete(Response.error(StrategyErrorType.INVALID_REQUEST));
                return future;
            }

            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && inflight.size() >= maxInflight) {
                Response resp = Response.error(StrategyErrorType.QUEUE_FULL);
                future.complete(resp);
                return future;
            }

            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                future.complete(routeResponse != null
                        ? routeResponse
                        : Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollback(routeResponse);
                Response resp = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
                future.complete(resp);
                return future;
            }

            String prefillIpPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(prefillIpPort);
            if (prefillEp == null) {
                rollback(routeResponse);
                Response resp = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
                future.complete(resp);
                return future;
            }

            DecodeEndpoint decodeEp = null;
            if (decode != null) {
                String decodeIpPort = decode.getServerIp() + ":" + decode.getHttpPort();
                decodeEp = endpointRegistry.getDecode(decodeIpPort);
            }

            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode),
                    prefillEp, decodeEp, /* sortKey set by batcher */ 0, System.currentTimeMillis());
            inflight.put(ctx.getRequestId(), new InflightEntry(item));
            WorkerBatcher batcher = prefillEp.getBatcher();
            batcher.offer(item);
        } catch (Throwable t) {
            if (ctx != null) {
                inflight.remove(ctx.getRequestId());
            }
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode());
            errorResp.setErrorMessage("Submit failed: " + t.getMessage());
            future.complete(errorResp);
        }
        return future;
    }

    // ==================== Cancellation ====================

    public void cancel(long requestId) {
        InflightEntry entry = inflight.remove(requestId);
        if (entry == null) {
            Logger.debug("flexlb batch cancel ignored; request {} not found in inflight", requestId);
            return;
        }

        synchronized (entry) {
            entry.cancelled.set(true);
            if (!entry.ackFinished) {
                completeCancelled(entry);
            } else if (!entry.item.future().isDone()) {
                rollbackOnce(entry);
            }
            // If ackFinished and future already done (success), skip rollback
        }
        cancelPrefill(entry);
        // Remove this request from PrefillEndpoint.inflightBatches to prevent
        // inflight leak. Without this, the batch entry stays until calibrate
        // (if engine reports the cancel) or TTL (300s) — inflating the count
        // and causing false backpressure when FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES is enabled.
        repackPrefillBatch(entry);
    }

    // ==================== Completion from worker status ====================

    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse response) {
        if (response == null) {
            return;
        }
        Map<String, TaskInfo> finishedTaskInfo = response.getFinishedTaskInfo();
        if (finishedTaskInfo == null || finishedTaskInfo.isEmpty()) {
            return;
        }

        boolean isPrefill = response.getRole() == RoleType.PREFILL;

        for (TaskInfo task : finishedTaskInfo.values()) {
            long requestId = task.getRequestId();

            // Prefill success: decode is still running, keep scheduler inflight entry
            if (isPrefill && task.getErrorCode() == 0) {
                continue;
            }

            // Remove from scheduler inflight (prefill error, or any decode completion)
            InflightEntry entry = inflight.remove(requestId);

            // Prefill error: rollback decode KV reservation since decode will never run
            if (isPrefill && entry != null) {
                rollbackOnce(entry);
            }
            // Decode completion (success or error): scheduler only cleans its own map.
            // DecodeEndpoint.calibrate() independently handles its own inflightRequests cleanup.
        }
    }

    public void removeInflight(long requestId) {
        inflight.remove(requestId);
    }

    // ==================== Inflight TTL cleanup ====================

    @Scheduled(fixedRate = 60000L)
    public void cleanupInflight() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        inflightEvictor.evictExpired(ttlMs);
    }

    // ==================== BatchDecisionHandler callbacks (from WorkerBatcher) ====================

    @Override
    public void onExpired(BatchItem head) {
        removeInflight(head.requestId());
        rollback(head);
        if (!head.future().isDone()) {
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode());
            errorResp.setErrorMessage("FlexLB request deadline expired — cannot meet TTFT SLO");
            head.future().complete(errorResp);
        }
    }

    @Override
    public void onUrgent(BatchItem head, DispatchMeta meta) {
        flushItems(List.of(head), meta.reason());
    }

    @Override
    public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
        flushItems(items, meta.reason());
    }

    @Override
    public void onOfferFailure(BatchItem item, Throwable error) {
        removeInflight(item.requestId());
        rollback(item);
        if (!item.future().isDone()) {
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode());
            errorResp.setErrorMessage("Batcher offer failed: " + error.getMessage());
            item.future().complete(errorResp);
        }
    }

    // ==================== Dispatch pipeline ====================

    /**
     * Commit batch to PrefillEndpoint, filter cancelled items, then delegate
     * to {@link BatchDispatcher} for asynchronous gRPC dispatch.
     * <p>
     * Filtering is done synchronously — it only reads inflight (ConcurrentHashMap)
     * and performs fast in-memory operations. The heavy gRPC I/O is handled
     * asynchronously by the dispatcher's own thread pool.
     */
    private void flushItems(List<BatchItem> items, String reason) {
        PrefillEndpoint prefillEp = items.get(0).prefillEp();

        // [SYNC] Filter cancelled/done items first — avoid committing them to the endpoint
        List<BatchItem> active = items.stream()
                .filter(item -> !isCancelled(item) && !item.future().isDone())
                .toList();

        // Complete items that were cancelled before dispatch
        for (BatchItem item : items) {
            if (!active.contains(item)) {
                completeCancelled(item);
            }
        }

        if (active.isEmpty()) {
            return;
        }

        // [SYNC] Compute prediction and commit only active items to endpoint
        long predMs = 0;
        long batchId = batchIdGenerator.incrementAndGet();
        if (prefillEp != null) {
            PrefillTimePredictor predictor = prefillEp.getPredictor();
            predMs = predictor.predictBatchMs(active);
            prefillEp.commitBatch(batchId, predMs, active);
        }

        // Store batchId in inflight entries so cancel() can repack the batch.
        // Must be set AFTER commitBatch (so repackBatch finds the entry) and
        // BEFORE dispatch (so cancel during gRPC can also repack).
        // Edge case: cancel in the tiny window between commitBatch and this
        // loop sees batchId=-1, skips repack; calibrate cleans up (~10ms).
        for (BatchItem item : active) {
            InflightEntry entry = inflight.get(item.requestId());
            if (entry != null) {
                entry.batchId = batchId;
            }
        }

        // [ASYNC] Delegate gRPC dispatch — dispatcher owns its own thread pool
        long waitMs = System.currentTimeMillis() - items.get(0).enqueuedAtMs();
        reporter.reportBatchWaitTimeMs(RoleType.PREFILL.name(), prefillEp != null ? prefillEp.getIp() : "", waitMs);
        dispatcher.dispatch(active, prefillEp, batchId, predMs, reason, this);
    }

    // ==================== DispatchCallback implementation ====================

    @Override
    public void onSuccess(BatchItem item, long batchId) {
        InflightEntry entry = inflight.get(item.requestId());
        if (entry == null) {
            // cancel() already removed entry and handled cleanup
            return;
        }

        boolean cancelAfterAck = false;
        synchronized (entry) {
            entry.ackFinished = true;
            if (entry.cancelled.get()) {
                cancelAfterAck = true;
            } else if (!item.future().isDone()) {
                Response success = copyResponse(item.routeResponse());
                success.setSuccess(true);
                success.setCode(200);
                success.setEnqueuedByMaster(true);
                success.setQueueLength(inflight.size());
                item.future().complete(success);
                Logger.debug("FlexLB batch enqueued request {} in batch {}", item.requestId(), batchId);
            }
        }

        if (cancelAfterAck) {
            inflight.remove(item.requestId());
            cancelPrefill(entry);
            repackPrefillBatch(entry);
            completeCancelled(entry);
        }
    }

    @Override
    public void onFailure(BatchItem item, Throwable error) {
        failAck(item, error);
    }

    // ==================== Internal: inflight state management ====================

    void failAck(BatchItem item, Throwable error) {
        InflightEntry entry = inflight.remove(item.requestId());
        if (entry != null) {
            synchronized (entry) {
                entry.ackFinished = true;
                rollbackOnce(entry);
                if (!item.future().isDone()) {
                    Response errorResp = new Response();
                    errorResp.setSuccess(false);
                    errorResp.setCode(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode());
                    errorResp.setErrorMessage(error.getMessage());
                    item.future().complete(errorResp);
                }
            }
            return;
        }
        rollback(item);
        if (!item.future().isDone()) {
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode());
            errorResp.setErrorMessage(error.getMessage());
            item.future().complete(errorResp);
        }
    }

    private void completeCancelled(BatchItem item) {
        InflightEntry entry = inflight.remove(item.requestId());
        if (entry != null) {
            synchronized (entry) {
                completeCancelled(entry);
            }
            return;
        }
        item.ctx().cancel();
        rollback(item);
        if (!item.future().isDone()) {
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.REQUEST_CANCELLED.getErrorCode());
            errorResp.setErrorMessage("Request cancelled by client");
            item.future().complete(errorResp);
        }
    }

    private void completeCancelled(InflightEntry entry) {
        entry.item.ctx().cancel();
        rollbackOnce(entry);
        if (!entry.item.future().isDone()) {
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.REQUEST_CANCELLED.getErrorCode());
            errorResp.setErrorMessage("Request cancelled by client");
            entry.item.future().complete(errorResp);
        }
    }

    // ==================== Internal: resource rollback ====================

    private void rollbackOnce(InflightEntry entry) {
        if (entry.rolledBack.compareAndSet(false, true)) {
            rollback(entry.item);
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

    // ==================== Internal: inflight queries ====================

    private boolean isCancelled(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return item.ctx().isCancelled() || (entry != null && entry.cancelled.get());
    }

    // ==================== Internal: engine cancel ====================

    /**
     * Cancel request on the prefill engine via gRPC.
     * <p>
     * Only prefill needs an explicit cancel — there is no symmetric {@code cancelDecode()}.
     * The prefill engine owns the full request lifecycle in PD-separated architecture:
     * {@code PrefillRpcServer::Cancel()} cancels the response entry, which cascades
     * internally to interrupt the prefill→decode flow.
     */
    private void cancelPrefill(InflightEntry entry) {
        PrefillEndpoint prefillEp = entry.item.prefillEp();
        if (prefillEp == null) {
            return;
        }
        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            grpcClient.cancel(prefillEp.getIp(),
                    prefillEp.getGrpcPort(),
                    entry.item.requestId(),
                    deadlineMs);
        } catch (RuntimeException e) {
            Logger.warn("FlexLB batch cancel failed for request {}", entry.item.requestId(), e);
        }
    }

    /**
     * Remove a cancelled request from its prefill batch entry to prevent
     * inflight leak.  Uses {@link PrefillEndpoint#repackBatch} which:
     * <ul>
     *   <li>Single-request batch → removes the entire entry (batch becomes empty)</li>
     *   <li>Multi-request batch → keeps survivors, removes only this request</li>
     *   <li>Batch already removed (calibrate or releaseBatch ran first) → no-op</li>
     * </ul>
     * Safe to call multiple times (idempotent via ConcurrentHashMap.computeIfPresent).
     */
    private void repackPrefillBatch(InflightEntry entry) {
        if (entry.batchId < 0) {
            return;
        }
        PrefillEndpoint prefillEp = entry.item.prefillEp();
        if (prefillEp != null) {
            prefillEp.repackBatch(entry.batchId, Set.of(entry.item.requestId()));
            Logger.info("FlexLB cancel repack: request_id={} batch_id={} engine={}",
                    entry.item.requestId(), entry.batchId, prefillEp.getIp());
        }
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

    public BatchSchedulerReporter getReporter() {
        return reporter;
    }

    @Scheduled(fixedRate = 2000L)
    public void reportBatchMetrics() {
        reporter.reportSchedulerInflightSize(inflight.size());

        // Per-worker metrics: prefill endpoints
        for (Map.Entry<String, PrefillEndpoint> entry : endpointRegistry.getPrefillEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }

        // Per-worker metrics: decode endpoints
        for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }
    }

    @PreDestroy
    public void shutdown() {
        endpointRegistry.close();
    }

    // ==================== Inflight entry ====================

    static final class InflightEntry implements InflightEvictor.TtlTracked {
        final BatchItem item;
        private final long createdAtMs = System.currentTimeMillis();
        final AtomicBoolean cancelled = new AtomicBoolean(false);
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        boolean ackFinished;

        /**
         * Batch ID assigned by flushItems() when the batch is committed to
         * PrefillEndpoint.inflightBatches.  -1 means the batch has not been
         * committed yet (request is still in the batcher queue or dispatch
         * has not started).  Volatile so cancel() on another thread sees
         * the value set by flushItems() without explicit synchronization.
         */
        volatile long batchId = -1;

        InflightEntry(BatchItem item) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
        }

        @Override
        public long createdAtMs() {
            return createdAtMs;
        }
    }
}
