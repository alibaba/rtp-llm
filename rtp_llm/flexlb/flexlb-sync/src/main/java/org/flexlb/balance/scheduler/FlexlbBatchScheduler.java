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
 *   <li>Request admission and routing</li>
 *   <li>Inflight lifecycle management (inflight map, TTL cleanup)</li>
 *   <li>Batch assembly coordination — commits to PrefillEndpoint,
 *       filters completed items, delegates gRPC dispatch to {@link BatchDispatcher}</li>
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
    final EngineWorkerStatus engineWorkerStatus;
    final EndpointRegistry endpointRegistry;
    final BatchDispatcher dispatcher;
    final BatchSchedulerReporter reporter;
    final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    final Map<Long, Object> admissionTokens = new ConcurrentHashMap<>();
    final AtomicLong batchIdGenerator = new AtomicLong(0);
    private final InflightEvictor<Long, InflightEntry> inflightEvictor
            = new InflightEvictor<>(inflight, entry -> {
                synchronized (entry) {
                    rollbackOnce(entry);
                    repackPrefillBatch(entry);
                    if (!entry.item.future().isDone()) {
                        entry.item.future().complete(Response.error(StrategyErrorType.BATCH_SLO_EXPIRED));
                    }
                    releaseAdmission(entry);
                }
            });

    public FlexlbBatchScheduler(ConfigService configService,
                                @Lazy Router router,
                                EngineWorkerStatus engineWorkerStatus,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter) {
        this.configService = configService;
        this.router = router;
        this.engineWorkerStatus = engineWorkerStatus;
        this.endpointRegistry = endpointRegistry;
        this.dispatcher = dispatcher;
        this.reporter = reporter;
    }

    // ==================== Request submission ====================

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        Object                      admissionToken = null;
        InflightEntry               submittedEntry = null;
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

            admissionToken = new Object();
            if (admissionTokens.putIfAbsent(ctx.getRequestId(), admissionToken) != null) {
                future.complete(Response.error(StrategyErrorType.INVALID_REQUEST));
                return future;
            }

            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                releaseAdmission(ctx.getRequestId(), admissionToken);
                future.complete(routeResponse != null
                        ? routeResponse
                        : Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollback(routeResponse);
                releaseAdmission(ctx.getRequestId(), admissionToken);
                Response resp = Response.error(StrategyErrorType.NO_PREFILL_WORKER);
                future.complete(resp);
                return future;
            }

            String prefillIpPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(prefillIpPort);
            if (prefillEp == null) {
                rollback(routeResponse);
                releaseAdmission(ctx.getRequestId(), admissionToken);
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
            submittedEntry = new InflightEntry(item, admissionToken);
            if (inflight.putIfAbsent(ctx.getRequestId(), submittedEntry) != null) {
                throw new IllegalStateException("request_id already exists after admission: " + ctx.getRequestId());
            }
            WorkerBatcher batcher = prefillEp.getBatcher();
            batcher.offer(item);
        } catch (Throwable t) {
            if (ctx != null) {
                if (submittedEntry != null && inflight.remove(ctx.getRequestId(), submittedEntry)) {
                    rollbackOnce(submittedEntry);
                }
                if (admissionToken != null) {
                    releaseAdmission(ctx.getRequestId(), admissionToken);
                }
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
            if (entry != null) {
                releaseAdmission(entry);
            }
            // Decode completion (success or error): scheduler only cleans its own map.
            // DecodeEndpoint.calibrate() independently handles its own inflightRequests cleanup.
        }
    }

    private boolean removeAndRollback(BatchItem item) {
        InflightEntry entry = matchingEntry(item);
        if (entry == null || !inflight.remove(item.requestId(), entry)) {
            return false;
        }
        synchronized (entry) {
            rollbackOnce(entry);
            releaseAdmission(entry);
        }
        return true;
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
        if (!removeAndRollback(head)) {
            return;
        }
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
        if (!removeAndRollback(item)) {
            return;
        }
        if (!item.future().isDone()) {
            Response errorResp = new Response();
            errorResp.setSuccess(false);
            errorResp.setCode(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode());
            errorResp.setErrorMessage("Batcher offer failed: " + error.getMessage());
            item.future().complete(errorResp);
        }
    }

    // ==================== Dispatch pipeline ====================

    private InflightEntry matchingEntry(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return entry != null && entry.item == item ? entry : null;
    }

    private void releaseAdmission(InflightEntry entry) {
        releaseAdmission(entry.item.requestId(), entry.admissionToken);
    }

    private void releaseAdmission(long requestId, Object admissionToken) {
        admissionTokens.remove(requestId, admissionToken);
    }

    /**
     * Commit batch to PrefillEndpoint, filter completed items, then delegate
     * to {@link BatchDispatcher} for asynchronous gRPC dispatch.
     * <p>
     * Filtering is done synchronously — it only reads inflight (ConcurrentHashMap)
     * and performs fast in-memory operations. The heavy gRPC I/O is handled
     * asynchronously by the dispatcher's own thread pool.
     */
    private void flushItems(List<BatchItem> items, String reason) {
        PrefillEndpoint prefillEp = items.get(0).prefillEp();

        // Claim only the current, unfinished entry for dispatch. Identity matching prevents a
        // stale queued item from affecting a newer request that reused the same request ID.
        List<BatchItem> active = new ArrayList<>();
        for (BatchItem item : items) {
            InflightEntry entry = matchingEntry(item);
            if (entry != null && !item.future().isDone()) {
                active.add(item);
            } else if (entry != null) {
                removeAndRollback(item);
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

        // Store batchId after commit so dispatcher skip and TTL cleanup can repack this batch.
        for (BatchItem item : active) {
            InflightEntry entry = matchingEntry(item);
            boolean       batchIdStored = false;
            if (entry != null) {
                synchronized (entry) {
                    if (inflight.get(item.requestId()) == entry) {
                        entry.batchId  = batchId;
                        batchIdStored = true;
                    }
                }
            }
            if (!batchIdStored && prefillEp != null) {
                // TTL cleanup won the commit-to-assignment race. Its entry still had batchId=-1,
                // so remove the old item from this exact batch without consulting request_id state.
                prefillEp.repackBatch(batchId, Set.of(item.requestId()));
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
        InflightEntry entry = matchingEntry(item);
        if (entry == null) {
            // A stale callback must not affect a newer request with the same request ID.
            return;
        }

        synchronized (entry) {
            if (!item.future().isDone()) {
                Response success = copyResponse(item.routeResponse());
                success.setSuccess(true);
                success.setCode(200);
                success.setEnqueuedByMaster(true);
                success.setQueueLength(inflight.size());
                item.future().complete(success);
                Logger.debug("FlexLB batch enqueued request {} in batch {}", item.requestId(), batchId);
            }
        }
    }

    @Override
    public void onFailure(BatchItem item, Throwable error) {
        failAck(item, error);
    }

    // ==================== Internal: inflight state management ====================

    void failAck(BatchItem item, Throwable error) {
        InflightEntry entry = matchingEntry(item);
        if (entry == null || !inflight.remove(item.requestId(), entry)) {
            return;
        }
        synchronized (entry) {
            rollbackOnce(entry);
            repackPrefillBatch(entry);
            if (!item.future().isDone()) {
                Response errorResp = new Response();
                errorResp.setSuccess(false);
                errorResp.setCode(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode());
                errorResp.setErrorMessage(error.getMessage());
                item.future().complete(errorResp);
            }
            releaseAdmission(entry);
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

    /**
     * Remove an undispatched or expired request from its prefill batch entry. Uses
     * {@link PrefillEndpoint#repackBatch} which:
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
            Logger.info("FlexLB batch repack: request_id={} batch_id={} engine={}",
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
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        final Object admissionToken;

        /**
         * Batch ID assigned by flushItems() when the batch is committed to
         * PrefillEndpoint.inflightBatches. -1 means the batch has not been committed yet.
         */
        volatile long batchId = -1;

        InflightEntry(BatchItem item, Object admissionToken) {
            this.item = Objects.requireNonNull(item);
            this.admissionToken = Objects.requireNonNull(admissionToken);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
        }

        @Override
        public long createdAtMs() {
            return createdAtMs;
        }
    }
}
