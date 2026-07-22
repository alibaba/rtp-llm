package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
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
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Coordinates batch scheduling for FlexLB disaggregated inference.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Request admission and routing</li>
 *   <li>Inflight lifecycle management (inflight map, TTL cleanup)</li>
 *   <li>Batch assembly coordination — commits to PrefillEndpoint,
 *       delegates gRPC dispatch to {@link BatchDispatcher}</li>
 *   <li>Resource rollback on failure or completion</li>
 * </ul>
 *
 * <p>The actual gRPC dispatch (build protobuf, send, parse response) is
 * delegated to {@link BatchDispatcher}. Per-item results come back through
 * {@link DispatchCallback} which this class implements.
 */
@Component
public class FlexlbBatchScheduler implements BatchDecisionHandler, DispatchCallback {

    private final ConfigService configService;
    private final Router router;
    private final EndpointRegistry endpointRegistry;
    private final BatchDispatcher dispatcher;
    private final BatchSchedulerReporter reporter;
    private final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    private final Map<Long, RequestLifecycleSnapshot> terminalStates = new ConcurrentHashMap<>();
    private final BatchIdGenerator batchIdGenerator;

    @Autowired
    public FlexlbBatchScheduler(ConfigService configService,
                                Router router,
                                EndpointRegistry endpointRegistry,
                                BatchDispatcher dispatcher,
                                BatchSchedulerReporter reporter,
                                Environment environment) {
        this.configService = configService;
        this.router = router;
        this.endpointRegistry = endpointRegistry;
        this.dispatcher = dispatcher;
        this.reporter = reporter;
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

            if (inflight.containsKey(ctx.getRequestId()) || terminalStates.containsKey(ctx.getRequestId())) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }

            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && inflight.size() >= maxInflight) {
                completeError(future, StrategyErrorType.QUEUE_FULL, null);
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
            InflightEntry entry = new InflightEntry(item);
            InflightEntry existing = inflight.putIfAbsent(ctx.getRequestId(), entry);
            if (existing != null || terminalStates.containsKey(ctx.getRequestId())) {
                if (existing == null) {
                    inflight.remove(ctx.getRequestId(), entry);
                }
                rollback(item);
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + ctx.getRequestId());
                return future;
            }
            WorkerBatcher batcher = prefillEp.getBatcher();
            ctx.setRouteSubmittedNanos(System.nanoTime());
            batcher.offer(item);

            // Report route+submit time: from schedule() entry (ctx.startTime) to batcher offer completion
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    prefillEp.getIp(),
                    prefillEp.ipPort(),
                    System.currentTimeMillis() - ctx.getStartTime());
        } catch (Throwable t) {
            if (ctx != null) {
                inflight.remove(ctx.getRequestId());
            }
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + t.getMessage());
        }
        return future;
    }

    // ==================== Completion from worker status ====================

    public void onWorkerStatusUpdate(WorkerStatusResponse response) {
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

            // Finish scheduler inflight state (prefill error, or any decode completion).
            InflightEntry entry = inflight.get(requestId);

            if (entry != null) {
                RequestLifecycleSnapshot terminal;
                synchronized (entry) {
                    RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
                    if (task.getBatchId() >= 0 && task.getBatchId() != current.batchId()) {
                        Logger.warn("Ignoring stale worker completion request_id={} batch_id={}",
                                requestId, task.getBatchId());
                        continue;
                    }
                    if (task.getErrorCode() == 0) {
                        terminal = entry.lifecycle.complete("decode completed");
                    } else {
                        terminal = entry.lifecycle.fail("worker error code " + task.getErrorCode());
                    }
                    if (isPrefill) {
                        rollbackOnce(entry);
                    }
                    finishEntry(entry, terminal);
                }
            }
        }
    }

    public int getInflightSize() {
        return inflight.size();
    }

    public RequestLifecycleSnapshot getRequestState(long requestId,
                                                    long expectedBatchId) {
        InflightEntry entry = inflight.get(requestId);
        RequestLifecycleSnapshot snapshot = entry != null
                ? entry.lifecycle.snapshot()
                : terminalStates.get(requestId);
        return batchMatches(snapshot, expectedBatchId) ? snapshot : null;
    }

    // ==================== Inflight TTL cleanup ====================

    @Scheduled(fixedRate = 60000L)
    public void cleanupInflight() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        long now = System.currentTimeMillis();
        for (Map.Entry<Long, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            if (now - entry.createdAtMs() <= ttlMs) {
                continue;
            }
            synchronized (entry) {
                if (inflight.get(candidate.getKey()) != entry) {
                    continue;
                }
                timeoutEntry(entry, "inflight TTL expired");
            }
        }
        long cutoff = System.currentTimeMillis() - ttlMs;
        terminalStates.entrySet().removeIf(entry -> entry.getValue().updatedAtMs() < cutoff);
    }

    // ==================== BatchDecisionHandler callbacks (from WorkerBatcher) ====================

    @Override
    public void onExpired(BatchItem head) {
        InflightEntry entry = entryFor(head);
        if (entry != null) {
            synchronized (entry) {
                timeoutEntry(entry, "batch SLO expired before dispatch");
            }
        } else if (!head.future().isDone() && !terminalStates.containsKey(head.requestId())) {
            rollback(head);
        }
    }

    @Override
    public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
        flushItems(items, meta);
    }

    @Override
    public void onOfferFailure(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            synchronized (entry) {
                rollbackOnce(entry);
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(
                        "batcher offer failed: " + error.getMessage());
                finishEntry(entry, terminal);
            }
        } else if (!item.future().isDone() && !terminalStates.containsKey(item.requestId())) {
            rollback(item);
        }
        completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED,
                "Batcher offer failed: " + error.getMessage());
    }

    // ==================== Dispatch pipeline ====================

    /**
     * Commit batch to PrefillEndpoint, then delegate to {@link BatchDispatcher}
     * for asynchronous gRPC dispatch.
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

        // [SYNC] Compute prediction and commit only active items to endpoint
        long predMs = 0;
        long batchId = batchIdGenerator.nextBatchId();
        List<BatchItem> dispatchable = new ArrayList<>(active.size());
        for (BatchItem item : active) {
            InflightEntry entry = entryFor(item);
            if (entry == null) {
                continue;
            }
            synchronized (entry) {
                if (entry.lifecycle.isTerminal()) {
                    continue;
                }
                entry.lifecycle.startDispatch(batchId);
                dispatchable.add(item);
            }
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
        reporter.reportBatchWaitTimeMs(RoleType.PREFILL.name(), prefillEp != null ? prefillEp.getIp() : "", prefillEp != null ? prefillEp.ipPort() : "", waitMs);
        FlexlbConfig config = configService.loadBalanceConfig();
        Logger.info("flexlb_batch_dispatch batch_id={} reason={} batch_size={} wait_ms={} "
                        + "predicted_ms={} threshold_ms={} fixed_wait_ms={} batch_size_max={} "
                        + "queue_after={} worker={}",
                batchId, reason, dispatchable.size(), waitMs, predMs,
                config.getFlexlbBatchPredictThresholdMs(), config.getFlexlbBatchFixedWaitMs(),
                config.getFlexlbBatchSizeMax(), meta.queueDepth(),
                prefillEp != null ? prefillEp.ipPort() : "");

        // Record dispatch timestamp for dispatch-to-ACK latency metric
        for (BatchItem item : dispatchable) {
            InflightEntry entry = entryFor(item);
            if (entry != null) {
                entry.lifecycle.markDispatched();
                item.ctx().setBatchDispatchedNanos(System.nanoTime());
            }
        }

        dispatcher.dispatch(dispatchable, prefillEp, batchId, predMs, reason, this);
    }

    // ==================== DispatchCallback implementation ====================

    @Override
    public void onSuccess(BatchItem item, long batchId) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            // A fast worker can report decode completion before the EnqueueBatch
            // ACK callback runs. The lifecycle is tombstoned, but the Schedule
            // future still needs the successful ACK response.
            RequestLifecycleSnapshot terminal = terminalStates.get(item.requestId());
            if (terminal != null
                    && terminal.state() == RequestLifecycleState.COMPLETED
                    && !item.future().isDone()) {
                item.ctx().setAckAtMs(System.currentTimeMillis());
                item.ctx().setAckAtNanos(System.nanoTime());
                completeSuccess(item);
            }
            return;
        }

        synchronized (entry) {
            long assignedBatchId = entry.lifecycle.snapshot().batchId();
            if (batchId != assignedBatchId) {
                Logger.warn("Ignoring stale EnqueueBatch ACK request_id={} batch_id={}",
                        item.requestId(), batchId);
                return;
            }
            RequestLifecycleSnapshot snapshot = entry.lifecycle.acknowledge();
            if (snapshot.state() == RequestLifecycleState.ACKNOWLEDGED) {
                // Record ACK timestamp for ack_to_response_time_ms metric (reported in FlexlbServiceImpl.completeSchedule)
                item.ctx().setAckAtMs(System.currentTimeMillis());
                item.ctx().setAckAtNanos(System.nanoTime());

                long dispatchedAtMs = entry.lifecycle.getDispatchedAtMs();
                if (dispatchedAtMs > 0) {
                    PrefillEndpoint ep = item.prefillEp();
                    reporter.reportDispatchAckTimeMs(
                            RoleType.PREFILL.name(),
                            ep != null ? ep.getIp() : "",
                            ep != null ? ep.ipPort() : "",
                            System.currentTimeMillis() - dispatchedAtMs);
                }
            }
            if (!snapshot.state().isTerminal() && !item.future().isDone()) {
                completeSuccess(item);
                Logger.debug("FlexLB batch enqueued request {} in batch_id={}",
                        item.requestId(), batchId);
            }
        }
    }

    private void completeSuccess(BatchItem item) {
        Response success = copyResponse(item.routeResponse());
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(true);
        success.setQueueLength(inflight.size());
        item.future().complete(success);
    }

    @Override
    public void onFailure(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            synchronized (entry) {
                rollbackOnce(entry);
                removeFromPrefillBatch(entry);
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(error.getMessage());
                finishEntry(entry, terminal);
                completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED, error.getMessage());
            }
            return;
        }
        if (!item.future().isDone() && !terminalStates.containsKey(item.requestId())) {
            rollback(item);
            completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED, error.getMessage());
        }
    }

    @Override
    public void onTimeout(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return;
        }
        synchronized (entry) {
            timeoutEntry(entry, "EnqueueBatch deadline exceeded: " + error.getMessage());
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

    private InflightEntry entryFor(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return entry != null && entry.item == item ? entry : null;
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
    private void removeFromPrefillBatch(InflightEntry entry) {
        long batchId = entry.lifecycle.snapshot().batchId();
        if (batchId <= 0) {
            return;
        }
        PrefillEndpoint prefillEp = entry.item.prefillEp();
        if (prefillEp != null) {
            prefillEp.repackBatch(batchId, Set.of(entry.item.requestId()));
            Logger.info("FlexLB remove from prefill batch: request_id={} batch_id={} engine={}",
                    entry.item.requestId(), batchId, prefillEp.getIp());
        }
    }

    private void timeoutEntry(InflightEntry entry, String detail) {
        RequestLifecycleSnapshot terminal = entry.lifecycle.timeout(detail);
        rollbackOnce(entry);
        removeFromPrefillBatch(entry);
        completeError(entry.item.future(), StrategyErrorType.BATCH_SLO_EXPIRED, detail);
        finishEntry(entry, terminal);
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

    private void finishEntry(InflightEntry entry,
                             RequestLifecycleSnapshot terminal) {
        // Publish the tombstone before removing inflight. submit() then observes
        // at least one side of the handoff and cannot revive the request ID.
        terminalStates.put(terminal.requestId(), terminal);
        inflight.remove(terminal.requestId(), entry);
    }

    private static boolean batchMatches(RequestLifecycleSnapshot snapshot,
                                        long expectedBatchId) {
        if (snapshot == null) {
            return false;
        }
        return expectedBatchId == 0 || snapshot.batchId() == expectedBatchId;
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

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
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

    private static final class InflightEntry {
        final BatchItem item;
        final RequestLifecycle lifecycle;
        final AtomicBoolean rolledBack = new AtomicBoolean(false);

        InflightEntry(BatchItem item) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
            this.lifecycle = new RequestLifecycle(item.requestId());
        }

        public long createdAtMs() {
            return lifecycle.snapshot().createdAtMs();
        }
    }
}
