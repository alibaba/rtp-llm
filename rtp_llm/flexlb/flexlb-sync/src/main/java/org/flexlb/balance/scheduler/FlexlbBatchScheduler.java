package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/** Owns request admission, lifecycle and batch dispatch coordination. */
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
    private final Object admissionMutex = new Object();
    private int activeAdmissions;
    private boolean accepting = true;

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

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        if (ctx == null || ctx.getRequest() == null) {
            completeError(future, StrategyErrorType.INVALID_REQUEST, null);
            return future;
        }

        long requestId = ctx.getRequestId();
        InflightEntry entry;
        synchronized (admissionMutex) {
            if (!accepting) {
                completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        "batch scheduler stopped");
                return future;
            }
            if (inflight.containsKey(requestId) || terminalStates.containsKey(requestId)) {
                completeError(future, StrategyErrorType.INVALID_REQUEST,
                        "duplicate request_id: " + requestId);
                return future;
            }
            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && inflight.size() >= maxInflight) {
                completeError(future, StrategyErrorType.QUEUE_FULL, null);
                return future;
            }
            entry = new InflightEntry(ctx, future);
            inflight.put(requestId, entry);
            activeAdmissions++;
        }

        try {
            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                synchronized (entry) {
                    RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
                    if (!current.state().isTerminal()) {
                        current = entry.lifecycle.fail("route failed");
                    }
                    finishEntry(entry, current);
                }
                if (routeResponse != null) {
                    future.complete(routeResponse);
                } else {
                    completeError(future, StrategyErrorType.NO_AVAILABLE_WORKER, null);
                }
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            DecodeEndpoint.Lease decodeLease = null;
            if (decode != null) {
                String decodeIpPort = decode.getServerIp() + ":" + decode.getHttpPort();
                DecodeEndpoint decodeEp = endpointRegistry.getDecode(decodeIpPort);
                decodeLease = decodeEp == null ? null : decodeEp.leaseFor(requestId, ctx);
            }
            if (prefill == null) {
                release(decodeLease);
                completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                synchronized (entry) {
                    finishEntry(entry, entry.lifecycle.fail("prefill route missing"));
                }
                return future;
            }

            String prefillIpPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
            PrefillEndpoint prefillEp = endpointRegistry.getPrefill(prefillIpPort);
            if (prefillEp == null) {
                release(decodeLease);
                completeError(future, StrategyErrorType.NO_PREFILL_WORKER, null);
                synchronized (entry) {
                    finishEntry(entry, entry.lifecycle.fail("prefill endpoint missing"));
                }
                return future;
            }

            long hitCache = prefill.getDebugInfo() == null ? 0 : prefill.getDebugInfo().getHitCacheLen();
            BatchItem item = new BatchItem(ctx, future, routeResponse, hitCache,
                    prefillEp, System.currentTimeMillis());
            synchronized (entry) {
                entry.decodeLease = decodeLease;
                if (entry.lifecycle.isTerminal()) {
                    rollbackOnce(entry);
                    finishEntry(entry, entry.lifecycle.snapshot());
                    return future;
                }
                entry.item = item;
                entry.lifecycle.queued();
                ctx.setRouteSubmittedNanos(System.nanoTime());
                entry.queueHandle = prefillEp.getBatcher().offer(item);
            }

            try {
                reporter.reportRouteSubmitTimeMs(
                        RoleType.PREFILL.name(),
                        prefillEp.getIp(),
                        System.currentTimeMillis() - ctx.getStartTime());
            } catch (RuntimeException reportFailure) {
                Logger.debug("FlexLB route-submit metric failed", reportFailure);
            }
        } catch (RuntimeException failure) {
            synchronized (entry) {
                rollbackOnce(entry);
                RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
                if (!current.state().isTerminal()) {
                    current = entry.lifecycle.fail("submit failed: " + failure.getMessage());
                }
                finishEntry(entry, current);
            }
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    requestId, failure);
            completeError(future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                    "Submit failed: " + failure.getMessage());
        } finally {
            synchronized (admissionMutex) {
                activeAdmissions--;
                admissionMutex.notifyAll();
            }
        }
        return future;
    }

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

            if (isPrefill && task.getErrorCode() == 0) {
                continue;
            }

            InflightEntry entry = inflight.get(requestId);

            if (entry != null) {
                RequestLifecycleSnapshot terminal;
                synchronized (entry) {
                    RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
                    if (task.getBatchId() <= 0 || task.getBatchId() != current.batchId()) {
                        Logger.warn("Ignoring stale worker completion request_id={} batch_id={}",
                                requestId, task.getBatchId());
                        continue;
                    }
                    if (task.getErrorCode() == 0) {
                        terminal = entry.lifecycle.complete("decode completed");
                    } else {
                        terminal = entry.lifecycle.fail("worker error code " + task.getErrorCode());
                        completeError(entry.future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                                terminal.detail());
                    }
                    if (terminal.state() != RequestLifecycleState.COMPLETED) {
                        repackPrefillBatch(entry);
                    }
                    rollbackOnce(entry);
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
                if (entry.item == null) {
                    RequestLifecycleSnapshot terminal = entry.lifecycle.timeout(
                            "inflight TTL expired while routing");
                    completeError(entry.future, StrategyErrorType.BATCH_SLO_EXPIRED,
                            terminal.detail());
                    finishEntry(entry, terminal);
                    continue;
                }
                timeoutEntry(entry, "inflight TTL expired");
            }
        }
        // Do not evict tombstones published during this cleanup pass.
        long cutoff = now - ttlMs;
        terminalStates.entrySet().removeIf(entry -> entry.getValue().updatedAtMs() < cutoff);
    }

    @Override
    public void onExpired(BatchItem head) {
        InflightEntry entry = entryFor(head);
        if (entry != null) {
            synchronized (entry) {
                entry.queueHandle = null;
                timeoutEntry(entry, "batch SLO expired before dispatch");
            }
        }
    }

    @Override
    public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
        try {
            flushItems(items, meta);
        } catch (Throwable failure) {
            Logger.error("FlexLB claimed batch preparation failed, items={}",
                    items.size(), failure);
            failClaimedItems(items, failure);
        }
    }

    @Override
    public void onOfferFailure(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry != null) {
            synchronized (entry) {
                entry.queueHandle = null;
                rollbackOnce(entry);
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(
                        "batcher offer failed: " + error.getMessage());
                finishEntry(entry, terminal);
            }
        }
        completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED,
                "Batcher offer failed: " + error.getMessage());
    }

    private void flushItems(List<BatchItem> items, DispatchMeta meta) {
        String reason = meta.reason();
        PrefillEndpoint prefillEp = items.get(0).prefillEp();

        for (BatchItem item : items) {
            InflightEntry entry = entryFor(item);
            if (entry != null) {
                synchronized (entry) {
                    entry.queueHandle = null;
                }
            }
        }

        List<BatchItem> active = items.stream()
                .filter(item -> !item.future().isDone())
                .toList();

        if (active.isEmpty()) {
            return;
        }

        long predMs = 0;
        long batchId = batchIdGenerator.nextBatchId();
        List<BatchItem> dispatchable = new ArrayList<>(active.size());
        for (BatchItem item : active) {
            if (tryStartDispatch(item, batchId)) {
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

        long waitMs = System.currentTimeMillis() - items.get(0).enqueuedAtMs();
        reportDispatchBestEffort(prefillEp, batchId, reason, dispatchable.size(),
                waitMs, predMs, meta.queueDepth());

        for (BatchItem item : dispatchable) {
            InflightEntry entry = entryFor(item);
            if (entry != null) {
                entry.lifecycle.markDispatched();
                item.ctx().setBatchDispatchedNanos(System.nanoTime());
            }
        }

        dispatcher.dispatch(dispatchable, prefillEp, batchId, predMs, reason, this);
    }

    private void reportDispatchBestEffort(PrefillEndpoint prefillEp,
                                          long batchId,
                                          String reason,
                                          int batchSize,
                                          long waitMs,
                                          long predMs,
                                          int queueDepth) {
        try {
            reporter.reportBatchWaitTimeMs(RoleType.PREFILL.name(),
                    prefillEp != null ? prefillEp.getIp() : "",
                    waitMs);
            FlexlbConfig config = configService.loadBalanceConfig();
            Logger.info("flexlb_batch_dispatch batch_id={} reason={} batch_size={} wait_ms={} "
                            + "predicted_ms={} threshold_ms={} fixed_wait_ms={} batch_size_max={} "
                            + "queue_after={} worker={}",
                    batchId, reason, batchSize, waitMs, predMs,
                    config.getFlexlbBatchPredictThresholdMs(), config.getFlexlbBatchFixedWaitMs(),
                    config.getFlexlbBatchSizeMax(), queueDepth,
                    prefillEp != null ? prefillEp.ipPort() : "");
        } catch (RuntimeException metricsFailure) {
            Logger.warn("FlexLB batch dispatch telemetry failed batch_id={}",
                    batchId, metricsFailure);
        }
    }

    private void failClaimedItems(List<BatchItem> items, Throwable failure) {
        Set<Long> releasedBatches = new HashSet<>();
        for (BatchItem item : items) {
            long batchId = item.batchId();
            PrefillEndpoint prefillEp = item.prefillEp();
            if (batchId > 0 && prefillEp != null && releasedBatches.add(batchId)) {
                try {
                    prefillEp.releaseBatch(batchId);
                } catch (RuntimeException releaseFailure) {
                    Logger.error("FlexLB failed to release claimed Prefill batch {}",
                            batchId, releaseFailure);
                }
            }
        }

        String message = failure.getMessage() == null
                ? failure.getClass().getSimpleName() : failure.getMessage();
        for (BatchItem item : items) {
            InflightEntry entry = entryFor(item);
            if (entry == null) {
                completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED, message);
                continue;
            }
            synchronized (entry) {
                rollbackOnce(entry);
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(message);
                finishEntry(entry, terminal);
                completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED, message);
            }
        }
    }

    boolean tryStartDispatch(BatchItem item, long batchId) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            return false;
        }
        synchronized (entry) {
            RequestLifecycleSnapshot current = entry.lifecycle.snapshot();
            if (current.state().isTerminal()) {
                rollbackOnce(entry);
                finishEntry(entry, current);
                return false;
            }
            if (entry.decodeLease != null && !entry.decodeLease.bindBatch(batchId)) {
                rollbackOnce(entry);
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(
                        "decode reservation expired before dispatch");
                completeError(entry.future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        terminal.detail());
                finishEntry(entry, terminal);
                return false;
            }
            entry.lifecycle.startDispatch(batchId);
            item.assignBatchId(batchId);
            return true;
        }
    }

    @Override
    public void onSuccess(BatchItem item, long batchId) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            completeFromWorkerCompletion(item);
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
                item.ctx().setAckAtMs(System.currentTimeMillis());
                item.ctx().setAckAtNanos(System.nanoTime());

                long dispatchedAtMs = entry.lifecycle.getDispatchedAtMs();
                if (dispatchedAtMs > 0) {
                    reportDispatchAckBestEffort(item, batchId,
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

    private void reportDispatchAckBestEffort(BatchItem item, long batchId, long ackMs) {
        try {
            PrefillEndpoint ep = item.prefillEp();
            reporter.reportDispatchAckTimeMs(
                    RoleType.PREFILL.name(),
                    ep != null ? ep.getIp() : "",
                    ackMs);
        } catch (RuntimeException metricsFailure) {
            Logger.warn("FlexLB dispatch ACK telemetry failed batch_id={} request_id={}",
                    batchId, item.requestId(), metricsFailure);
        }
    }

    private void completeSuccess(BatchItem item) {
        Response success = item.routeResponse();
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
                repackPrefillBatch(entry);
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(error.getMessage());
                finishEntry(entry, terminal);
                completeError(item.future(), StrategyErrorType.BATCH_DISPATCH_FAILED, error.getMessage());
            }
            return;
        }
        completeFromWorkerCompletion(item);
    }

    @Override
    public void onTimeout(BatchItem item, Throwable error) {
        InflightEntry entry = entryFor(item);
        if (entry == null) {
            completeFromWorkerCompletion(item);
            return;
        }
        synchronized (entry) {
            timeoutEntry(entry, "EnqueueBatch deadline exceeded: " + error.getMessage());
        }
    }

    private void completeFromWorkerCompletion(BatchItem item) {
        RequestLifecycleSnapshot terminal = terminalStates.get(item.requestId());
        if (terminal != null
                && terminal.state() == RequestLifecycleState.COMPLETED
                && terminal.batchId() == item.batchId()
                && !item.future().isDone()) {
            item.ctx().setAckAtMs(System.currentTimeMillis());
            item.ctx().setAckAtNanos(System.nanoTime());
            completeSuccess(item);
        }
    }

    private void rollbackOnce(InflightEntry entry) {
        release(entry.decodeLease);
        entry.decodeLease = null;
    }

    private static void release(DecodeEndpoint.Lease lease) {
        if (lease != null) {
            lease.release();
        }
    }

    private InflightEntry entryFor(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return entry != null && entry.item == item ? entry : null;
    }

    /** Idempotently removes this request from the endpoint's committed batch. */
    private void repackPrefillBatch(InflightEntry entry) {
        long batchId = entry.lifecycle.snapshot().batchId();
        if (batchId <= 0) {
            return;
        }
        PrefillEndpoint prefillEp = entry.item.prefillEp();
        if (prefillEp != null) {
            try {
                prefillEp.repackBatch(batchId, Set.of(entry.item.requestId()));
                Logger.info("FlexLB remove from prefill batch: request_id={} batch_id={} engine={}",
                        entry.item.requestId(), batchId, prefillEp.getIp());
            } catch (RuntimeException cleanupFailure) {
                Logger.error("FlexLB repack failed request_id={} batch_id={} engine={}",
                        entry.item.requestId(), batchId, prefillEp.getIp(), cleanupFailure);
            }
        }
    }

    private void timeoutEntry(InflightEntry entry, String detail) {
        RequestLifecycleSnapshot terminal = entry.lifecycle.timeout(detail);
        rollbackOnce(entry);
        repackPrefillBatch(entry);
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
        synchronized (entry) {
            if (entry.finished) {
                return;
            }
            entry.finished = true;
            // Publish the tombstone before removing inflight. submit() then observes
            // at least one side of the handoff and cannot revive the request ID.
            terminalStates.put(terminal.requestId(), terminal);
            inflight.remove(terminal.requestId(), entry);
        }
    }

    private static boolean batchMatches(RequestLifecycleSnapshot snapshot,
                                        long expectedBatchId) {
        if (snapshot == null) {
            return false;
        }
        return expectedBatchId == 0 || snapshot.batchId() == expectedBatchId;
    }

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

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void reportBatchMetrics() {
        reporter.reportSchedulerInflightSize(inflight.size());

        for (Map.Entry<String, PrefillEndpoint> entry : endpointRegistry.getPrefillEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }

        for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
            entry.getValue().reportBatchMetrics(reporter);
        }
    }

    @PreDestroy
    public void shutdown() {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        boolean interrupted = false;
        synchronized (admissionMutex) {
            accepting = false;
            while (activeAdmissions > 0) {
                long remainingNanos = deadlineNanos - System.nanoTime();
                if (remainingNanos <= 0) {
                    break;
                }
                try {
                    TimeUnit.NANOSECONDS.timedWait(admissionMutex, remainingNanos);
                } catch (InterruptedException ignored) {
                    interrupted = true;
                    break;
                }
            }
        }
        failRoutingAdmissionsForShutdown();
        dispatcher.shutdown();
        endpointRegistry.close();
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void failRoutingAdmissionsForShutdown() {
        for (Map.Entry<Long, InflightEntry> candidate : inflight.entrySet()) {
            InflightEntry entry = candidate.getValue();
            synchronized (entry) {
                if (inflight.get(candidate.getKey()) != entry
                        || entry.lifecycle.snapshot().state() != RequestLifecycleState.ROUTING) {
                    continue;
                }
                rollbackOnce(entry);
                for (DecodeEndpoint decode : endpointRegistry.getDecodeEndpoints().values()) {
                    release(decode.leaseFor(candidate.getKey(), entry.context));
                }
                RequestLifecycleSnapshot terminal = entry.lifecycle.fail(
                        "scheduler stopped while routing");
                completeError(entry.future, StrategyErrorType.BATCH_DISPATCH_FAILED,
                        terminal.detail());
                finishEntry(entry, terminal);
            }
        }
    }

    private static final class InflightEntry {
        final BalanceContext context;
        final CompletableFuture<Response> future;
        final RequestLifecycle lifecycle;
        BatchItem item;
        DecodeEndpoint.Lease decodeLease;
        WorkerBatcher.QueueHandle queueHandle;
        boolean finished;

        InflightEntry(BalanceContext context, CompletableFuture<Response> future) {
            this.context = context;
            this.future = future;
            this.lifecycle = new RequestLifecycle(
                    context.getRequestId(), RequestLifecycleState.ROUTING);
        }

        public long createdAtMs() {
            return lifecycle.snapshot().createdAtMs();
        }
    }
}
