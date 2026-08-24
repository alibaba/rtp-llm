package org.flexlb.balance.endpoint;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DispatchMeta;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightState;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.FormulaPredictor;
import org.flexlb.balance.strategy.LearningPredictor;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntSupplier;
import java.util.stream.Collectors;

/**
 * Prefill worker endpoint.
 *
 * <p>Owns the full batch pipeline for its worker: batching (via the embedded
 * {@link WorkerBatcher}), batch commit / inflight tracking, and asynchronous
 * gRPC dispatch to the engine ({@link #submitBatch}). Per-item results are
 * settled directly through {@link BatchItem} terminal transitions — no
 * scheduler callback is involved.
 */
public class PrefillEndpoint extends WorkerEndpoint {

    private static final org.slf4j.Logger logger = LoggerFactory.getLogger("syncLogger");

    private final FlexlbConfig config;
    private final EngineGrpcClient grpcClient;
    private final BatchDispatchExecutor dispatchExecutor;
    private final BatchIdGenerator batchIdGenerator;
    private final IntSupplier globalActiveCount;
    private final PrefillTimePredictor predictor;
    private final InflightStore inflightStore;

    /** Layer 1: dispatched, not yet acknowledged by the engine (strict inflight). */
    private final ConcurrentHashMap<Long, PrefillInflightEntry> inflightEntries = new ConcurrentHashMap<>();

    /** Layer 2: engine-acknowledged tasks with phase state and lastSeenRound. */
    private final ConcurrentHashMap<Long, EngineTask<PrefillInflightEntry>> engineWork = new ConcurrentHashMap<>();

    /**
     * Total requests tracked across both layers (sum of requestCount per map
     * membership). Increments/decrements are tied to successful map inserts/
     * removals, so the layer-1 → layer-2 migration window can only cause a
     * transient over-count (conservative, rejection-biased), never a
     * double-decrement.
     */
    private final AtomicInteger inflightRequestCount = new AtomicInteger(0);
    private final WorkerBatcher batcher;
    private final InflightEvictor<Long, PrefillInflightEntry> inflightEvictor;
    private final InflightEvictor<Long, EngineTask<PrefillInflightEntry>> engineWorkEvictor;
    private final BatchSchedulerReporter reporter;

    /** Monotonic calibrate round counter driving stale engineWork eviction. */
    private final AtomicLong calibrateRound = new AtomicLong(0);

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           EngineGrpcClient grpcClient,
                           BatchDispatchExecutor dispatchExecutor,
                           BatchIdGenerator batchIdGenerator,
                           IntSupplier globalActiveCount,
                           BatchSchedulerReporter reporter,
                           InflightStore inflightStore) {
        super(status);
        this.config = config;
        this.grpcClient = grpcClient;
        this.dispatchExecutor = dispatchExecutor;
        this.batchIdGenerator = batchIdGenerator;
        this.globalActiveCount = globalActiveCount;
        this.reporter = reporter;
        this.inflightStore = inflightStore;
        this.predictor = createPredictor(config);
        this.batcher = new WorkerBatcher(status.getIpPort(), this, config, reporter);
        this.inflightEvictor = new InflightEvictor<>(inflightEntries,
                entry -> inflightRequestCount.addAndGet(-entry.requestCount()));
        this.engineWorkEvictor = new InflightEvictor<>(engineWork,
                task -> inflightRequestCount.addAndGet(-task.entry().requestCount()));
        this.batcher.start();
    }

    public WorkerBatcher getBatcher() {
        return batcher;
    }

    @Override
    public void close() {
        try {
            batcher.shutdown();
        } finally {
            drainInflight("EP closed");
            super.close();
        }
    }

    /**
     * Drain all tracked inflight entries from both layers, terminating their
     * bound {@link InflightItem}s so clients are notified immediately instead
     * of waiting for the 300s TTL safety net (review A4).
     *
     * <p>Collects all items first, clears the maps, then terminates — this
     * avoids concurrent modification when terminate() triggers whenComplete
     * callbacks that call back into the EP's maps (all idempotent no-ops
     * once the maps are cleared).
     */
    private void drainInflight(String reason) {
        List<InflightItem> toTerminate = new ArrayList<>();
        for (PrefillInflightEntry entry : inflightEntries.values()) {
            collectEntryItems(entry, toTerminate);
        }
        for (EngineTask<PrefillInflightEntry> task : engineWork.values()) {
            collectEntryItems(task.entry(), toTerminate);
        }
        inflightEntries.clear();
        engineWork.clear();
        inflightRequestCount.set(0);
        for (InflightItem item : toTerminate) {
            if (!item.isTerminated()) {
                item.complete(Response.error(StrategyErrorType.WORKER_EXECUTION_FAILED, reason),
                        InflightState.FAILED);
            }
        }
    }

    /**
     * Collect all {@link InflightItem}s bound to an inflight entry by looking
     * them up in the {@link InflightStore} by requestId.
     */
    private void collectEntryItems(PrefillInflightEntry entry, List<InflightItem> sink) {
        if (inflightStore == null) return;
        switch (entry) {
            case PrefillInflightBatch batch -> {
                for (BatchItem item : batch.requests()) {
                    collectItem(item.requestId(), sink);
                }
            }
            case PrefillInflightRequest request -> collectItem(request.requestId(), sink);
        }
    }

    /**
     * Look up an {@link InflightItem} by requestId and add it to the sink if
     * found and not already terminal. Null-safe on {@code inflightStore}
     * (tests may pass null).
     */
    private void collectItem(long requestId, List<InflightItem> sink) {
        if (inflightStore == null) return;
        InflightItem item = inflightStore.get(String.valueOf(requestId));
        if (item != null && !item.isTerminated()) {
            sink.add(item);
        }
    }

    public long batcherWaitMs() {
        return batcher.queueWaitMs();
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig cfg) {
        if ("learning".equalsIgnoreCase(cfg.getPrefillPredictorType())) {
            return new LearningPredictor();
        }
        return new FormulaPredictor(cfg.getCostFormula());
    }

    /** Commit a dispatched batch into layer-1 inflight tracking. */
    public void commitBatch(long batchId, long predictMs, List<BatchItem> requests) {
        putInflightEntry(batchId,
                new PrefillInflightBatch(batchId, predictMs, requests, System.currentTimeMillis()));
    }

    /**
     * Commit a single directly-dispatched request (non-batch path:
     * CostBased / ShortestTTFT) into layer-1 inflight tracking, keyed by
     * requestId (the engine reports these with {@code batch_id=-1}).
     */
    public void commitRequest(long requestId, long predictMs) {
        putInflightEntry(requestId,
                new PrefillInflightRequest(requestId, predictMs, System.currentTimeMillis()));
    }

    private void putInflightEntry(long key, PrefillInflightEntry entry) {
        PrefillInflightEntry prev = inflightEntries.put(key, entry);
        if (prev != null) {
            // key already exists — subtract the old request count before overwriting,
            // otherwise the old value is silently lost and the counter stays inflated.
            inflightRequestCount.addAndGet(-prev.requestCount());
        }
        inflightRequestCount.addAndGet(entry.requestCount());
    }

    /** Remove the tracked entry for {@code batchId} from both inflight layers. */
    public void releaseBatch(long batchId) {
        EngineTask<PrefillInflightEntry> task = engineWork.remove(batchId);
        if (task != null) {
            inflightRequestCount.addAndGet(-task.entry().requestCount());
        }
        PrefillInflightEntry entry = inflightEntries.remove(batchId);
        if (entry != null) {
            inflightRequestCount.addAndGet(-entry.requestCount());
        }
    }

    /**
     * Unified release entry point for EP-level resource cleanup.
     * <p>Non-batch path: {@code requestId} is used directly as the inflight
     * key (engine reports these with {@code batch_id=-1}), so we delegate to
     * {@link #releaseBatch(long)} which removes the entry and decrements
     * counters.
     * <p>Batch path: batch-level release is handled separately by the batch's
     * own terminate/complete path calling {@link #releaseBatch(long)} with
     * the batchId, so this method does not need to handle that case.
     */
    @Override
    public void release(long requestId) {
        releaseBatch(requestId);
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     * Works on whichever layer currently tracks the batch.
     */
    public void repackBatch(long batchId, Set<Long> failedRequestIds) {
        engineWork.computeIfPresent(batchId, (id, task) -> {
            PrefillInflightEntry shrunk = shrinkEntry(task.entry(), failedRequestIds);
            if (shrunk == null) {
                return null; // removes entry from map
            }
            if (shrunk != task.entry()) {
                task.updateEntry(shrunk);
            }
            return task;
        });
        inflightEntries.computeIfPresent(batchId, (id, entry) -> shrinkEntry(entry, failedRequestIds));
    }

    /**
     * Drop the given requestIds from an inflight entry, adjusting counters.
     *
     * @return the shrunk entry, the same entry if nothing matched, or
     *         {@code null} if no request survives (entry should be removed)
     */
    private PrefillInflightEntry shrinkEntry(PrefillInflightEntry entry, Set<Long> droppedRequestIds) {
        return switch (entry) {
            case PrefillInflightRequest request -> {
                if (!droppedRequestIds.contains(request.requestId())) {
                    yield request;
                }
                inflightRequestCount.addAndGet(-1);
                yield null;
            }
            case PrefillInflightBatch batch -> {
                List<BatchItem> survivors = batch.requests().stream()
                        .filter(r -> !droppedRequestIds.contains(r.requestId()))
                        .toList();
                if (survivors.size() == batch.requestCount()) {
                    yield batch;
                }
                inflightRequestCount.addAndGet(-(batch.requestCount() - survivors.size()));
                if (survivors.isEmpty()) {
                    yield null;
                }
                long newPredMs = (long) predictor.predictBatchMs(survivors);
                yield batch.repack(newPredMs, survivors);
            }
        };
    }

    // ==================== Batch submission (sync part) ====================

    /**
     * EP-level batch submission — the full dispatch function lives on the EP.
     *
     * <p>Synchronous part: filter already-finished items, assign a batch ID,
     * run DP-aware prediction, commit the batch into inflight tracking, then
     * hand the batch to the shared dispatch executor for asynchronous gRPC
     * dispatch. Per-item results are settled through {@link BatchItem}
     * terminal transitions.
     *
     * @param items batch assembled by the {@link WorkerBatcher} (all items
     *              belong to this endpoint)
     * @param meta  dispatch metadata from the batcher algorithm
     */
    public void submitBatch(List<BatchItem> items, DispatchMeta meta) {
        // A timeout or prior failure may finish an item while it is still queued.
        List<BatchItem> dispatchable = new ArrayList<>(items.size());
        for (BatchItem item : items) {
            if (!item.future().isDone()) {
                dispatchable.add(item);
            }
        }
        if (dispatchable.isEmpty()) {
            return;
        }

        // [SYNC] Assign batch ID, DP-aware prediction (bucket effect: max across
        // DP ranks), commit into inflight tracking
        long batchId = batchIdGenerator.nextBatchId();
        for (BatchItem item : dispatchable) {
            item.setAssignedBatchId(batchId);
        }
        long predMs = (long) predictor.predictBatchMsByDp(groupByDpRank(dispatchable));
        commitBatch(batchId, predMs, dispatchable);

        // Report queue wait: from batcher enqueue to dispatch hand-off
        long waitMs = System.currentTimeMillis() - items.get(0).enqueuedAtMs();
        reporter.reportBatchWaitTimeMs(RoleType.PREFILL.name(), getIp(), waitMs);

        // Record dispatch timestamp for dispatch-to-ACK latency metric
        for (BatchItem item : dispatchable) {
            item.setDispatchedAtMs(System.currentTimeMillis());
            item.ctx().setBatchDispatchedNanos(System.nanoTime());
        }

        // [ASYNC] gRPC dispatch on the shared executor
        dispatch(dispatchable, batchId, predMs, meta.reason());
    }

    /** Group items by prefill DP rank (sorted ascending) for DP-aware prediction. */
    private static List<List<BatchItem>> groupByDpRank(List<BatchItem> items) {
        Map<Long, List<BatchItem>> byDpRank = new TreeMap<>();
        for (BatchItem item : items) {
            Long rank = item.prefill() != null ? item.prefill().getDpRank() : null;
            byDpRank.computeIfAbsent(rank != null ? rank : 0L, ignored -> new ArrayList<>()).add(item);
        }
        return new ArrayList<>(byDpRank.values());
    }

    // ==================== Batch dispatch (async part, runs on shared executor) ====================

    private void dispatch(List<BatchItem> items, long batchId, long predMs, String reason) {
        try {
            dispatchExecutor.execute(() -> doDispatch(items, batchId, predMs, reason));
        } catch (RejectedExecutionException e) {
            Logger.warn("FlexLB batch dispatch rejected (executor shutdown), failing {} items", items.size());
            repackBatch(batchId, items.stream()
                    .map(BatchItem::requestId)
                    .collect(Collectors.toSet()));
            for (BatchItem item : items) {
                item.failDispatch(e);
            }
        }
    }

    private void doDispatch(List<BatchItem> items, long batchId, long predMs, String reason) {
        try {
            doDispatchInternal(items, batchId, predMs, reason);
        } catch (Throwable t) {
            // Safety net: ensure every item is settled even for unexpected errors
            Logger.error("Unexpected error in doDispatch batchId={}", batchId, t);
            for (BatchItem item : items) {
                try {
                    item.failDispatch(t);
                } catch (Throwable ignored) {
                    // best-effort
                }
            }
        }
    }

    private void doDispatchInternal(List<BatchItem> items, long batchId, long predMs, String reason) {
        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, items);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failItems(items, batchId, "Batch request build failed: " + e.getMessage());
            return;
        }

        // 2. Log dispatch
        logDispatch(batchId, items, predMs, reason);

        // 3. Send gRPC (async)
        long deadlineMs = config.getFlexlbBatchEnqueueDeadlineMs();
        grpcClient.batchEnqueueAsync(getIp(), getGrpcPort(), request, deadlineMs)
                .whenCompleteAsync((response, ex) -> {
                    try {
                        if (ex != null) {
                            Throwable cause = ex instanceof CompletionException ? ex.getCause() : ex;
                            Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                                    batchId, getIp(), getGrpcPort(), cause.getMessage());
                            if (Status.fromThrowable(cause).getCode() == Status.Code.DEADLINE_EXCEEDED) {
                                repackBatch(batchId, items.stream()
                                        .map(BatchItem::requestId)
                                        .collect(Collectors.toSet()));
                                for (BatchItem item : items) {
                                    item.failTimeout(cause);
                                }
                            } else {
                                failItems(items, batchId,
                                        "gRPC dispatch failed: " + cause.getMessage());
                            }
                        } else if (response == null) {
                            failItems(items, batchId, "EnqueueBatch returned null response");
                        } else {
                            handleResponse(batchId, items, response);
                        }
                    } catch (Throwable t) {
                        // Safety net: ensure every item is settled even for unexpected errors
                        Logger.error("Unexpected error in EnqueueBatch callback batchId={}", batchId, t);
                        failItems(items, batchId,
                                "Unexpected callback error: " + t.getMessage());
                    }
                }, dispatchExecutor);
    }

    private void failItems(List<BatchItem> items, long batchId, String message) {
        // Use repackBatch (computeIfPresent) instead of releaseBatch (remove)
        // to avoid the dual-layer accounting race: releaseBatch's non-atomic remove from both
        // layers can race with calibrate's layer-1→layer-2 migration,
        // causing inflightRequestCount to underflow. repackBatch with all
        // request IDs atomically shrinks the entry to zero survivors on
        // whichever layer currently tracks it.
        Set<Long> failedIds = items.stream()
                .map(BatchItem::requestId)
                .collect(Collectors.toSet());
        repackBatch(batchId, failedIds);
        RuntimeException error = new RuntimeException(message);
        for (BatchItem item : items) {
            item.failDispatch(error);
        }
    }

    // ==================== Response parsing ====================

    private void handleResponse(long batchId, List<BatchItem> items,
                                EngineRpcService.EnqueueBatchResponsePB response) {
        if (response.getBatchId() != batchId) {
            RuntimeException mismatch = new RuntimeException(
                    "EnqueueBatch batch_id mismatch: expected " + batchId
                            + " but got " + response.getBatchId());
            for (BatchItem item : items) {
                item.failDispatch(mismatch);
            }
            return;
        }
        Map<Long, EngineRpcService.EnqueueBatchErrorPB> errorByRequestId = new HashMap<>();
        for (EngineRpcService.EnqueueBatchErrorPB error : response.getErrorsList()) {
            errorByRequestId.put(error.getRequestId(), error);
        }
        Set<Long> successIds = new HashSet<>();
        for (EngineRpcService.EnqueueBatchSuccessPB success : response.getSuccessesList()) {
            successIds.add(success.getRequestId());
        }

        for (BatchItem item : items) {
            if (successIds.contains(item.requestId())) {
                onDispatchSuccess(item, batchId);
            } else if (errorByRequestId.containsKey(item.requestId())) {
                EngineRpcService.EnqueueBatchErrorPB error = errorByRequestId.get(item.requestId());
                String errorMessage = error.hasErrorInfo()
                        ? error.getErrorInfo().getErrorMessage()
                        : "missing error_info";
                item.failDispatch(new RuntimeException(
                        "EnqueueBatch rejected request " + item.requestId() + ": " + errorMessage));
            } else {
                item.failDispatch(new RuntimeException(
                        "EnqueueBatch missing ack for request " + item.requestId()));
            }
        }
    }

    private void onDispatchSuccess(BatchItem item, long batchId) {
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
            reporter.reportDispatchAckTimeMs(
                    RoleType.PREFILL.name(), getIp(),
                    System.currentTimeMillis() - item.dispatchedAtMs());
        }

        item.completeSuccess(globalActiveCount.getAsInt());
        Logger.debug("FlexLB batch enqueued request {} in batch_id={}",
                item.requestId(), batchId);
    }

    // ==================== gRPC request building ====================

    private EngineRpcService.EnqueueBatchRequestPB buildBatchRequest(long batchId, List<BatchItem> items)
            throws InvalidProtocolBufferException {
        EngineRpcService.EnqueueBatchRequestPB.Builder builder =
                EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(batchId);
        Map<Long, List<BatchItem>> byDpRank = new HashMap<>();
        for (BatchItem item : items) {
            byDpRank.computeIfAbsent(item.prefill().getDpRank(), ignored -> new ArrayList<>()).add(item);
        }
        try {
            byDpRank.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(entry -> {
                        EngineRpcService.EnqueueBatchDpSlotPB.Builder slot =
                                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                                        .setDpRank(entry.getKey().intValue());
                        for (BatchItem item : entry.getValue()) {
                            try {
                                slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                                        .setInput(buildInput(item))
                                        .build());
                            } catch (InvalidProtocolBufferException e) {
                                throw new BatchRequestBuildException(e);
                            }
                        }
                        builder.addDpSlots(slot.build());
                    });
        } catch (BatchRequestBuildException e) {
            throw (InvalidProtocolBufferException) e.getCause();
        }
        return builder.build();
    }

    private EngineRpcService.GenerateInputPB buildInput(BatchItem item)
            throws InvalidProtocolBufferException {
        byte[] bytes = item.ctx().getGenerateInputPbBytes();
        if (bytes == null || bytes.length == 0) {
            throw new IllegalArgumentException("generateInputPbBytes is missing for request " + item.requestId());
        }
        EngineRpcService.GenerateInputPB.Builder input =
                EngineRpcService.GenerateInputPB.parseFrom(bytes).toBuilder();
        if (input.getRequestId() != item.requestId()) {
            throw new IllegalArgumentException("request_id mismatch between schedule request and GenerateInputPB");
        }
        EngineRpcService.GenerateConfigPB.Builder generateConfig = input.getGenerateConfigBuilder();
        generateConfig.clearRoleAddrs();
        addRoleAddr(generateConfig, item.prefill());
        addRoleAddr(generateConfig, item.decode());
        return input.build();
    }

    private void addRoleAddr(EngineRpcService.GenerateConfigPB.Builder generateConfig,
                             org.flexlb.dao.loadbalance.ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        RoleType role = serverStatus.getRole();
        generateConfig.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(role.getCode())
                .setRoleType(RoleTypeProtoConverter.toProto(role))
                .setIp(serverStatus.getServerIp())
                .setHttpPort(serverStatus.getHttpPort())
                .setGrpcPort(serverStatus.getGrpcPort())
                .build());
    }

    // ==================== Dispatch logging ====================

    private void logDispatch(long batchId, List<BatchItem> items, long predMs, String reason) {
        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            BatchItem item = items.get(i);
            long seqLen = item.seqLen();
            long hitCache = item.hitCache();
            totalTokens += seqLen;
            totalHit += hitCache;
            if (i > 0) {
                itemDetail.append(", ");
            }
            itemDetail.append("{req_id=").append(item.requestId())
                    .append(" seq_len=").append(seqLen)
                    .append(" hit_cache=").append(hitCache).append('}');
        }

        BatchItem head = items.get(0);
        long now = System.currentTimeMillis();
        long waitMs = now - head.enqueuedAtMs();

        Logger.info("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} reason={} wait_ms={} "
                        + "prefill={}:{} items=[{}]",
                batchId, items.size(), totalTokens, totalHit, predMs, reason,
                waitMs,
                getIp(), getHttpPort(),
                itemDetail);
    }

    // ==================== Worker status calibration ====================

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getFinishedTaskInfo(), resp.getRunningTaskInfo());
    }

    /**
     * Full calibration against worker status report, driving the two-layer
     * inflight state machine:
     *
     * <ol>
     *   <li>acceptance — a key reported in runningTaskInfo migrates from
     *       layer 1 (inflightEntries) to layer 2 (engineWork) with its
     *       initial phase; already-migrated tasks get phase/lastSeenRound
     *       refreshed. Migration inserts into layer 2 <b>before</b> removing
     *       from layer 1 (conservative order: transient double-count beats
     *       transient under-count).</li>
     *   <li>completion — finished tasks shrink their entry in whichever
     *       layer tracks it (fast path: finished while still in layer 1);
     *       a batch is removed only when all members have finished.</li>
     *   <li>staleness — engineWork entries absent from reports for
     *       {@code flexlbStaleEvictRounds} consecutive rounds are evicted.</li>
     * </ol>
     */
    private void calibrate(Map<String, TaskInfo> finishedTaskInfo, Map<String, TaskInfo> runningTaskInfo) {
        long statusMs = System.currentTimeMillis();
        long round = calibrateRound.incrementAndGet();

        int finishedSize = finishedTaskInfo != null ? finishedTaskInfo.size() : 0;
        int runningSize = runningTaskInfo != null ? runningTaskInfo.size() : 0;
        if (finishedSize > 0 || !inflightEntries.isEmpty() || !engineWork.isEmpty()) {
            logger.info("Prefill calibrate: finishedTasks={}, runningTasks={}, inflightEntries={}, engineWork={}",
                    finishedSize, runningSize, inflightEntries.size(), engineWork.size());
        }

        observeRunningTasks(runningTaskInfo, round, statusMs);
        processFinishedTasks(finishedTaskInfo);
        evictStaleEngineWork(round);
    }

    /**
     * Acceptance step: migrate reported keys layer 1 → layer 2 and refresh
     * phases of already-accepted tasks. Batch phase is aggregated as the
     * minimum across reported members (weakest-link rule).
     */
    private void observeRunningTasks(Map<String, TaskInfo> runningTaskInfo, long round, long statusMs) {
        if (runningTaskInfo == null || runningTaskInfo.isEmpty()) {
            return;
        }
        Map<Long, EngineTaskPhase> phaseByKey = new HashMap<>();
        Map<Long, Set<Long>> reportedRequestIds = new HashMap<>();
        for (TaskInfo task : runningTaskInfo.values()) {
            long key = task.getBatchId() >= 0 ? task.getBatchId() : task.getRequestId();
            phaseByKey.merge(key, EngineTaskPhase.fromPrefill(task.getPhase()), EngineTaskPhase::min);
            reportedRequestIds.computeIfAbsent(key, k -> new HashSet<>()).add(task.getRequestId());
        }

        for (Map.Entry<Long, EngineTaskPhase> observed : phaseByKey.entrySet()) {
            long key = observed.getKey();
            EngineTaskPhase phase = observed.getValue();

            EngineTask<PrefillInflightEntry> existing = engineWork.get(key);
            if (existing != null) {
                existing.observe(phase, round, statusMs);
                continue;
            }

            PrefillInflightEntry entry = inflightEntries.get(key);
            if (entry == null) {
                // Foreign key pre-check: the key is not in layer 1 or layer 2.
                // Check the global InflightStore to distinguish between:
                //   (a) foreign key — requestId belongs to another master (e.g.
                //       multi-master failover where the engine reports tasks from
                //       both masters in the same WorkerStatusResponse). These
                //       should NOT create engineWork entries — they are not ours.
                //   (b) already terminal — the request has finished/cancelled but
                //       the engine still reports it as running (stale report).
                // In both cases we skip. For cross-EP failover (store has it and
                // RUNNING), we still skip for prefill because we cannot reconstruct
                // the PrefillInflightEntry (batch members + predictMs) from the
                // engine report alone — the batch metadata lives only in the
                // scheduler that originally dispatched it.
                if (inflightStore != null && !isForeignKey(key, reportedRequestIds.get(key))) {
                    logger.info("Prefill calibrate: cross-EP failover for key={} requestIds={} — entry lost, skipping (cannot reconstruct batch metadata)",
                            key, reportedRequestIds.get(key));
                } else {
                    logger.debug("Prefill calibrate: running request(s) {} key={} not tracked in either inflight layer (foreign key or terminal), skipping",
                            reportedRequestIds.get(key), key);
                }
                continue;
            }
            // Defense-in-depth: a batch report must carry at least one member
            // requestId of the local batch — otherwise it is a stale or
            // foreign status report reusing the same batchId.
            if (entry instanceof PrefillInflightBatch batch
                    && !ownsAnyRequest(batch, reportedRequestIds.get(key))) {
                logger.warn("Prefill calibrate: running report for batchId={} has no matching requestId in local batch. "
                        + "Likely stale or foreign status report. Skipping migration.", key);
                continue;
            }

            // Migrate: insert into engineWork first, then remove from
            // inflightEntries. Counters follow map membership, so the
            // migration window can only over-count (rejection-biased).
            EngineTask<PrefillInflightEntry> accepted = new EngineTask<>(entry, phase, round, statusMs);
            if (engineWork.putIfAbsent(key, accepted) == null) {
                inflightRequestCount.addAndGet(entry.requestCount());
            }
            PrefillInflightEntry removed = inflightEntries.remove(key);
            if (removed != null) {
                inflightRequestCount.addAndGet(-removed.requestCount());
            }
        }
    }

    private static boolean ownsAnyRequest(PrefillInflightBatch batch, Set<Long> requestIds) {
        for (BatchItem member : batch.requests()) {
            if (requestIds.contains(member.requestId())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Foreign key check: returns {@code true} if ALL reported requestIds for
     * this key are absent from the InflightStore or already in a terminal
     * state (meaning they belong to another master or are finished). Returns
     * {@code false} if at least one requestId is found in the store and
     * still RUNNING (cross-EP failover scenario).
     */
    private boolean isForeignKey(long key, Set<Long> requestIds) {
        if (inflightStore == null || requestIds == null || requestIds.isEmpty()) {
            return true;
        }
        for (Long requestId : requestIds) {
            InflightItem item = inflightStore.get(String.valueOf(requestId));
            if (item != null && !item.state().isTerminal()) {
                return false; // found a non-terminal entry — this is our request
            }
        }
        return true; // all absent or terminal — foreign key
    }

    /**
     * Completion step. Non-batch requests are removed directly by requestId.
     * For batches, finished members (success or failure) are shrunk out of
     * the entry; the entry is removed once every member has finished
     * (partial completion keeps survivors tracked — repack shrink).
     */
    private void processFinishedTasks(Map<String, TaskInfo> finishedTaskInfo) {
        if (finishedTaskInfo == null || finishedTaskInfo.isEmpty()) {
            return;
        }

        Map<Long, List<TaskInfo>> finishedByBatch = new HashMap<>();
        for (TaskInfo task : finishedTaskInfo.values()) {
            if (task.getBatchId() < 0) {
                removeFinishedRequest(task);
            } else {
                finishedByBatch.computeIfAbsent(task.getBatchId(), k -> new ArrayList<>()).add(task);
            }
        }

        for (Map.Entry<Long, List<TaskInfo>> entry : finishedByBatch.entrySet()) {
            finishBatchMembers(entry.getKey(), entry.getValue(), finishedTaskInfo);
        }
    }

    /** Fast/normal path removal of a finished non-batch request (engine reports batch_id=-1). */
    private void removeFinishedRequest(TaskInfo task) {
        long requestId = task.getRequestId();

        EngineTask<PrefillInflightEntry> accepted = engineWork.get(requestId);
        if (accepted != null) {
            if (!(accepted.entry() instanceof PrefillInflightRequest)) {
                logger.warn("Prefill calibrate: finished non-batch reqId={} collides with a tracked batch key, skipping",
                        requestId);
                return;
            }
            if (engineWork.remove(requestId, accepted)) {
                inflightRequestCount.addAndGet(-accepted.entry().requestCount());
            }
            return;
        }

        // Cross-round fast path: finished before ever being observed running
        PrefillInflightEntry entry = inflightEntries.get(requestId);
        if (entry == null) {
            logger.warn("Prefill calibrate: finished non-batch request reqId={} not in inflight", requestId);
            return;
        }
        if (!(entry instanceof PrefillInflightRequest)) {
            logger.warn("Prefill calibrate: finished non-batch reqId={} collides with a tracked batch key, skipping",
                    requestId);
            return;
        }
        if (inflightEntries.remove(requestId, entry)) {
            inflightRequestCount.addAndGet(-entry.requestCount());
        }
    }

    /**
     * Apply this round's finished tasks of one batch to whichever layer
     * tracks it. Foreign requestIds (stale report reusing the batchId) are
     * ignored with a warning.
     */
    private void finishBatchMembers(long batchId, List<TaskInfo> finishedTasks,
                                    Map<String, TaskInfo> finishedTaskInfo) {
        EngineTask<PrefillInflightEntry> accepted = engineWork.get(batchId);
        PrefillInflightEntry tracked = accepted != null ? accepted.entry() : inflightEntries.get(batchId);
        if (tracked == null) {
            logger.debug("Prefill calibrate: finished batchId={} not tracked (already released?)", batchId);
            return;
        }
        if (!(tracked instanceof PrefillInflightBatch batch)) {
            logger.warn("Prefill calibrate: finished batchId={} collides with a tracked non-batch key, skipping",
                    batchId);
            return;
        }

        Set<Long> localRequestIds = batch.requests().stream()
                .map(BatchItem::requestId)
                .collect(Collectors.toSet());
        Set<Long> localFinishedIds = new HashSet<>();
        boolean anySuccess = false;
        int foreignCount = 0;
        for (TaskInfo task : finishedTasks) {
            if (!localRequestIds.contains(task.getRequestId())) {
                foreignCount++;
                continue;
            }
            localFinishedIds.add(task.getRequestId());
            if (task.getErrorCode() == 0) {
                anySuccess = true;
            } else {
                logger.warn("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                        batchId, task.getRequestId(), task.getErrorMessage());
            }
        }
        if (foreignCount > 0) {
            logger.warn("Prefill calibrate: batchId={} has {} finished tasks with foreign requestIds. "
                    + "Likely stale or foreign status report. Ignoring them.", batchId, foreignCount);
        }
        if (localFinishedIds.isEmpty()) {
            return;
        }

        if (localFinishedIds.containsAll(localRequestIds)) {
            // Every member finished — remove the whole entry from its layer
            releaseBatch(batchId);
            if (anySuccess) {
                reportBatchCompletion(batchId, batch, finishedTaskInfo);
            }
        } else {
            // Partial completion — shrink survivors, keep tracking
            repackBatch(batchId, localFinishedIds);
        }
    }

    /**
     * Staleness step: evict engineWork entries that have been absent from both
     * running and finished reports for {@code flexlbStaleEvictRounds}
     * consecutive calibrate rounds (lost completion report).
     */
    private void evictStaleEngineWork(long round) {
        List<InflightItem> toTerminate = new ArrayList<>();
        for (Map.Entry<Long, EngineTask<PrefillInflightEntry>> entry : engineWork.entrySet()) {
            EngineTask<PrefillInflightEntry> task = entry.getValue();
            if (round - task.lastSeenRound() < config.getFlexlbStaleEvictRounds()) {
                continue;
            }
            if (engineWork.remove(entry.getKey(), task)) {
                inflightRequestCount.addAndGet(-task.entry().requestCount());
                logger.warn("Prefill calibrate: engineWork key={} phase={} unseen for {} rounds, evicting as stale",
                        entry.getKey(), task.phase(), round - task.lastSeenRound());
                // A3: STALE eviction now drives the bound InflightItem to a
                // terminal state so the client future is settled in seconds,
                // not the 300s TTL safety net.
                collectEntryItems(task.entry(), toTerminate);
            }
        }
        for (InflightItem item : toTerminate) {
            if (!item.isTerminated()) {
                item.complete(Response.error(StrategyErrorType.WORKER_EXECUTION_FAILED,
                        "engine evicted as stale"), InflightState.FAILED);
            }
        }
    }

    // ==================== 新三视图（显式接口） ====================

    /** Layer-1 entry count: dispatched, not yet acknowledged by the engine. */
    public int prefillInflightCount() {
        return inflightEntries.size();
    }

    /** Layer-2 task count: entries the engine has acknowledged. */
    public int prefillEngineWorkCount() {
        return engineWork.size();
    }

    /** Layer-2 tasks currently in the WAITING phase. */
    public int prefillEngineWaitingCount() {
        return countEngineWorkInPhase(EngineTaskPhase.WAITING);
    }

    /** Layer-2 tasks currently in the RUNNING phase. */
    public int prefillEngineRunningCount() {
        return countEngineWorkInPhase(EngineTaskPhase.RUNNING);
    }

    private int countEngineWorkInPhase(EngineTaskPhase phase) {
        int count = 0;
        for (EngineTask<PrefillInflightEntry> task : engineWork.values()) {
            if (task.phase() == phase) {
                count++;
            }
        }
        return count;
    }

    /**
     * Two-layer weighted wait-time estimate:
     * {@code Σ(inflightEntries.predictMs) + Σ(engineWork[WAITING].predictMs)
     * + running remainder}, where a RUNNING task contributes
     * {@code max(predictMs - elapsed since it started running, 0)}.
     *
     * <p>Unlike the legacy single-pool formula, this distinguishes
     * not-yet-accepted / queued / running segments and only discounts
     * elapsed time from running tasks.
     */
    public long prefillEstimatedWaitTimeMs() {
        long nowMs = System.currentTimeMillis();
        long totalMs = 0;
        for (PrefillInflightEntry entry : inflightEntries.values()) {
            totalMs += Math.max(0, entry.predictMs());
        }
        for (EngineTask<PrefillInflightEntry> task : engineWork.values()) {
            long predictMs = Math.max(0, task.entry().predictMs());
            if (task.running()) {
                long elapsedMs = Math.max(0, nowMs - task.progressBaseMs());
                totalMs += Math.max(0, predictMs - elapsedMs);
            } else {
                totalMs += predictMs;
            }
        }
        return totalMs;
    }

    /** Batcher pending-dispatch queue depth (unified naming for the existing view). */
    public int prefillBatcherQueueSize() {
        return batcher.queueSize();
    }

    /**
     * Request-level pending count: total requests the engine will face.
     * Includes master-tracked inflight requests (both layers) + batcher
     * queue + engine-accepted tasks in the WAITING phase (queued on the
     * engine side but not yet running).
     */
    public long prefillPendingRequestCount() {
        return inflightRequestCount.get() + batcher.queueSize()
                + countEngineWorkInPhase(EngineTaskPhase.WAITING);
    }

    /**
     * Evict entries older than {@code ttlMs} from both inflight layers.
     * Called periodically by {@code EndpointRegistry.scheduledEviction()} with
     * the EP-level TTL ({@code flexlbEpInflightTtlMs}, default 600s).
     *
     * <p>TTL layering design:
     * <ul>
     *   <li><b>Scheduler-level</b> ({@code flexlbInflightTtlMs}, default 300s):
     *       used by {@link InflightStore} for RUNNING item timeout — the safety
     *       net for requests that never reached a terminal state (lost ACK).</li>
     *   <li><b>EP-level</b> ({@code flexlbEpInflightTtlMs}, default 600s): used
     *       here for both inflight layers — longer than scheduler-level because
     *       engine-accepted tasks legitimately run longer (decode generation)
     *       and should not be prematurely evicted by the wall-clock backstop.</li>
     *   <li><b>Tombstone</b> ({@code flexlbTombstoneTtlMs}, default 60s): used by
     *       {@link InflightStore} for terminal item cleanup — short because
     *       tombstones only exist to reject duplicate cancels.</li>
     * </ul>
     * The EP-level TTL is the backstop for a worker that stops reporting
     * entirely, where stale-round eviction cannot fire (calibrate rounds
     * no longer advance without status reports).
     *
     * @return number of entries evicted
     */
    public int evictExpiredBatches(long ttlMs) {
        return inflightEvictor.evictExpired(ttlMs) + engineWorkEvictor.evictExpired(ttlMs);
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@code RouteService#triggerSchedulerMetrics()}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        int queueSize = prefillBatcherQueueSize();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(),
                prefillInflightCount() + prefillEngineWorkCount());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), inflightRequestCount.get());
        // Two-layer breakdown
        reporter.reportPrefillInflightEntriesCount(RoleType.PREFILL.name(), getIp(), prefillInflightCount());
        reporter.reportPrefillEngineWorkCount(RoleType.PREFILL.name(), getIp(), prefillEngineWorkCount());
    }

    /**
     * On batch completion, compare the formula-predicted execution time against the
     * engine-reported actual execution time (max across the batch's finished tasks),
     * then log and emit prediction-accuracy metrics.
     */
    private void reportBatchCompletion(long batchId, PrefillInflightBatch batch, Map<String, TaskInfo> finishedTaskInfo) {
        logger.debug("run reportBatchCompletion, batchId: {}, finishedTaskInfo size: {}",
                batchId, finishedTaskInfo.size());
        long actualMs = -1;
        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task.getBatchId() == batchId && task.getExecutionTimeMs() > 0) {
                    actualMs = Math.max(actualMs, task.getExecutionTimeMs());
                }
            }
        }
        if (actualMs < 0) {
            logger.debug("actualMs < 0: {}", actualMs);
            return;
        }

        long predictedMs = batch.predictMs();
        long gapMs = actualMs - predictedMs;
        Logger.info(
                "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs, batch.requests().size(), getIp());

        // Feed the actual-vs-predicted timing back into the predictor for future learning.
        predictor.learn(batch.requests(), predictedMs, actualMs);

        reporter.reportBatchPredictedTimeMs(RoleType.PREFILL.name(), getIp(), predictedMs);
        reporter.reportBatchActualTimeMs(RoleType.PREFILL.name(), getIp(), actualMs);
        reporter.reportBatchPredictGapMs(RoleType.PREFILL.name(), getIp(), gapMs);
    }

    // ==================== Internal exception wrapper ====================

    /**
     * Wraps checked {@link InvalidProtocolBufferException} to propagate through
     * stream lambdas in {@link #buildBatchRequest}.
     */
    private static final class BatchRequestBuildException extends RuntimeException {
        private BatchRequestBuildException(Throwable cause) {
            super(cause);
        }
    }
}
