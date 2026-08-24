package org.flexlb.balance.endpoint;

import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import org.flexlb.balance.scheduler.BatchIdGenerator;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DispatchMeta;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.FormulaPredictor;
import org.flexlb.balance.strategy.LearningPredictor;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.state.PrefillEndpointCounters;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.CompletionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.IntSupplier;

/**
 * Prefill worker endpoint.
 *
 * <p>Owns the batch dispatch pipeline for its worker: batching (via the
 * embedded {@link WorkerBatcher}), batch prediction and commit into the
 * state ledger ({@link #commitBatch} / {@link #commitRequest} — the
 * per-request ledger attach points), and asynchronous gRPC dispatch to the
 * engine ({@link #submitBatch}). Per-item results are settled directly
 * through {@link BatchItem} terminal transitions — no scheduler callback is
 * involved.
 *
 * <p>Legacy two-layer inflight maps (layer-1 inflight entries + layer-2
 * engine tasks) and the calibrate state machine have been removed: the
 * ledger's prefill side is the single accounting source — engine-reported
 * state is the single source of truth, with the LedgerJanitor providing the
 * stale/TTL safety net. All scheduling read points are served by the
 * per-EP counter cache refreshed on each engine status tick
 * ({@link #onWorkerStatusUpdate}); in ledger-disabled (degraded) mode all
 * read points return zero — no accounting source without the ledger.
 */
public class PrefillEndpoint extends WorkerEndpoint {

    private final FlexlbConfig config;
    private final EngineGrpcClient grpcClient;
    private final BatchDispatchExecutor dispatchExecutor;
    private final BatchIdGenerator batchIdGenerator;
    private final IntSupplier globalActiveCount;
    private final PrefillTimePredictor predictor;
    private final WorkerBatcher batcher;
    private final BatchSchedulerReporter reporter;

    /** 状态账本门面（装配点注入；null / 关态时读点全零——退化模式）。 */
    private final StateShadowBridge shadowBridge;

    /** 端点稳定 ID（ipPort 哈希——与账本 translator 的 endpointId 同映射）。 */
    private final int endpointId;

    /**
     * per-EP 账本计数缓存：引擎状态报文 tick 刷新，读点零锁 volatile
     * 读——策略 select 热路径不触发按需聚合。关态恒 null（读数全零）。
     */
    private volatile PrefillEndpointCounters ledgerCounters;

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           EngineGrpcClient grpcClient,
                           BatchDispatchExecutor dispatchExecutor,
                           BatchIdGenerator batchIdGenerator,
                           IntSupplier globalActiveCount,
                           BatchSchedulerReporter reporter,
                           StateShadowBridge shadowBridge) {
        super(status);
        this.config = config;
        this.grpcClient = grpcClient;
        this.dispatchExecutor = dispatchExecutor;
        this.batchIdGenerator = batchIdGenerator;
        this.globalActiveCount = globalActiveCount;
        this.reporter = reporter;
        this.shadowBridge = shadowBridge;
        this.endpointId = ipPort() != null ? ipPort().hashCode() : 0;
        this.predictor = createPredictor(config);
        this.batcher = new WorkerBatcher(status.getIpPort(), this, config, reporter);
        this.batcher.start();
    }

    /** per-EP 账本计数（未刷新/退化模式时全零——读数退化方向为低估，不阻断调度）。 */
    private PrefillEndpointCounters ledgerCountersOrZero() {
        PrefillEndpointCounters c = ledgerCounters;
        return c != null ? c : PrefillEndpointCounters.empty();
    }

    public WorkerBatcher getBatcher() {
        return batcher;
    }

    @Override
    public void close() {
        try {
            batcher.shutdown();
        } finally {
            super.close();
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

    // ==================== Ledger attach points (dispatch commit) ====================

    /**
     * Commit a dispatched batch into the state ledger — one entry per member
     * request with the batch foreign key and the per-request share of the
     * batch prediction (data source of the wait-estimate read point).
     */
    public void commitBatch(long batchId, long predictMs, List<BatchItem> requests) {
        // 账本挂点：P 条目派发提交——onDispatching（批次外键）+ 分摊批次预测
        // 耗时（等待估算读点数据源）+ onDispatched（绑定端点世代）。门面为
        // null（非 P/D 分离角色不进账本）或关态时短路——退化模式无记账源。
        if (shadowBridge != null && shadowBridge.isEnabled()) {
            long shareMs = requests.isEmpty() ? 0L : predictMs / requests.size();
            for (BatchItem item : requests) {
                shadowBridge.onPrefillDispatched(item.requestId(), batchId, ipPort(), shareMs);
            }
        }
    }

    /**
     * Commit a single directly-dispatched request (non-batch path:
     * CostBased / ShortestTTFT) into the state ledger, keyed by requestId
     * (the engine reports these with {@code batch_id=-1}).
     */
    public void commitRequest(long requestId, long predictMs) {
        // 散请求提交挂点（batchId=-1，分摊即单请求预测）。
        if (shadowBridge != null && shadowBridge.isEnabled()) {
            shadowBridge.onPrefillDispatched(requestId, -1L, ipPort(), predictMs);
        }
    }

    // ==================== Batch submission (sync part) ====================

    /**
     * EP-level batch submission — the full dispatch function lives on the EP.
     *
     * <p>Synchronous part: filter already-finished items, assign a batch ID,
     * run DP-aware prediction, commit the batch into the state ledger, then
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
        // DP ranks), commit into the state ledger
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
            failItems(items, "Batch request build failed: " + e.getMessage());
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
                                for (BatchItem item : items) {
                                    item.failTimeout(cause);
                                }
                            } else {
                                failItems(items, "gRPC dispatch failed: " + cause.getMessage());
                            }
                        } else if (response == null) {
                            failItems(items, "EnqueueBatch returned null response");
                        } else {
                            handleResponse(batchId, items, response);
                        }
                    } catch (Throwable t) {
                        // Safety net: ensure every item is settled even for unexpected errors
                        Logger.error("Unexpected error in EnqueueBatch callback batchId={}", batchId, t);
                        failItems(items, "Unexpected callback error: " + t.getMessage());
                    }
                }, dispatchExecutor);
    }

    private void failItems(List<BatchItem> items, String message) {
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

    // ==================== Engine status tick ====================

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        // per-EP 账本计数缓存随引擎状态 tick 刷新（引擎上报 = 唯一事实源，
        // 本端点不再维护镜像条目——终局与陈旧清理由 ledger 事件泵与 janitor 承担）。
        if (shadowBridge != null) {
            this.ledgerCounters = shadowBridge.prefillEndpointCounters(endpointId);
        }
    }

    // ==================== 调度读点（账本 per-EP 视图） ====================

    /**
     * 该端点已派发未终局的活跃条目数（账本 per-EP 视图；请求级口径——
     * 批次成员各计 1）。退化模式（账本关）无计数源，返回 0。
     */
    public int prefillActiveRequestCount() {
        return ledgerCountersOrZero().activeTotal();
    }

    /** 引擎已接管（引擎上报观察）条目数。 */
    public int prefillEngineOwnedCount() {
        return ledgerCountersOrZero().engineOwnedCount();
    }

    /** 引擎等待中条目数（KV 未装载 + 已装载待执行；账本相位人口读点）。 */
    public long prefillEngineWaitingCount() {
        return ledgerCountersOrZero().engineWaitingCount();
    }

    /** 引擎执行中条目数（prefill 迭代执行；账本相位人口读点）。 */
    public long prefillEngineRunningCount() {
        return ledgerCountersOrZero().engineRunningCount();
    }

    /**
     * Wait-time estimate for a newly enqueued request: the ledger's
     * per-EP Σ shared batch-prediction over active entries. Entries
     * currently executing on the engine are not discounted (conservative
     * over-estimate, rejection-biased). Degraded mode returns 0.
     */
    public long prefillEstimatedWaitTimeMs() {
        return ledgerCountersOrZero().estimatedWaitMs();
    }

    /** Batcher pending-dispatch queue depth (unified naming for the existing view). */
    public int prefillBatcherQueueSize() {
        return batcher.queueSize();
    }

    /**
     * Request-level pending count: total requests the engine will face —
     * 账本 per-EP 已派发未终局条目数（含引擎侧排队窗口）+ batcher 队列
     * 深度（排队/攒批窗口）。退化模式只剩 batcher 队列深度。
     */
    public long prefillPendingRequestCount() {
        return ledgerCountersOrZero().activeTotal() + batcher.queueSize();
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
        PrefillEndpointCounters c = ledgerCountersOrZero();
        int queueSize = prefillBatcherQueueSize();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(), c.activeTotal());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), c.activeTotal());
        // 账本口径分解：引擎已接管条目 vs 已派发未被引擎接管的未确认窗口
        reporter.reportPrefillInflightEntriesCount(RoleType.PREFILL.name(), getIp(),
                Math.max(0, c.activeTotal() - c.engineOwnedCount()));
        reporter.reportPrefillEngineWorkCount(RoleType.PREFILL.name(), getIp(), c.engineOwnedCount());
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
