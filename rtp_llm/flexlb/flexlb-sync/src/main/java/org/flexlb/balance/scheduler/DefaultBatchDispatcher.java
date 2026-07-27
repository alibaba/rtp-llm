package org.flexlb.balance.scheduler;

import com.google.protobuf.Int64Value;
import com.google.protobuf.InvalidProtocolBufferException;
import io.grpc.Status;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Default implementation of {@link BatchDispatcher}.
 * <p>
 * Owns its own thread pool for asynchronous gRPC dispatch.
 * Handles the full pipeline: build request → send → parse response → callback.
 * Does NOT manage inflight state — results are reported via {@link DispatchCallback}.
 */
@Component
public class DefaultBatchDispatcher implements BatchDispatcher {

    private static final String METRIC_PREFIX = "flexlb.";

    private final EngineGrpcClient grpcClient;
    private final ConfigService configService;
    private final ThreadPoolExecutor dispatchExecutor;
    private final MeterRegistry meterRegistry;
    private final BatchSchedulerReporter reporter;

    public DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
                                  @Autowired(required = false) MeterRegistry meterRegistry,
                                  @Autowired(required = false) BatchSchedulerReporter reporter) {
        this.grpcClient = grpcClient;
        this.configService = configService;
        this.meterRegistry = meterRegistry;
        this.reporter = reporter;
        int poolSize = configService.loadBalanceConfig().getFlexlbBatchDispatchPoolSize();
        int queueSize = configService.loadBalanceConfig().getFlexlbBatchDispatchQueueSize();
        Logger.info("FlexLB dispatch executor config: poolSize={}, queueSize={}, threadFactory=flexlb-dispatch-executor, rejectionPolicy=AbortPolicy",
                poolSize, queueSize);
        this.dispatchExecutor = new ThreadPoolExecutor(
                poolSize, poolSize,
                60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(queueSize),
                new NamedThreadFactory("flexlb-dispatch-executor"),
                new ThreadPoolExecutor.AbortPolicy());
        registerMetrics();
    }

    /**
     * Register Micrometer gauges and function counters for the dispatch executor.
     *
     * <p>Metrics exposed:
     * <ul>
     *   <li>{@code flexlb_dispatch_executor_active_threads} — gauge: active thread count</li>
     *   <li>{@code flexlb_dispatch_executor_queue_size} — gauge: pending task queue length</li>
     *   <li>{@code flexlb_dispatch_executor_pool_size} — gauge: current thread pool size</li>
     *   <li>{@code flexlb_dispatch_executor_completed_tasks_total} — counter: completed task count</li>
     * </ul>
     *
     * <p>When {@link MeterRegistry} is not available, metric registration is silently skipped.
     */
    private void registerMetrics() {
        if (meterRegistry == null) {
            Logger.info("MeterRegistry not available, skipping dispatch executor metrics");
            return;
        }

        Gauge.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS,
                        dispatchExecutor, ThreadPoolExecutor::getActiveCount)
                .description("Dispatch executor active thread count")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE,
                        dispatchExecutor, exec -> exec.getQueue().size())
                .description("Dispatch executor pending task queue size")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE,
                        dispatchExecutor, ThreadPoolExecutor::getPoolSize)
                .description("Dispatch executor current pool size")
                .register(meterRegistry);

        FunctionCounter.builder(METRIC_PREFIX + MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS,
                        dispatchExecutor, ThreadPoolExecutor::getCompletedTaskCount)
                .description("Dispatch executor total completed tasks")
                .register(meterRegistry);

        Logger.info("FlexLB dispatch executor metrics registered with MeterRegistry");
    }

    @Override
    public void dispatch(List<BatchItem> items, PrefillEndpoint prefillEp,
                         long batchId, long predMs, String reason, DispatchCallback callback) {
        try {
            dispatchExecutor.execute(() -> doDispatch(items, prefillEp, batchId, predMs, reason, callback));
        } catch (RejectedExecutionException e) {
            Logger.warn("FlexLB batch dispatch rejected (executor shutdown), failing {} items", items.size());
            prefillEp.releaseBatch(batchId);
            for (BatchItem item : items) {
                callback.onFailure(item, e);
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        dispatchExecutor.shutdownNow();
    }

    // ==================== Internal: dispatch pipeline (runs on executor thread) ====================

    private void doDispatch(List<BatchItem> items, PrefillEndpoint prefillEp,
                            long batchId, long predMs, String reason, DispatchCallback callback) {
        try {
            doDispatchInternal(items, prefillEp, batchId, predMs, reason, callback);
        } catch (Throwable t) {
            // Safety net: ensure callbacks are always invoked even for unexpected errors
            Logger.error("Unexpected error in doDispatch batchId={}", batchId, t);
            // Release the batch first to avoid leaking inflight state on the
            // unexpected-error path (releaseBatch is idempotent, safe to call).
            try {
                prefillEp.releaseBatch(batchId);
            } catch (Throwable ignored) {
                // best-effort
            }
            for (BatchItem item : items) {
                try {
                    callback.onFailure(item, t);
                } catch (Throwable ignored) {
                    // best-effort
                }
            }
        }
    }

    private void doDispatchInternal(List<BatchItem> items, PrefillEndpoint prefillEp,
                                    long batchId, long predMs, String reason, DispatchCallback callback) {
        // Filter out items that reached a terminal state before dispatch
        List<BatchItem> active = new ArrayList<>();
        for (BatchItem item : items) {
            if (!item.future().isDone()) {
                active.add(item);
            } else {
                Logger.debug("Skipping completed item in dispatch: request_id={}, batch_id={}",
                        item.requestId(), batchId);
                callback.onFailure(item, new IllegalStateException("request completed before EnqueueBatch was sent"));
            }
        }

        if (active.isEmpty()) {
            Logger.debug("All items completed before dispatch, batch_id={}", batchId);
            prefillEp.releaseBatch(batchId);
            return;
        }

        // Per-item expiry check: drop items whose absolute deadline already passed
        // so they never reach the engine (downstream would otherwise treat a
        // clamped timeout as "unset" and fall back to the default timeout).
        long now = nowMs();
        List<BatchItem> valid = new ArrayList<>(active.size());
        List<BatchItem> expired = new ArrayList<>();
        for (BatchItem item : active) {
            if (item.absoluteDeadlineMs() > 0 && now >= item.absoluteDeadlineMs()) {
                expired.add(item);
            } else {
                valid.add(item);
            }
        }
        for (BatchItem item : expired) {
            long overdueMs = now - item.absoluteDeadlineMs();
            Logger.warn("Dropping expired item before dispatch: batch_id={}, request_id={}, overdue_ms={}",
                    batchId, item.requestId(), overdueMs);
            try {
                callback.onTimeout(item, new RuntimeException(
                        "Item absolute deadline exceeded before dispatch, requestId=" + item.requestId()
                                + ", overdueMs=" + overdueMs));
            } catch (Throwable t) {
                // A misbehaving callback must not break the remaining items.
                Logger.error("onTimeout callback failed for request_id={}, batch_id={}",
                        item.requestId(), batchId, t);
                try {
                    callback.onFailure(item, t);
                } catch (Throwable ignored) {
                    // best-effort
                }
            }
        }
        if (!expired.isEmpty()) {
            // Counts items already expired at the per-item check above. This set is
            // disjoint from the batch-level fallback report below (which only covers
            // items expiring after this check), so no item is ever counted twice.
            reportDispatchExpired(prefillEp, expired.size());
        }
        if (valid.isEmpty()) {
            // onTimeout may have already repacked/emptied this batch; releaseBatch is idempotent.
            Logger.warn("All items expired before dispatch, batch_id={}, expired_count={}",
                    batchId, expired.size());
            prefillEp.releaseBatch(batchId);
            return;
        }

        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, valid);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failItems(valid, prefillEp, batchId, "Batch request build failed: " + e.getMessage(), callback);
            return;
        }

        // 2. Log dispatch
        logDispatch(batchId, valid, prefillEp, predMs, reason);

        // 3. Compute EnqueueBatch deadline from absolute_deadline_ms if set.
        // Uses the minimum absolute deadline across all items in the batch.
        long configDeadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
        long minAbsoluteDeadline = Long.MAX_VALUE;
        for (BatchItem item : valid) {
            if (item.absoluteDeadlineMs() > 0) {
                minAbsoluteDeadline = Math.min(minAbsoluteDeadline, item.absoluteDeadlineMs());
            }
        }

        long deadlineMs;
        if (minAbsoluteDeadline != Long.MAX_VALUE) {
            long remaining = minAbsoluteDeadline - nowMs();
            if (remaining <= 0) {
                // Absolute deadline already passed — don't dispatch, mark as timed out.
                // Race fallback: only reachable for items that were still valid at the
                // per-item expiry check but crossed their deadline before this point.
                Logger.warn("EnqueueBatch skipped: absolute deadline already passed, "
                        + "batchId={}, minAbsoluteDeadline={}, now={}",
                        batchId, minAbsoluteDeadline, nowMs());
                // Counts only items that expired between the per-item check and this
                // deadline computation — disjoint from the per-item report above,
                // so no item is ever counted twice.
                reportDispatchExpired(prefillEp, valid.size());
                prefillEp.releaseBatch(batchId);
                RuntimeException timeoutError = new RuntimeException(
                        "EnqueueBatch deadline already exceeded (absolute deadline passed)");
                for (BatchItem item : valid) {
                    callback.onTimeout(item, timeoutError);
                }
                return;
            }
            deadlineMs = Math.min(remaining, configDeadlineMs);
        } else {
            // Fallback: absolute_deadline_ms not set, use original config deadline
            deadlineMs = configDeadlineMs;
        }

        // 4. Send gRPC (async)
        grpcClient.batchEnqueueAsync(prefillEp.getIp(), prefillEp.getGrpcPort(), request, deadlineMs)
                .whenCompleteAsync((response, ex) -> {
                    try {
                        if (ex != null) {
                            Throwable cause = ex instanceof CompletionException ? ex.getCause() : ex;
                            Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                                    batchId, prefillEp.getIp(), prefillEp.getGrpcPort(), cause.getMessage());
                            if (Status.fromThrowable(cause).getCode() == Status.Code.DEADLINE_EXCEEDED) {
                                prefillEp.releaseBatch(batchId);
                                for (BatchItem item : valid) {
                                    callback.onTimeout(item, cause);
                                }
                            } else {
                                failItems(valid, prefillEp, batchId,
                                        "gRPC dispatch failed: " + cause.getMessage(), callback);
                            }
                        } else if (response == null) {
                            failItems(valid, prefillEp, batchId, "EnqueueBatch returned null response", callback);
                        } else {
                            handleResponse(batchId, valid, response, callback);
                        }
                    } catch (Throwable t) {
                        // Safety net: ensure callbacks are always invoked even for unexpected errors
                        Logger.error("Unexpected error in EnqueueBatch callback batchId={}", batchId, t);
                        failItems(valid, prefillEp, batchId,
                                "Unexpected callback error: " + t.getMessage(), callback);
                    }
                }, dispatchExecutor);
    }

    private void reportDispatchExpired(PrefillEndpoint prefillEp, int count) {
        if (reporter != null) {
            reporter.reportDispatchExpired(RoleType.PREFILL.name(), prefillEp.ipPort(), count);
        }
    }

    /** Overridable time source for the expiry checks (package-private for deterministic tests). */
    long nowMs() {
        return System.currentTimeMillis();
    }

    private void failItems(List<BatchItem> items, PrefillEndpoint prefillEp,
                           long batchId, String message, DispatchCallback callback) {
        prefillEp.releaseBatch(batchId);
        RuntimeException error = new RuntimeException(message);
        for (BatchItem item : items) {
            callback.onFailure(item, error);
        }
    }

    // ==================== Response parsing ====================

    private void handleResponse(long batchId, List<BatchItem> items,
                                EngineRpcService.EnqueueBatchResponsePB response,
                                DispatchCallback callback) {
        if (response.getBatchId() != batchId) {
            RuntimeException mismatch = new RuntimeException(
                    "EnqueueBatch batch_id mismatch: expected " + batchId
                            + " but got " + response.getBatchId());
            for (BatchItem item : items) {
                callback.onFailure(item, mismatch);
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
                callback.onSuccess(item, batchId);
            } else if (errorByRequestId.containsKey(item.requestId())) {
                EngineRpcService.EnqueueBatchErrorPB error = errorByRequestId.get(item.requestId());
                String errorMessage = error.hasErrorInfo()
                        ? error.getErrorInfo().getErrorMessage()
                        : "missing error_info";
                callback.onFailure(item, new RuntimeException(
                        "EnqueueBatch rejected request " + item.requestId() + ": " + errorMessage));
            } else {
                callback.onFailure(item, new RuntimeException(
                        "EnqueueBatch missing ack for request " + item.requestId()));
            }
        }
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
                        int groupSize = entry.getValue().size();
                        for (BatchItem item : entry.getValue()) {
                            try {
                                slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                                        .setInput(buildInput(batchId, groupSize, item))
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

    private EngineRpcService.GenerateInputPB buildInput(long batchId, int groupSize,
                                                        BatchItem item)
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
        input.setGroupId(Int64Value.of(batchId));
        input.setGroupSize(groupSize);

        EngineRpcService.GenerateConfigPB.Builder config = input.getGenerateConfigBuilder();
        // Rewrite timeout_ms to the dispatch-time remaining budget so the engine
        // side (gRPC deadline, CHECK_REQUEST_TIMEOUT, etc.) automatically respects
        // the end-to-end deadline.  When absoluteDeadlineMs is set (>0), compute
        // remaining = absoluteDeadlineMs - now; otherwise keep the original timeout_ms.
        if (item.absoluteDeadlineMs() > 0) {
            long remaining = item.absoluteDeadlineMs() - System.currentTimeMillis();
            // Clamp to [1, Integer.MAX_VALUE]: floor of 1ms ensures the downstream
            // CHECK_REQUEST_TIMEOUT fires immediately instead of treating 0 as
            // "unset" and falling back to the default timeout; the upper bound
            // avoids int overflow on cast.
            config.setTimeoutMs((int) Math.min(Math.max(1, remaining), Integer.MAX_VALUE));
        }
        config.clearRoleAddrs();
        addRoleAddr(config, item.prefill());
        addRoleAddr(config, item.decode());
        return input.build();
    }

    private void addRoleAddr(EngineRpcService.GenerateConfigPB.Builder config, ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        RoleType role = serverStatus.getRole();
        config.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(role.getCode())
                .setRoleType(RoleTypeProtoConverter.toProto(role))
                .setIp(serverStatus.getServerIp())
                .setHttpPort(serverStatus.getHttpPort())
                .setGrpcPort(serverStatus.getGrpcPort())
                .build());
    }

    // ==================== Logging ====================

    private void logDispatch(long batchId, List<BatchItem> items,
                             PrefillEndpoint prefillEp, long predMs, String reason) {
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
        long budgetMs = head.sortKey() - now;

        Logger.info("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} reason={} wait_ms={} budget_ms={} "
                        + "prefill={}:{} items=[{}]",
                batchId, items.size(), totalTokens, totalHit, predMs, reason,
                waitMs, budgetMs,
                prefillEp.getIp(), prefillEp.getHttpPort(),
                itemDetail);
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
