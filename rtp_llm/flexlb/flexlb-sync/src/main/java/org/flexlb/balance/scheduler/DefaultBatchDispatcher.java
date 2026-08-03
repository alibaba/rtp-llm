package org.flexlb.balance.scheduler;

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

    public DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService,
                                  @Autowired(required = false) MeterRegistry meterRegistry) {
        this.grpcClient = grpcClient;
        this.configService = configService;
        this.meterRegistry = meterRegistry;
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
        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, items);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failItems(items, prefillEp, batchId, "Batch request build failed: " + e.getMessage(), callback);
            return;
        }

        // 2. Log dispatch
        logDispatch(batchId, items, prefillEp, predMs, reason);

        // 3. Send gRPC (async)
        long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
        grpcClient.batchEnqueueAsync(prefillEp.getIp(), prefillEp.getGrpcPort(), request, deadlineMs)
                .whenCompleteAsync((response, ex) -> {
                    try {
                        if (ex != null) {
                            Throwable cause = ex instanceof CompletionException ? ex.getCause() : ex;
                            Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                                    batchId, prefillEp.getIp(), prefillEp.getGrpcPort(), cause.getMessage());
                            if (Status.fromThrowable(cause).getCode() == Status.Code.DEADLINE_EXCEEDED) {
                                prefillEp.releaseBatch(batchId);
                                for (BatchItem item : items) {
                                    callback.onTimeout(item, cause);
                                }
                            } else {
                                failItems(items, prefillEp, batchId,
                                        "gRPC dispatch failed: " + cause.getMessage(), callback);
                            }
                        } else if (response == null) {
                            failItems(items, prefillEp, batchId, "EnqueueBatch returned null response", callback);
                        } else {
                            handleResponse(batchId, items, response, callback);
                        }
                    } catch (Throwable t) {
                        // Safety net: ensure callbacks are always invoked even for unexpected errors
                        Logger.error("Unexpected error in EnqueueBatch callback batchId={}", batchId, t);
                        failItems(items, prefillEp, batchId,
                                "Unexpected callback error: " + t.getMessage(), callback);
                    }
                }, dispatchExecutor);
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
        EngineRpcService.GenerateConfigPB.Builder config = input.getGenerateConfigBuilder();
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
