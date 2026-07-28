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
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantReadWriteLock;

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
    private final Map<Long, Runnable> pendingFailures = new ConcurrentHashMap<>();
    private final Set<Long> rpcStarted = ConcurrentHashMap.newKeySet();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final ReentrantReadWriteLock sendGate = new ReentrantReadWriteLock();

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
        Runnable shutdownFailure = () -> failItems(items, prefillEp, batchId,
                new CancellationException("FlexLB dispatcher stopped"), callback);
        if (pendingFailures.putIfAbsent(batchId, shutdownFailure) != null) {
            IllegalStateException duplicate = new IllegalStateException(
                    "duplicate pending FlexLB batch_id=" + batchId);
            for (BatchItem item : items) {
                safeOnFailure(callback, item, duplicate, batchId);
            }
            return;
        }
        if (closed.get()) {
            failPending(batchId, shutdownFailure, items, prefillEp,
                    new CancellationException("FlexLB dispatcher stopped"), callback);
            return;
        }
        try {
            dispatchExecutor.execute(() -> doDispatch(
                    items, prefillEp, batchId, predMs, reason, callback, shutdownFailure));
        } catch (RejectedExecutionException e) {
            Logger.warn("FlexLB batch dispatch rejected (executor shutdown), failing {} items", items.size());
            failPending(batchId, shutdownFailure, items, prefillEp, e, callback);
        }
    }

    @Override
    @PreDestroy
    public void shutdown() {
        sendGate.writeLock().lock();
        try {
            if (!closed.compareAndSet(false, true)) {
                return;
            }
        } finally {
            sendGate.writeLock().unlock();
        }
        pendingFailures.forEach((batchId, failure) -> {
            if (!rpcStarted.contains(batchId)
                    && pendingFailures.remove(batchId, failure)) {
                failure.run();
            }
        });
        for (Runnable abandoned : dispatchExecutor.shutdownNow()) {
            try {
                abandoned.run();
            } catch (Throwable failure) {
                Logger.error("FlexLB abandoned dispatch task failed during shutdown", failure);
            }
        }
    }

    // ==================== Internal: dispatch pipeline (runs on executor thread) ====================

    private void doDispatch(List<BatchItem> items, PrefillEndpoint prefillEp,
                            long batchId, long predMs, String reason,
                            DispatchCallback callback, Runnable pendingFailure) {
        try {
            doDispatchInternal(items, prefillEp, batchId, predMs, reason,
                    callback, pendingFailure);
        } catch (Throwable t) {
            Logger.error("Unexpected error in doDispatch batchId={}", batchId, t);
            failPending(batchId, pendingFailure, items, prefillEp, t, callback);
        }
    }

    private void doDispatchInternal(List<BatchItem> items, PrefillEndpoint prefillEp,
                                    long batchId, long predMs, String reason,
                                    DispatchCallback callback, Runnable pendingFailure) {
        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, items);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failPending(batchId, pendingFailure, items, prefillEp,
                    new RuntimeException("Batch request build failed: " + e.getMessage(), e), callback);
            return;
        }

        // 2. Log dispatch
        logDispatch(batchId, items, prefillEp, predMs, reason);

        // 3. Send gRPC (async)
        long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
        java.util.concurrent.CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture;
        sendGate.readLock().lock();
        try {
            if (closed.get() || pendingFailures.get(batchId) != pendingFailure) {
                return;
            }
            rpcStarted.add(batchId);
            rpcFuture = grpcClient.batchEnqueueAsync(
                    prefillEp.getIp(), prefillEp.getGrpcPort(), request, deadlineMs);
        } finally {
            sendGate.readLock().unlock();
        }
        rpcFuture
                .whenComplete((response, error) -> deliverCompletion(
                        items, prefillEp, batchId, response, error, callback, pendingFailure));
    }

    private void failItems(List<BatchItem> items, PrefillEndpoint prefillEp,
                           long batchId, Throwable error, DispatchCallback callback) {
        try {
            prefillEp.releaseBatch(batchId);
        } catch (Throwable releaseFailure) {
            Logger.error("Failed to release Prefill batch {}", batchId, releaseFailure);
        }
        for (BatchItem item : items) {
            safeOnFailure(callback, item, error, batchId);
        }
    }

    private void failPending(long batchId,
                             Runnable pendingFailure,
                             List<BatchItem> items,
                             PrefillEndpoint prefillEp,
                             Throwable error,
                             DispatchCallback callback) {
        if (pendingFailures.remove(batchId, pendingFailure)) {
            rpcStarted.remove(batchId);
            failItems(items, prefillEp, batchId, error, callback);
        }
    }

    private void deliverCompletion(List<BatchItem> items,
                                   PrefillEndpoint prefillEp,
                                   long batchId,
                                   EngineRpcService.EnqueueBatchResponsePB response,
                                   Throwable error,
                                   DispatchCallback callback,
                                   Runnable pendingFailure) {
        if (pendingFailures.get(batchId) != pendingFailure) {
            return;
        }
        Runnable completion = () -> {
            if (!pendingFailures.remove(batchId, pendingFailure)) {
                return;
            }
            rpcStarted.remove(batchId);
            try {
                if (error != null) {
                    Throwable cause = error instanceof CompletionException ? error.getCause() : error;
                    Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                            batchId, prefillEp.getIp(), prefillEp.getGrpcPort(), cause.getMessage());
                    if (Status.fromThrowable(cause).getCode() == Status.Code.DEADLINE_EXCEEDED) {
                        try {
                            prefillEp.releaseBatch(batchId);
                        } catch (Throwable releaseFailure) {
                            Logger.error("Failed to release timed out Prefill batch {}",
                                    batchId, releaseFailure);
                        }
                        for (BatchItem item : items) {
                            safeOnTimeout(callback, item, cause, batchId);
                        }
                    } else {
                        failItems(items, prefillEp, batchId,
                                new RuntimeException("gRPC dispatch failed: " + cause.getMessage(), cause),
                                callback);
                    }
                } else if (response == null) {
                    failItems(items, prefillEp, batchId,
                            new RuntimeException("EnqueueBatch returned null response"), callback);
                } else {
                    handleResponse(batchId, items, response, callback);
                }
            } catch (Throwable callbackFailure) {
                Logger.error("Unexpected error in EnqueueBatch callback batchId={}",
                        batchId, callbackFailure);
                failItems(items, prefillEp, batchId, callbackFailure, callback);
            }
        };
        try {
            dispatchExecutor.execute(completion);
        } catch (RejectedExecutionException rejected) {
            Logger.warn("FlexLB completion executor rejected batch {}, running inline", batchId);
            completion.run();
        }
    }

    private static void safeOnSuccess(DispatchCallback callback, BatchItem item, long batchId) {
        try {
            callback.onSuccess(item, batchId);
        } catch (Throwable callbackFailure) {
            Logger.error("FlexLB success callback failed request_id={} batch_id={}",
                    item.requestId(), batchId, callbackFailure);
        }
    }

    private static void safeOnFailure(DispatchCallback callback,
                                      BatchItem item,
                                      Throwable error,
                                      long batchId) {
        try {
            callback.onFailure(item, error);
        } catch (Throwable callbackFailure) {
            Logger.error("FlexLB failure callback failed request_id={} batch_id={}",
                    item.requestId(), batchId, callbackFailure);
        }
    }

    private static void safeOnTimeout(DispatchCallback callback,
                                      BatchItem item,
                                      Throwable error,
                                      long batchId) {
        try {
            callback.onTimeout(item, error);
        } catch (Throwable callbackFailure) {
            Logger.error("FlexLB timeout callback failed request_id={} batch_id={}",
                    item.requestId(), batchId, callbackFailure);
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
                safeOnFailure(callback, item, mismatch, batchId);
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
                safeOnSuccess(callback, item, batchId);
            } else if (errorByRequestId.containsKey(item.requestId())) {
                EngineRpcService.EnqueueBatchErrorPB error = errorByRequestId.get(item.requestId());
                String errorMessage = error.hasErrorInfo()
                        ? error.getErrorInfo().getErrorMessage()
                        : "missing error_info";
                safeOnFailure(callback, item, new RuntimeException(
                        "EnqueueBatch rejected request " + item.requestId() + ": " + errorMessage), batchId);
            } else {
                safeOnFailure(callback, item, new RuntimeException(
                        "EnqueueBatch missing ack for request " + item.requestId()), batchId);
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
            ServerStatus prefill = server(item, RoleType.PREFILL);
            if (prefill == null) {
                throw new IllegalArgumentException("prefill route is missing for request " + item.requestId());
            }
            byDpRank.computeIfAbsent(prefill.getDpRank(), ignored -> new ArrayList<>()).add(item);
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
        addRoleAddr(config, server(item, RoleType.PREFILL));
        addRoleAddr(config, server(item, RoleType.DECODE));
        return input.build();
    }

    private static ServerStatus server(BatchItem item, RoleType role) {
        if (item.routeResponse() == null || item.routeResponse().getServerStatus() == null) {
            return null;
        }
        for (ServerStatus status : item.routeResponse().getServerStatus()) {
            if (status != null && status.getRole() == role) {
                return status;
            }
        }
        return null;
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
