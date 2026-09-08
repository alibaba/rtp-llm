package org.flexlb.balance.scheduler;

import com.google.protobuf.InvalidProtocolBufferException;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.loadbalance.Request;
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
import java.util.concurrent.CompletableFuture;
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
            Logger.warn("FlexLB batch dispatch rejected by executor, failing {} items", items.size());
            failItems(items, prefillEp, batchId, e, callback);
        }
    }

    @PreDestroy
    public void shutdown() {
        dispatchExecutor.shutdownNow();
    }

    // ==================== Internal: dispatch pipeline (runs on executor thread) ====================

    private void doDispatch(List<BatchItem> items, PrefillEndpoint prefillEp,
                            long batchId, long predMs, String reason, DispatchCallback callback) {
        DispatchAttempt attempt = new DispatchAttempt();
        try {
            doDispatchInternal(items, prefillEp, batchId, predMs, reason, callback, attempt);
        } catch (Throwable unexpectedFailure) {
            Logger.error("Unexpected dispatch failure batch_id={} rpc_invocation_started={}",
                    batchId, attempt.rpcInvocationStarted, unexpectedFailure);
            if (attempt.rpcInvocationStarted) {
                // Once invocation starts, cleanup is unsafe even if the
                // exception escaped an otherwise defensive post-send path.
                markUncertain(items, batchId, unexpectedFailure, callback);
            } else {
                failItems(items, prefillEp, batchId, unexpectedFailure, callback);
            }
        }
    }

    private void doDispatchInternal(List<BatchItem> items, PrefillEndpoint prefillEp,
                                    long batchId, long predMs, String reason, DispatchCallback callback,
                                    DispatchAttempt attempt) {
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
        try {
            logDispatch(batchId, items, prefillEp, predMs, reason);
        } catch (Throwable loggingFailure) {
            failItems(items, prefillEp, batchId,
                    "Batch dispatch preparation failed: " + loggingFailure.getMessage(), callback);
            return;
        }

        // 3. Send gRPC (async)
        // Resolve every potentially fallible argument before entering the RPC
        // invocation block. A failure here is definitely pre-send and is
        // handled by doDispatch's outer guard.
        long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
        String prefillIp = prefillEp.getIp();
        int prefillGrpcPort = prefillEp.getGrpcPort();
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> rpcFuture;
        try {
            attempt.rpcInvocationStarted = true;
            rpcFuture = grpcClient.batchEnqueueAsync(
                    prefillIp, prefillGrpcPort, request, deadlineMs);
        } catch (Throwable invocationFailure) {
            // Once client invocation starts, a synchronous exception does not
            // prove that no bytes were written. Treat it as ambiguous.
            markUncertain(items, batchId, invocationFailure, callback);
            return;
        }
        if (rpcFuture == null) {
            RuntimeException missingFuture = new RuntimeException(
                    "EnqueueBatch client returned null future after invocation");
            markUncertain(items, batchId, missingFuture, callback);
            return;
        }
        try {
            CompletableFuture<Void> completionObserver = rpcFuture.handleAsync((response, ex) -> {
                try {
                    if (ex != null) {
                        Throwable cause = unwrapCompletionFailure(ex);
                        Logger.debug("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                                batchId, prefillIp, prefillGrpcPort, cause.getMessage());
                        // Once the asynchronous RPC is invoked, no
                        // transport status proves the server did not
                        // accept the request. Reconcile every transport
                        // failure through the Engine-side request-id fence.
                        markUncertain(items, batchId, cause, callback);
                    } else if (response == null) {
                        markUncertain(items, batchId, new RuntimeException(
                                "EnqueueBatch returned null response"), callback);
                    } else {
                        handleResponse(batchId, items, response, callback);
                    }
                } catch (Throwable completionFailure) {
                    // This callback is unconditionally post-invocation. Never
                    // let an unexpected response-processing failure fall back
                    // to definite failure/cleanup.
                    markUncertain(items, batchId, completionFailure, callback);
                }
                return null;
            }, dispatchExecutor);

            // handleAsync consumes an RPC failure in the lambda above. Its
            // returned future can still fail if the executor rejects callback
            // execution, which is also post-invocation and therefore
            // ambiguous.
            completionObserver.exceptionally(observerFailure -> {
                markUncertain(items, batchId, unwrapCompletionFailure(observerFailure), callback);
                return null;
            });
        } catch (Throwable registrationFailure) {
            // Callback registration is post-invocation. The RPC may already
            // be in flight even though no completion observer was installed.
            markUncertain(items, batchId, registrationFailure, callback);
        }
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }

    private static void markUncertain(List<BatchItem> items, long batchId,
                                      Throwable error, DispatchCallback callback) {
        for (BatchItem item : items) {
            try {
                callback.onDispatchUncertain(item, batchId, error);
            } catch (Throwable callbackFailure) {
                Logger.error("Dispatch-uncertain callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
        }
    }

    private void failItems(List<BatchItem> items, PrefillEndpoint prefillEp,
                           long batchId, String message, DispatchCallback callback) {
        failItems(items, prefillEp, batchId, new RuntimeException(message), callback);
    }

    private void failItems(List<BatchItem> items, PrefillEndpoint prefillEp,
                           long batchId, Throwable error, DispatchCallback callback) {
        try {
            prefillEp.releaseBatch(batchId);
        } catch (Throwable releaseFailure) {
            Logger.error("Failed to release prefill batch batch_id={}", batchId, releaseFailure);
        }
        for (BatchItem item : items) {
            try {
                callback.onFailure(item, error);
            } catch (Throwable callbackFailure) {
                Logger.error("Dispatch-failure callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
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
            markUncertain(items, batchId, mismatch, callback);
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
            try {
                if (successIds.contains(item.requestId())) {
                    callback.onSuccess(item, batchId);
                } else if (errorByRequestId.containsKey(item.requestId())) {
                    EngineRpcService.EnqueueBatchErrorPB error = errorByRequestId.get(item.requestId());
                    long errorCode = error.hasErrorInfo()
                            ? error.getErrorInfo().getErrorCode()
                            : 0L;
                    String errorMessage = error.hasErrorInfo()
                            ? error.getErrorInfo().getErrorMessage()
                            : "missing error_info";
                    callback.onFailure(item, new EngineRejectedException(errorCode,
                            "EnqueueBatch rejected request " + item.requestId() + ": " + errorMessage));
                } else {
                    callback.onDispatchUncertain(item, batchId, new RuntimeException(
                            "EnqueueBatch missing ack for request " + item.requestId()));
                }
            } catch (Throwable callbackFailure) {
                // The callback may already have committed this item's state
                // before throwing. Never issue a second, contradictory
                // callback for it, and never let it reclassify earlier items.
                Logger.error("EnqueueBatch item callback failed request_id={} batch_id={}",
                        item.requestId(), batchId, callbackFailure);
            }
        }
    }

    /** Domain error returned for one item in an otherwise successful batch RPC. */
    public static final class EngineRejectedException extends RuntimeException {
        private final long errorCode;

        public EngineRejectedException(long errorCode, String message) {
            super(message);
            this.errorCode = errorCode;
        }

        public long errorCode() {
            return errorCode;
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
        // Pass the normalized Auto-TPM priority through to the engine
        // (metrics tagging only). normalize() always sets 1-100, so every
        // dispatched request carries its priority into the proto field.
        Request request = item.ctx().getRequest();
        if (request != null) {
            input.setPriority(request.getPriority());
        }
        return input.build();
    }

    private void addRoleAddr(EngineRpcService.GenerateConfigPB.Builder config, ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        RoleType role = serverStatus.getRole();
        config.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(RoleTypeProtoConverter.toLegacyProto(role))
                .setRoleStr(role.getCode())
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

        Logger.debug("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
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

    /** Per-dispatch phase marker used only by the executor thread. */
    private static final class DispatchAttempt {
        private boolean rpcInvocationStarted;
    }
}
