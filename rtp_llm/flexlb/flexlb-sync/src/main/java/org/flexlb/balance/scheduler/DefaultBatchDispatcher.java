package org.flexlb.balance.scheduler;

import com.google.protobuf.Int64Value;
import com.google.protobuf.InvalidProtocolBufferException;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.util.Logger;

import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
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

    private final EngineGrpcClient grpcClient;
    private final ConfigService configService;
    private final ExecutorService dispatchExecutor;

    public DefaultBatchDispatcher(EngineGrpcClient grpcClient, ConfigService configService) {
        this.grpcClient = grpcClient;
        this.configService = configService;
        int poolSize = configService.loadBalanceConfig().getFlexlbBatchDispatchPoolSize();
        int queueSize = configService.loadBalanceConfig().getFlexlbBatchDispatchQueueSize();
        this.dispatchExecutor = new ThreadPoolExecutor(
                poolSize, poolSize,
                60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(queueSize),
                new ThreadPoolExecutor.AbortPolicy());
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
        // Filter out items that were cancelled before dispatch
        List<BatchItem> active = new ArrayList<>();
        for (BatchItem item : items) {
            if (!item.future().isDone() && !item.ctx().isCancelled()) {
                active.add(item);
            } else {
                Logger.debug("Skipping cancelled item in dispatch: request_id={}, batch_id={}",
                        item.requestId(), batchId);
            }
        }

        if (active.isEmpty()) {
            Logger.debug("All items cancelled before dispatch, batch_id={}", batchId);
            prefillEp.releaseBatch(batchId);
            return;
        }

        // 1. Build gRPC request
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, active);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            failItems(active, prefillEp, batchId, "Batch request build failed: " + e.getMessage(), callback);
            return;
        }

        // 2. Log dispatch
        logDispatch(batchId, active, prefillEp, predMs, reason);

        // 3. Send gRPC
        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            EngineRpcService.EnqueueBatchResponsePB response =
                    grpcClient.batchEnqueue(prefillEp.getIp(), prefillEp.getGrpcPort(),
                            request, deadlineMs);
            if (response == null) {
                failItems(active, prefillEp, batchId, "EnqueueBatch returned null response", callback);
                return;
            }
            handleResponse(batchId, active, response, callback);
        } catch (Throwable t) {
            Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                    batchId, prefillEp.getIp(), prefillEp.getGrpcPort(), t.getMessage());
            failItems(active, prefillEp, batchId, "gRPC dispatch failed: " + t.getMessage(), callback);
        }
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

    private EngineRpcService.GenerateInputPB buildInput(long batchId, int groupSize, BatchItem item)
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
        config.clearRoleAddrs();
        addRoleAddr(config, item.prefill());
        addRoleAddr(config, item.decode());
        return input.build();
    }

    private void addRoleAddr(EngineRpcService.GenerateConfigPB.Builder config, ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        EngineRpcService.RoleTypePB role = switch (serverStatus.getRole()) {
            case PREFILL -> EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL;
            case DECODE -> EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE;
            case PDFUSION -> EngineRpcService.RoleTypePB.ROLE_TYPE_PDFUSION;
            case VIT -> EngineRpcService.RoleTypePB.ROLE_TYPE_VIT;
        };
        config.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(role)
                .setIp(serverStatus.getServerIp())
                .setHttpPort(serverStatus.getHttpPort())
                .setGrpcPort(serverStatus.getGrpcPort())
                .build());
    }

    // ==================== Logging ====================

    private void logDispatch(long batchId, List<BatchItem> items, PrefillEndpoint prefillEp, long predMs, String reason) {
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
        long budgetMs = head.deadlineMs() - now;

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
