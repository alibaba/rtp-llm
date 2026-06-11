package org.flexlb.balance.scheduler;

import com.google.protobuf.Int64Value;
import com.google.protobuf.InvalidProtocolBufferException;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.BatcherSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.BatchRequest;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.Lazy;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class FlexlbBatchScheduler implements BatchDecisionHandler {

    public final ConfigService configService;
    private final Router router;
    final EngineGrpcClient grpcClient;
    final EngineWorkerStatus engineWorkerStatus;
    final EndpointRegistry endpointRegistry;
    final ExecutorService dispatchExecutor;
    final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    final AtomicLong batchIdGenerator = new AtomicLong(0);

    public FlexlbBatchScheduler(ConfigService configService,
                                @Lazy Router router,
                                EngineGrpcClient grpcClient,
                                EngineWorkerStatus engineWorkerStatus,
                                EndpointRegistry endpointRegistry) {
        this.configService = configService;
        this.router = router;
        this.grpcClient = grpcClient;
        this.engineWorkerStatus = engineWorkerStatus;
        this.endpointRegistry = endpointRegistry;
        this.dispatchExecutor = Executors.newCachedThreadPool();
    }

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            if (ctx == null || ctx.getRequest() == null) {
                future.complete(Response.error(StrategyErrorType.INVALID_REQUEST));
                return future;
            }

            int maxInflight = configService.loadBalanceConfig().getFlexlbBatchMaxInflight();
            if (maxInflight > 0 && inflight.size() >= maxInflight) {
                future.complete(Response.error(StrategyErrorType.QUEUE_FULL));
                return future;
            }

            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                future.complete(routeResponse != null
                        ? routeResponse
                        : Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollback(routeResponse);
                future.complete(Response.error(StrategyErrorType.NO_PREFILL_WORKER));
                return future;
            }

            PrefillEndpoint ep = endpointRegistry.getPrefill(prefill.getServerIp() + ":" + prefill.getHttpPort());
            if (ep == null) {
                rollback(routeResponse);
                future.complete(Response.error(StrategyErrorType.NO_PREFILL_WORKER));
                return future;
            }
            long seqLen = ctx.getRequest().getSeqLen();
            long sloMs = configService.loadBalanceConfig().resolveSloMs(seqLen);
            long deadlineMs = ep.computeDeadlineMs(seqLen, BatchItem.hitCacheOf(prefill), sloMs);
            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode), deadlineMs, System.currentTimeMillis());
            inflight.put(ctx.getRequestId(), new InflightEntry(item));
            WorkerBatcher batcher = ep.getBatcher();
            batcher.offer(item);
        } catch (Throwable t) {
            if (ctx != null) {
                inflight.remove(ctx.getRequestId());
            }
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            future.completeExceptionally(t);
        }
        return future;
    }

    public void cancel(long requestId) {
        InflightEntry entry = inflight.remove(requestId);
        if (entry == null) {
            Logger.debug("flexlb batch cancel ignored; request {} not found in inflight", requestId);
            return;
        }

        boolean removeNow;
        synchronized (entry) {
            entry.cancelled.set(true);
            if (!entry.ackFinished) {
                completeCancelled(entry);
                removeNow = false;
            } else {
                rollbackOnce(entry);
                removeNow = true;
            }
        }
        cancelPrefill(entry);
        if (removeNow) {
            inflight.remove(requestId);
        }
    }

    public void onRequestsFinished(List<Long> requestIds) {
        if (requestIds == null || requestIds.isEmpty()) {
            return;
        }
        for (Long requestId : requestIds) {
            InflightEntry entry = inflight.remove(requestId);
            if (entry != null) {
                rollbackOnce(entry);
            }
        }
    }

    public void removeInflight(long requestId) {
        inflight.remove(requestId);
    }

    @Scheduled(fixedRate = 60000L)
    public void cleanupInflight() {
        long now = System.currentTimeMillis();
        long ttlMs = Math.max(1000L, configService.loadBalanceConfig().getFlexlbBatchInflightTtlMs());
        Iterator<Map.Entry<Long, InflightEntry>> iterator = inflight.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Long, InflightEntry> entry = iterator.next();
            if (now - entry.getValue().createdAtMs > ttlMs) {
                iterator.remove();
            }
        }
    }

    private void flushItems(List<BatchItem> items, DispatchMeta meta) {
        ServerStatus prefill = items.get(0).prefill();
        for (BatchItem item : items) {
            inflight.put(item.requestId(), new InflightEntry(item));
        }
        try {
            dispatchExecutor.execute(() -> dispatch(items, prefill, meta));
        } catch (java.util.concurrent.RejectedExecutionException e) {
            Logger.warn("FlexLB batch dispatch rejected (executor shutdown), failing {} items", items.size());
            for (BatchItem item : items) {
                failAck(item, e);
            }
        }
    }

    void dispatch(List<BatchItem> items, ServerStatus prefill, DispatchMeta meta) {
        List<BatchItem> activeItems = items.stream()
                .filter(item -> !isCancelled(item) && !item.future().isDone())
                .toList();
        for (BatchItem item : items) {
            if (!activeItems.contains(item)) {
                completeCancelled(item);
            }
        }
        if (activeItems.isEmpty()) {
            return;
        }

        long batchId = batchIdGenerator.incrementAndGet();
        EngineRpcService.EnqueueBatchRequestPB request;
        try {
            request = buildBatchRequest(batchId, activeItems);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            for (BatchItem item : activeItems) {
                failAck(item, e);
            }
            return;
        }

        List<BatchRequest> profiles = new ArrayList<>(activeItems.size());
        for (BatchItem item : activeItems) {
            profiles.add(new BatchRequest(item.requestId(), item.seqLen(), item.hitCache()));
        }
        String ipPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
        PrefillEndpoint ep = endpointRegistry.getPrefill(ipPort);
        PrefillTimePredictor predictor = ep.getPredictor();
        long predMs = predictor.predictBatchMs(profiles);
        long now = System.currentTimeMillis();
        BatchItem head = activeItems.get(0);
        long waitMs = now - head.enqueuedAtMs();
        long budgetMs = head.deadlineMs() - now;
        logBatchDispatch(batchId, activeItems, profiles, predMs, meta, waitMs, budgetMs, prefill);

        if (ep != null) {
            ep.commitBatch(batchId, predMs, profiles);
        }

        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            EngineRpcService.EnqueueBatchResponsePB response =
                    grpcClient.batchEnqueue(prefill.getServerIp(), prefill.getGrpcPort(), request, deadlineMs);
            handleAck(batchId, activeItems, response);
        } catch (Throwable t) {
            Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                    batchId, prefill.getServerIp(), prefill.getGrpcPort(), t.getMessage());
            PrefillEndpoint epFail = endpointRegistry.getPrefill(ipPort);
            if (epFail != null) {
                epFail.releaseBatch(batchId);
            }
            for (BatchItem item : activeItems) {
                failAck(item, t);
            }
        }
    }

    private void logBatchDispatch(
            long batchId, List<BatchItem> items, List<BatchRequest> profiles,
            long predMs, DispatchMeta meta, long waitMs, long budgetMs,
            ServerStatus prefill) {
        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            BatchRequest p = profiles.get(i);
            totalTokens += p.seqLen();
            totalHit += p.hitCache();
            if (i > 0) {
                itemDetail.append(", ");
            }
            itemDetail.append("{req_id=").append(items.get(i).requestId())
                    .append(" seq_len=").append(p.seqLen())
                    .append(" hit_cache=").append(p.hitCache()).append('}');
        }
        Logger.info("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} reason={} wait_ms={} budget_ms={} fill_ratio={} "
                        + "batch_max_tokens={} queue_remaining={} "
                        + "prefill={}:{} items=[{}]",
                batchId, items.size(), totalTokens, totalHit, predMs,
                meta.reason(), waitMs, budgetMs, String.format("%.4f", meta.fillRatio()),
                meta.batchMaxTokens(), meta.queueDepth(),
                prefill.getServerIp(), prefill.getGrpcPort(),
                itemDetail);
    }

    EngineRpcService.EnqueueBatchRequestPB buildBatchRequest(long batchId, List<BatchItem> items)
            throws InvalidProtocolBufferException {
        EngineRpcService.EnqueueBatchRequestPB.Builder builder = EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                .setBatchId(batchId);
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
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.parseFrom(bytes).toBuilder();
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

    void handleAck(long batchId, List<BatchItem> items, EngineRpcService.EnqueueBatchResponsePB response) {
        if (response == null) {
            RuntimeException error = new RuntimeException("EnqueueBatch returned null response");
            for (BatchItem item : items) {
                failAck(item, error);
            }
            return;
        }

        Map<Long, EngineRpcService.EnqueueBatchErrorPB> errorByRequestId =
                new HashMap<>(response.getErrorsCount() * 2);
        for (EngineRpcService.EnqueueBatchErrorPB error : response.getErrorsList()) {
            errorByRequestId.put(error.getRequestId(), error);
        }
        Map<Long, EngineRpcService.EnqueueBatchSuccessPB> successByRequestId =
                new HashMap<>(response.getSuccessesCount() * 2);
        for (EngineRpcService.EnqueueBatchSuccessPB success : response.getSuccessesList()) {
            successByRequestId.put(success.getRequestId(), success);
        }

        for (BatchItem item : items) {
            InflightEntry entry = inflight.get(item.requestId());
            if (successByRequestId.containsKey(item.requestId())) {
                completeAckSuccess(batchId, item, entry);
                continue;
            }

            if (errorByRequestId.containsKey(item.requestId())) {
                EngineRpcService.EnqueueBatchErrorPB error = errorByRequestId.get(item.requestId());
                String errorMessage = error.hasErrorInfo()
                        ? error.getErrorInfo().getErrorMessage()
                        : "missing error_info";
                failAck(item, new RuntimeException("EnqueueBatch rejected request "
                        + item.requestId() + ": " + errorMessage));
                continue;
            }

            failAck(item, new RuntimeException("EnqueueBatch missing ack for request " + item.requestId()));
        }
    }

    private void completeAckSuccess(long batchId, BatchItem item, InflightEntry entry) {
        if (entry == null) {
            // cancel() already removed entry and handled cleanup, just return
            return;
        }

        boolean cancelAfterAck = false;
        synchronized (entry) {
            entry.ackFinished = true;
            if (entry.cancelled.get()) {
                cancelAfterAck = true;
            } else if (!item.future().isDone()) {
                Response success = copyResponse(item.routeResponse());
                success.setSuccess(true);
                success.setCode(200);
                success.setEnqueuedByMaster(true);
                success.setQueueLength(inflight.size());
                item.future().complete(success);
                Logger.debug("FlexLB batch enqueued request {} in batch {}", item.requestId(), batchId);
            }
        }

        if (cancelAfterAck) {
            inflight.remove(item.requestId());
            cancelPrefill(entry);
            completeCancelled(entry);
        }
    }

    void failAck(BatchItem item, Throwable error) {
        InflightEntry entry = inflight.remove(item.requestId());
        if (entry != null) {
            synchronized (entry) {
                entry.ackFinished = true;
                rollbackOnce(entry);
                if (!item.future().isDone()) {
                    item.future().completeExceptionally(error);
                }
            }
            return;
        }
        rollback(item.routeResponse());
        if (!item.future().isDone()) {
            item.future().completeExceptionally(error);
        }
    }

    private void completeCancelled(BatchItem item) {
        InflightEntry entry = inflight.remove(item.requestId());
        if (entry != null) {
            synchronized (entry) {
                completeCancelled(entry);
            }
            return;
        }
        item.ctx().cancel();
        rollback(item.routeResponse());
        if (!item.future().isDone()) {
            item.future().completeExceptionally(new CancellationException("Request cancelled by client"));
        }
    }

    private void completeCancelled(InflightEntry entry) {
        entry.item.ctx().cancel();
        rollbackOnce(entry);
        if (!entry.item.future().isDone()) {
            entry.item.future().completeExceptionally(new CancellationException("Request cancelled by client"));
        }
    }

    private void rollbackOnce(InflightEntry entry) {
        if (entry.rolledBack.compareAndSet(false, true)) {
            rollback(entry.item.routeResponse());
        }
    }

    private boolean isCancelled(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return item.ctx().isCancelled() || (entry != null && entry.cancelled.get());
    }

    private void cancelPrefill(InflightEntry entry) {
        ServerStatus prefill = entry.item.prefill();
        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            grpcClient.cancel(prefill.getServerIp(),
                    prefill.getGrpcPort(),
                    entry.item.requestId(),
                    deadlineMs);
        } catch (RuntimeException e) {
            Logger.warn("FlexLB batch cancel failed for request {}", entry.item.requestId(), e);
        }
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

    @PreDestroy
    public void shutdown() {
        endpointRegistry.close();
        dispatchExecutor.shutdownNow();
    }

    @Override
    public void onExpired(BatchItem head) {
        removeInflight(head.requestId());
        rollback(head.routeResponse());
        if (!head.future().isDone()) {
            head.future().completeExceptionally(
                    new RuntimeException("FlexLB request deadline expired — cannot meet TTFT SLO"));
        }
    }

    @Override
    public void onUrgent(BatchItem head, DispatchMeta meta) {
        flushItems(List.of(head), meta);
    }

    @Override
    public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
        flushItems(items, meta);
    }

    @Override
    public void onOfferFailure(BatchItem item, Throwable error) {
        rollback(item.routeResponse());
        item.future().completeExceptionally(error);
    }

    private static final class BatchRequestBuildException extends RuntimeException {
        private BatchRequestBuildException(Throwable cause) {
            super(cause);
        }
    }

    static final class InflightEntry {
        final BatchItem item;
        final long createdAtMs = System.currentTimeMillis();
        final AtomicBoolean cancelled = new AtomicBoolean(false);
        final AtomicBoolean rolledBack = new AtomicBoolean(false);
        boolean ackFinished;

        InflightEntry(BatchItem item) {
            this.item = Objects.requireNonNull(item);
            Objects.requireNonNull(item.prefill(), "BatchItem.prefill must not be null");
        }
    }
}
