package org.flexlb.balance.scheduler;

import com.google.protobuf.Int32Value;
import com.google.protobuf.Int64Value;
import com.google.protobuf.InvalidProtocolBufferException;
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
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Base64;
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
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class FlexlbBatchScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EngineGrpcClient grpcClient;
    private final EngineWorkerStatus engineWorkerStatus;
    private final ScheduledExecutorService timerExecutor;
    private final ExecutorService dispatchExecutor;
    private final Map<String, WorkerBatcher> batchers = new ConcurrentHashMap<>();
    private final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    private final AtomicLong batchIdGenerator = new AtomicLong(0);

    public FlexlbBatchScheduler(ConfigService configService,
                                Router router,
                                EngineGrpcClient grpcClient,
                                EngineWorkerStatus engineWorkerStatus) {
        this.configService = configService;
        this.router = router;
        this.grpcClient = grpcClient;
        this.engineWorkerStatus = engineWorkerStatus;
        this.timerExecutor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "flexlb-batch-timer");
            t.setDaemon(true);
            return t;
        });
        this.dispatchExecutor = Executors.newFixedThreadPool(
                Math.max(1, configService.loadBalanceConfig().getScheduleWorkerSize()),
                r -> {
                    Thread t = new Thread(r, "flexlb-batch-dispatcher");
                    t.setDaemon(true);
                    return t;
                });
    }

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            if (ctx == null || ctx.getRequest() == null) {
                future.complete(Response.error(StrategyErrorType.INVALID_REQUEST));
                return future;
            }

            Response routeResponse = router.route(ctx);
            if (routeResponse == null || !routeResponse.isSuccess()) {
                future.complete(routeResponse != null ? routeResponse : Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
                return future;
            }

            ServerStatus prefill = findServer(routeResponse, RoleType.PREFILL);
            ServerStatus decode = findServer(routeResponse, RoleType.DECODE);
            if (prefill == null) {
                rollback(routeResponse);
                future.complete(Response.error(StrategyErrorType.NO_PREFILL_WORKER));
                return future;
            }

            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode));
            String batcherKey = batcherKey(ctx.getRequest(), prefill);
            batchers.computeIfAbsent(batcherKey, ignored -> new WorkerBatcher(batcherKey, copyOf(prefill))).offer(item);
        } catch (Throwable t) {
            Logger.error("FlexlbBatchScheduler submit failed for request id: {}",
                    ctx == null ? null : ctx.getRequestId(), t);
            future.completeExceptionally(t);
        }
        return future;
    }

    public void cancel(long requestId) {
        for (WorkerBatcher batcher : batchers.values()) {
            if (batcher.cancelQueued(requestId)) {
                return;
            }
        }

        InflightEntry entry = inflight.get(requestId);
        if (entry == null) {
            Logger.debug("flexlb batch cancel ignored; request {} not found", requestId);
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

    private void dispatch(List<BatchItem> items, ServerStatus prefill) {
        List<BatchItem> activeItems = items.stream()
                .filter(item -> !isCancelled(item) && !item.future.isDone())
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
        EngineRpcService.BatchEnqueueRequestPB request;
        try {
            request = buildBatchRequest(batchId, activeItems);
        } catch (Exception e) {
            Logger.error("Failed to build FlexLB batch request batchId: {}", batchId, e);
            for (BatchItem item : activeItems) {
                failAck(item, e);
            }
            return;
        }

        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            EngineRpcService.BatchEnqueueResponsePB response =
                    grpcClient.batchEnqueue(prefill.getServerIp(), prefill.getGrpcPort(), request, deadlineMs);
            handleAck(batchId, activeItems, response);
        } catch (Throwable t) {
            Logger.warn("BatchEnqueue failed batchId: {}, prefill: {}:{}, err: {}",
                    batchId, prefill.getServerIp(), prefill.getGrpcPort(), t.getMessage());
            for (BatchItem item : activeItems) {
                failAck(item, t);
            }
        }
    }

    private EngineRpcService.BatchEnqueueRequestPB buildBatchRequest(long batchId, List<BatchItem> items)
            throws InvalidProtocolBufferException {
        EngineRpcService.BatchEnqueueRequestPB.Builder builder = EngineRpcService.BatchEnqueueRequestPB.newBuilder()
                .setBatchId(batchId);
        int groupSize = items.size();
        for (BatchItem item : items) {
            builder.addInputs(buildInput(batchId, groupSize, item));
        }
        return builder.build();
    }

    private EngineRpcService.GenerateInputPB buildInput(long batchId, int groupSize, BatchItem item)
            throws InvalidProtocolBufferException {
        Request dto = item.ctx.getRequest();
        byte[] bytes = Base64.getDecoder().decode(dto.getGenerateInputPbB64());
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.parseFrom(bytes).toBuilder();
        if (input.getRequestId() != item.requestId()) {
            throw new IllegalArgumentException("request_id mismatch between schedule request and GenerateInputPB");
        }
        input.setBatchGroupId(Int64Value.of(batchId));
        input.setBatchGroupSize(groupSize);

        EngineRpcService.GenerateConfigPB.Builder config = input.getGenerateConfigBuilder();
        config.setForceBatch(Int32Value.of(1));
        config.clearRoleAddrs();
        addRoleAddr(config, item.prefill);
        addRoleAddr(config, item.decode);
        return input.build();
    }

    private void addRoleAddr(EngineRpcService.GenerateConfigPB.Builder config, ServerStatus serverStatus) {
        if (serverStatus == null) {
            return;
        }
        EngineRpcService.RoleAddrPB.RoleType role = switch (serverStatus.getRole()) {
            case PREFILL -> EngineRpcService.RoleAddrPB.RoleType.PREFILL;
            case DECODE -> EngineRpcService.RoleAddrPB.RoleType.DECODE;
            case PDFUSION -> EngineRpcService.RoleAddrPB.RoleType.PDFUSION;
            case VIT -> EngineRpcService.RoleAddrPB.RoleType.VIT;
        };
        config.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                .setRole(role)
                .setIp(serverStatus.getServerIp())
                .setHttpPort(serverStatus.getHttpPort())
                .setGrpcPort(serverStatus.getGrpcPort())
                .build());
    }

    private void handleAck(long batchId, List<BatchItem> items, EngineRpcService.BatchEnqueueResponsePB response) {
        if (response == null) {
            RuntimeException error = new RuntimeException("BatchEnqueue returned null response");
            for (BatchItem item : items) {
                failAck(item, error);
            }
            return;
        }

        Map<Long, EngineRpcService.BatchEnqueueAckPB> ackByRequestId = new HashMap<>(response.getAcksCount() * 2);
        for (EngineRpcService.BatchEnqueueAckPB ack : response.getAcksList()) {
            ackByRequestId.put(ack.getRequestId(), ack);
        }

        for (BatchItem item : items) {
            InflightEntry entry = inflight.get(item.requestId());
            EngineRpcService.BatchEnqueueAckPB ack = ackByRequestId.get(item.requestId());
            if (ack == null) {
                failAck(item, new RuntimeException("BatchEnqueue missing ack for request " + item.requestId()));
                continue;
            }
            if (ack.hasErrorInfo() && ack.getErrorInfo().getErrorCode() != 0L) {
                failAck(item, new RuntimeException("BatchEnqueue rejected request "
                        + item.requestId() + ": " + ack.getErrorInfo().getErrorMessage()));
                continue;
            }

            completeAckSuccess(batchId, item, entry);
        }
    }

    private void completeAckSuccess(long batchId, BatchItem item, InflightEntry entry) {
        if (entry == null) {
            if (!item.future.isDone()) {
                item.future.completeExceptionally(new RuntimeException("BatchEnqueue inflight entry missing"));
            }
            return;
        }

        boolean cancelAfterAck = false;
        synchronized (entry) {
            entry.ackFinished = true;
            if (entry.cancelled.get()) {
                cancelAfterAck = true;
            } else if (!item.future.isDone()) {
                Response success = copyResponse(item.routeResponse);
                success.setSuccess(true);
                success.setCode(200);
                success.setEnqueuedByMaster(true);
                item.future.complete(success);
                Logger.debug("FlexLB batch enqueued request {} in batch {}", item.requestId(), batchId);
            }
        }

        if (cancelAfterAck) {
            cancelPrefill(entry);
            inflight.remove(item.requestId());
            completeCancelled(entry);
        }
    }

    private void failAck(BatchItem item, Throwable error) {
        InflightEntry entry = inflight.remove(item.requestId());
        if (entry != null) {
            synchronized (entry) {
                entry.ackFinished = true;
                rollbackOnce(entry);
                if (!item.future.isDone()) {
                    item.future.completeExceptionally(error);
                }
            }
            return;
        }
        rollback(item.routeResponse);
        if (!item.future.isDone()) {
            item.future.completeExceptionally(error);
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
        item.ctx.cancel();
        rollback(item.routeResponse);
        if (!item.future.isDone()) {
            item.future.completeExceptionally(new CancellationException("Request cancelled by client"));
        }
    }

    private void completeCancelled(InflightEntry entry) {
        entry.item.ctx.cancel();
        rollbackOnce(entry);
        if (!entry.item.future.isDone()) {
            entry.item.future.completeExceptionally(new CancellationException("Request cancelled by client"));
        }
    }

    private void rollbackOnce(InflightEntry entry) {
        if (entry.rolledBack.compareAndSet(false, true)) {
            rollback(entry.item.routeResponse);
        }
    }

    private boolean isCancelled(BatchItem item) {
        InflightEntry entry = inflight.get(item.requestId());
        return item.ctx.isCancelled() || (entry != null && entry.cancelled.get());
    }

    private void cancelPrefill(InflightEntry entry) {
        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            grpcClient.cancel(entry.prefill.getServerIp(), entry.prefill.getGrpcPort(), entry.item.requestId(), deadlineMs);
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

    private String batcherKey(Request request, ServerStatus prefill) {
        String model = request.getModel() == null ? "" : request.getModel();
        return model + "|" + prefill.getServerIp() + ":" + prefill.getGrpcPort();
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
        if (serverStatus == null || serverStatus.getRole() == null) {
            return;
        }
        Map<String, WorkerStatus> workerStatusMap =
                engineWorkerStatus.selectModelWorkerStatus(serverStatus.getRole(), serverStatus.getGroup());
        WorkerStatus workerStatus = null;
        if (workerStatusMap != null) {
            workerStatus = workerStatusMap.get(serverStatus.getServerIp() + ":" + serverStatus.getHttpPort());
        }
        if (workerStatus != null) {
            workerStatus.removeLocalTask(serverStatus.getRequestId());
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
        for (WorkerBatcher batcher : batchers.values()) {
            batcher.shutdown();
        }
        timerExecutor.shutdownNow();
        dispatchExecutor.shutdownNow();
    }

    private final class WorkerBatcher {
        private final String key;
        private final ServerStatus prefill;
        private final List<BatchItem> queue = new ArrayList<>();
        private ScheduledFuture<?> flushFuture;
        private boolean stopped;

        private WorkerBatcher(String key, ServerStatus prefill) {
            this.key = key;
            this.prefill = prefill;
        }

        synchronized void offer(BatchItem item) {
            if (stopped) {
                rollback(item.routeResponse);
                item.future.completeExceptionally(new IllegalStateException("FlexLB batcher stopped"));
                return;
            }
            queue.add(item);
            if (queue.size() == 1) {
                scheduleFlush();
            }
            if (queue.size() >= Math.max(1, configService.loadBalanceConfig().getFlexlbBatchSizeMax())) {
                flushLocked();
            }
        }

        synchronized boolean cancelQueued(long requestId) {
            Iterator<BatchItem> iterator = queue.iterator();
            while (iterator.hasNext()) {
                BatchItem item = iterator.next();
                if (item.requestId() == requestId) {
                    iterator.remove();
                    completeCancelled(item);
                    if (queue.isEmpty()) {
                        cancelFlushTimer();
                    }
                    return true;
                }
            }
            return false;
        }

        private synchronized void flushFromTimer() {
            flushLocked();
        }

        private void scheduleFlush() {
            long delayMs = Math.max(1L, configService.loadBalanceConfig().getFlexlbBatchWindowMs());
            flushFuture = timerExecutor.schedule(this::flushFromTimer, delayMs, TimeUnit.MILLISECONDS);
        }

        private void flushLocked() {
            if (queue.isEmpty()) {
                cancelFlushTimer();
                return;
            }
            List<BatchItem> items = new ArrayList<>(queue);
            for (BatchItem item : items) {
                inflight.put(item.requestId(), new InflightEntry(item, prefill));
            }
            queue.clear();
            cancelFlushTimer();
            dispatchExecutor.execute(() -> dispatch(items, prefill));
        }

        private void cancelFlushTimer() {
            if (flushFuture != null) {
                flushFuture.cancel(false);
                flushFuture = null;
            }
        }

        synchronized void shutdown() {
            stopped = true;
            cancelFlushTimer();
            for (BatchItem item : queue) {
                item.future.completeExceptionally(new CancellationException("FlexLB batcher stopped: " + key));
            }
            queue.clear();
        }
    }

    private record BatchItem(BalanceContext ctx,
                             CompletableFuture<Response> future,
                             Response routeResponse,
                             ServerStatus prefill,
                             ServerStatus decode) {
        long requestId() {
            return ctx.getRequestId();
        }
    }

    private static final class InflightEntry {
        private final BatchItem item;
        private final ServerStatus prefill;
        private final long createdAtMs = System.currentTimeMillis();
        private final AtomicBoolean cancelled = new AtomicBoolean(false);
        private final AtomicBoolean rolledBack = new AtomicBoolean(false);
        private boolean ackFinished;

        private InflightEntry(BatchItem item, ServerStatus prefill) {
            this.item = Objects.requireNonNull(item);
            this.prefill = Objects.requireNonNull(prefill);
        }
    }
}
