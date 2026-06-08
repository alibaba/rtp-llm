package org.flexlb.balance.scheduler;

import com.google.protobuf.Int32Value;
import com.google.protobuf.Int64Value;
import com.google.protobuf.InvalidProtocolBufferException;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.BatcherSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.balance.strategy.RequestProfile;
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
import java.util.Base64;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

@Component
public class FlexlbBatchScheduler {

    private final ConfigService configService;
    private final Router router;
    private final EngineGrpcClient grpcClient;
    private final EngineWorkerStatus engineWorkerStatus;
    private final EndpointRegistry endpointRegistry;
    private final ExecutorService dispatchExecutor;
    private final Map<String, WorkerBatcher> batchers = new ConcurrentHashMap<>();
    private final Map<Long, InflightEntry> inflight = new ConcurrentHashMap<>();
    private final AtomicLong batchIdGenerator = new AtomicLong(0);

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

            FlexlbConfig cfg = configService.loadBalanceConfig();
            long deadlineMs = computeDeadlineMs(ctx, prefill, cfg);
            BatchItem item = new BatchItem(ctx, future, routeResponse, copyOf(prefill), copyOf(decode), deadlineMs);
            String batcherKey = batcherKey(ctx.getRequest(), prefill);
            batchers.computeIfAbsent(batcherKey, k -> {
                WorkerBatcher b = new WorkerBatcher(k, copyOf(prefill));
                b.start();
                return b;
            }).offer(item);
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

    public void onRequestsFinished(List<Long> requestIds) {
        if (requestIds == null || requestIds.isEmpty()) {
            return;
        }
        for (Long requestId : requestIds) {
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

        long totalTokens = 0;
        long totalHit = 0;
        StringBuilder itemDetail = new StringBuilder();
        for (int i = 0; i < activeItems.size(); i++) {
            BatchItem item = activeItems.get(i);
            long sl = seqLenOf(item);
            long hc = hitOf(item);
            totalTokens += sl;
            totalHit += hc;
            if (i > 0) {
                itemDetail.append(", ");
            }
            itemDetail.append("{req_id=").append(item.requestId())
                    .append(" seq_len=").append(sl)
                    .append(" hit_cache=").append(hc).append('}');
        }
        List<RequestProfile> profiles = new ArrayList<>(activeItems.size());
        for (BatchItem item : activeItems) {
            profiles.add(new RequestProfile(seqLenOf(item), hitOf(item)));
        }
        PrefillTimePredictor predictor = createPredictor(configService.loadBalanceConfig());
        long predMs = predictor.predictBatchMs(profiles);
        ServerStatus entrypoint = batchEntrypoint(prefill);
        Logger.info("flexlb_batch_dispatch batch_id={} batch_size={} total_tokens={} total_hit={} "
                        + "pred_ms={} prefill={}:{} entrypoint={}:{} items=[{}]",
                batchId, activeItems.size(), totalTokens, totalHit, predMs,
                prefill.getServerIp(), prefill.getGrpcPort(),
                entrypoint.getServerIp(), entrypoint.getGrpcPort(), itemDetail);

        String ipPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
        WorkerEndpoint ep = endpointRegistry.get(ipPort);
        if (ep != null) {
            ep.commitBatch(batchId, predMs);
        }

        try {
            long deadlineMs = configService.loadBalanceConfig().getFlexlbBatchEnqueueDeadlineMs();
            EngineRpcService.EnqueueBatchResponsePB response =
                    grpcClient.batchEnqueue(entrypoint.getServerIp(), entrypoint.getGrpcPort(), request, deadlineMs);
            handleAck(batchId, activeItems, response);
        } catch (Throwable t) {
            Logger.warn("EnqueueBatch failed batchId: {}, entrypoint: {}:{}, err: {}",
                    batchId, entrypoint.getServerIp(), entrypoint.getGrpcPort(), t.getMessage());
            if (ep != null) {
                ep.releaseBatch(batchId);
            }
            for (BatchItem item : activeItems) {
                failAck(item, t);
            }
        }
    }

    private EngineRpcService.EnqueueBatchRequestPB buildBatchRequest(long batchId, List<BatchItem> items)
            throws InvalidProtocolBufferException {
        EngineRpcService.EnqueueBatchRequestPB.Builder builder = EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                .setBatchId(batchId);
        Map<Long, List<BatchItem>> byDpRank = new HashMap<>();
        for (BatchItem item : items) {
            byDpRank.computeIfAbsent(item.prefill.getDpRank(), ignored -> new ArrayList<>()).add(item);
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
        Request dto = item.ctx.getRequest();
        byte[] bytes = Base64.getDecoder().decode(dto.getGenerateInputPbB64());
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.parseFrom(bytes).toBuilder();
        if (input.getRequestId() != item.requestId()) {
            throw new IllegalArgumentException("request_id mismatch between schedule request and GenerateInputPB");
        }
        input.setGroupId(Int64Value.of(batchId));
        input.setGroupSize(groupSize);

        EngineRpcService.GenerateConfigPB.Builder config = input.getGenerateConfigBuilder();
        config.setForceGroup(Int32Value.of(1));
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

    private void handleAck(long batchId, List<BatchItem> items, EngineRpcService.EnqueueBatchResponsePB response) {
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
            if (!item.future.isDone()) {
                item.future.completeExceptionally(new RuntimeException("EnqueueBatch inflight entry missing"));
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
            grpcClient.cancel(entry.prefill.getServerIp(),
                    entry.prefill.getGrpcPort(),
                    entry.item.requestId(),
                    deadlineMs);
            ServerStatus entrypoint = batchEntrypoint(entry.prefill);
            if (!sameGrpcEndpoint(entrypoint, entry.prefill)) {
                grpcClient.cancel(entrypoint.getServerIp(),
                        entrypoint.getGrpcPort(),
                        entry.item.requestId(),
                        deadlineMs);
            }
        } catch (RuntimeException e) {
            Logger.warn("FlexLB batch cancel failed for request {}", entry.item.requestId(), e);
        }
    }

    private ServerStatus batchEntrypoint(ServerStatus selectedPrefill) {
        if (selectedPrefill == null || selectedPrefill.getRole() == null) {
            return selectedPrefill;
        }
        Map<String, WorkerStatus> workerStatusMap =
                engineWorkerStatus.selectModelWorkerStatus(selectedPrefill.getRole(), selectedPrefill.getGroup());
        if (workerStatusMap == null) {
            return selectedPrefill;
        }
        for (WorkerStatus workerStatus : workerStatusMap.values()) {
            if (workerStatus != null
                    && workerStatus.getStatusVersion().get() >= 0
                    && workerStatus.getDpRank() == 0) {
                ServerStatus entrypoint = copyOf(selectedPrefill);
                entrypoint.setServerIp(workerStatus.getIp());
                entrypoint.setHttpPort(workerStatus.getPort());
                entrypoint.setGrpcPort(CommonUtils.toGrpcPort(workerStatus.getPort()));
                entrypoint.setDpRank(workerStatus.getDpRank());
                return entrypoint;
            }
        }
        return selectedPrefill;
    }

    private static boolean sameGrpcEndpoint(ServerStatus left, ServerStatus right) {
        if (left == null || right == null) {
            return left == right;
        }
        return Objects.equals(left.getServerIp(), right.getServerIp())
                && left.getGrpcPort() == right.getGrpcPort();
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

    public BatcherSnapshot snapshotForWorker(String model, String workerIp, int workerHttpPort) {
        String key = (model == null ? "" : model) + "|" + workerIp + ":" + CommonUtils.toGrpcPort(workerHttpPort);
        WorkerBatcher batcher = batchers.get(key);
        return batcher != null ? batcher.snapshot() : BatcherSnapshot.EMPTY;
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

    private long computeDeadlineMs(BalanceContext ctx, ServerStatus prefill, FlexlbConfig cfg) {
        long seqLen = ctx.getRequest().getSeqLen();
        long hitCache = prefill.getDebugInfo() != null ? prefill.getDebugInfo().getHitCacheLen() : 0;
        PrefillTimePredictor predictor = createPredictor(cfg);
        long predMs = predictor.estimateMs(seqLen, hitCache);
        long sloMs = cfg.resolveSloMs(seqLen);
        long workerQueueMs = getWorkerPredictedQueueTimeMs(prefill);
        return System.currentTimeMillis() + Math.max(0, sloMs - predMs - workerQueueMs);
    }

    private long getWorkerPredictedQueueTimeMs(ServerStatus prefill) {
        if (prefill == null || prefill.getRole() == null) {
            return 0;
        }
        Map<String, WorkerStatus> workerStatusMap =
                engineWorkerStatus.selectModelWorkerStatus(prefill.getRole(), prefill.getGroup());
        if (workerStatusMap == null) {
            return 0;
        }
        WorkerStatus workerStatus = workerStatusMap.get(prefill.getServerIp() + ":" + prefill.getHttpPort());
        if (workerStatus == null) {
            return 0;
        }
        return workerStatus.getPredictedQueueTimeMs().get();
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig cfg) {
        return new PrefillTimePredictor(
                cfg.getCostAlpha0(), cfg.getCostAlpha1(), cfg.getCostAlpha2(),
                cfg.getCostAlpha3(), cfg.getCostAlpha4(), cfg.getCostAlpha5());
    }

    private static long seqLenOf(BatchItem item) {
        return item.ctx() != null && item.ctx().getRequest() != null
                ? item.ctx().getRequest().getSeqLen() : 0;
    }

    private static long hitOf(BatchItem item) {
        return item.prefill() != null && item.prefill().getDebugInfo() != null
                ? item.prefill().getDebugInfo().getHitCacheLen() : 0;
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
        for (WorkerBatcher batcher : batchers.values()) {
            batcher.shutdown();
        }
        dispatchExecutor.shutdownNow();
    }

    // ==================== WorkerBatcher: SLO-budget EDF ====================

    private final class WorkerBatcher {
        private final String key;
        private final ServerStatus prefill;
        private final PriorityQueue<BatchItem> queue =
                new PriorityQueue<>(Comparator.comparingLong(BatchItem::deadlineMs));
        private final ReentrantLock lock = new ReentrantLock();
        private final Condition arrival = lock.newCondition();
        private final Thread workerThread;
        private volatile boolean stopped;

        private WorkerBatcher(String key, ServerStatus prefill) {
            this.key = key;
            this.prefill = prefill;
            this.workerThread = new Thread(this::runLoop, "flexlb-batcher-" + key);
            this.workerThread.setDaemon(true);
            this.workerThread.setUncaughtExceptionHandler((t, e) ->
                    Logger.error("WorkerBatcher[{}] thread died unexpectedly", key, e));
        }

        void start() {
            workerThread.start();
        }

        void offer(BatchItem item) {
            lock.lock();
            try {
                if (stopped) {
                    rollback(item.routeResponse);
                    item.future.completeExceptionally(new IllegalStateException("FlexLB batcher stopped"));
                    return;
                }
                int maxSize = configService.loadBalanceConfig().getFlexlbBatchQueueMaxSize();
                if (maxSize > 0 && queue.size() >= maxSize) {
                    rollback(item.routeResponse);
                    item.future.complete(Response.error(StrategyErrorType.QUEUE_FULL));
                    return;
                }
                queue.add(item);
                arrival.signalAll();
            } finally {
                lock.unlock();
            }
        }

        boolean cancelQueued(long requestId) {
            lock.lock();
            try {
                Iterator<BatchItem> it = queue.iterator();
                while (it.hasNext()) {
                    BatchItem item = it.next();
                    if (item.requestId() == requestId) {
                        it.remove();
                        completeCancelled(item);
                        return true;
                    }
                }
            } finally {
                lock.unlock();
            }
            return false;
        }

        BatcherSnapshot snapshot() {
            lock.lock();
            try {
                if (queue.isEmpty()) {
                    return BatcherSnapshot.EMPTY;
                }
                List<RequestProfile> requests = new ArrayList<>(queue.size());
                long earliest = Long.MAX_VALUE;
                long headDeadline = queue.peek().deadlineMs();
                for (BatchItem item : queue) {
                    requests.add(new RequestProfile(seqLenOf(item), hitOf(item)));
                    if (item.ctx() != null) {
                        earliest = Math.min(earliest, item.ctx().getStartTime());
                    }
                }
                return new BatcherSnapshot(queue.size(), requests, earliest, headDeadline);
            } finally {
                lock.unlock();
            }
        }

        void shutdown() {
            stopped = true;
            workerThread.interrupt();
            lock.lock();
            try {
                arrival.signalAll();
                for (BatchItem item : queue) {
                    item.future.completeExceptionally(new CancellationException("FlexLB batcher stopped: " + key));
                }
                queue.clear();
            } finally {
                lock.unlock();
            }
        }

        private void runLoop() {
            while (!stopped && !Thread.currentThread().isInterrupted()) {
                try {
                    waitForNonEmpty();
                    processQueue();
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Throwable t) {
                    Logger.error("WorkerBatcher[{}] loop failed", key, t);
                }
            }
        }

        private void waitForNonEmpty() throws InterruptedException {
            lock.lock();
            try {
                while (queue.isEmpty()) {
                    arrival.await();
                    if (stopped) {
                        throw new InterruptedException("stopped");
                    }
                }
            } finally {
                lock.unlock();
            }
        }

        private void processQueue() throws InterruptedException {
            lock.lock();
            try {
                if (queue.isEmpty()) {
                    return;
                }

                FlexlbConfig cfg = configService.loadBalanceConfig();
                long marginMs = cfg.getCostSloRiskMarginMs();
                int maxScan = cfg.getFlexlbBatchScanAhead();
                double fillThreshold = cfg.getFlexlbBatchFillThreshold();
                int bsIter = cfg.getFlexlbBatchSearchIter();
                int maxCapacity = cfg.getFlexlbBatchMaxCapacity();
                int batchSizeMax = cfg.getFlexlbBatchSizeMax();

                BatchItem head = queue.peek();
                long budgetMs = head.deadlineMs() - System.currentTimeMillis();

                // 1. expired → drop
                if (budgetMs < 0) {
                    queue.poll();
                    dropItem(head);
                    return;
                }

                // 2. urgent → dispatch head alone
                if (budgetMs < marginMs) {
                    queue.poll();
                    flushItems(List.of(head));
                    return;
                }

                // 3. binary search for max batch tokens within budget
                PrefillTimePredictor predictor = createPredictor(cfg);
                long headTokens = seqLenOf(head);
                long headHit = hitOf(head);
                long lo = headTokens;
                long hi = maxCapacity;
                for (int i = 0; i < bsIter && lo < hi; i++) {
                    long mid = lo + (hi - lo + 1) / 2;
                    if (predictor.estimateMs(mid, headHit) > budgetMs - marginMs) {
                        hi = mid - 1;
                    } else {
                        lo = mid;
                    }
                }
                long batchMaxTokens = Math.max(headTokens, lo);

                // 4. greedy fill from queue
                List<BatchItem> picked = new ArrayList<>();
                picked.add(head);
                long sumTokens = headTokens;
                int scanned = 0;
                for (BatchItem c : queue) {
                    if (c == head) {
                        continue;
                    }
                    if (scanned >= maxScan) {
                        break;
                    }
                    scanned++;
                    long cTok = seqLenOf(c);
                    if (sumTokens + cTok <= batchMaxTokens) {
                        picked.add(c);
                        sumTokens += cTok;
                    }
                }

                // 5. dispatch or wait
                double fillRatio = batchMaxTokens > 0 ? (double) sumTokens / batchMaxTokens : 1.0;
                if (fillRatio >= fillThreshold || picked.size() >= batchSizeMax) {
                    for (BatchItem item : picked) {
                        queue.remove(item);
                    }
                    flushItems(picked);
                    return;
                }

                // park — budget shrinks each iteration, converges to dispatch
                arrival.awaitNanos(1_000_000L);
            } finally {
                lock.unlock();
            }
        }

        private void flushItems(List<BatchItem> items) {
            for (BatchItem item : items) {
                inflight.put(item.requestId(), new InflightEntry(item, item.prefill()));
            }
            try {
                dispatchExecutor.execute(() -> dispatch(items, prefill));
            } catch (java.util.concurrent.RejectedExecutionException e) {
                Logger.warn("FlexLB batch dispatch rejected (executor shutdown), failing {} items", items.size());
                for (BatchItem item : items) {
                    failAck(item, e);
                }
            }
        }

        private void dropItem(BatchItem item) {
            rollback(item.routeResponse);
            if (!item.future.isDone()) {
                item.future.completeExceptionally(
                        new RuntimeException("FlexLB request deadline expired — cannot meet TTFT SLO"));
            }
        }
    }

    private static final class BatchRequestBuildException extends RuntimeException {
        private BatchRequestBuildException(Throwable cause) {
            super(cause);
        }
    }

    private record BatchItem(BalanceContext ctx,
                             CompletableFuture<Response> future,
                             Response routeResponse,
                             ServerStatus prefill,
                             ServerStatus decode,
                             long deadlineMs) {
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
