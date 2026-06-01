package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.DispatchPlanner;
import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.DispatchBatch;
import org.flexlb.balance.dp.DispatchBatcher;
import org.flexlb.balance.dp.PrefillTimePredictor;
import org.flexlb.balance.dp.QueuedRequest;
import org.flexlb.balance.dp.SimpleDpBatcher;
import org.flexlb.balance.dp.SloBudgetBatcher;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.DpRankStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.dp.DpGrpcClient;
import org.flexlb.service.monitor.DpBatchReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.DependsOn;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DpBatchScheduler {

    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final DispatchPlanner planner;
    private final DpGrpcClient grpcClient;
    private final InflightBatchRegistry inflightRegistry;
    private final PrefillTimePredictor prefillTimePredictor;
    private DpBatchReporter dpBatchReporter;

    private final Map<String, ConcurrentHashMap<String, DispatchBatcher>> batchers = new ConcurrentHashMap<>();

    private final AtomicLong batchIdGen = new AtomicLong(0);

    public DpBatchScheduler(ConfigService configService,
                            EngineWorkerStatus engineWorkerStatus,
                            DispatchPlanner planner,
                            DpGrpcClient grpcClient,
                            InflightBatchRegistry inflightRegistry,
                            PrefillTimePredictor prefillTimePredictor) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.planner = planner;
        this.grpcClient = grpcClient;
        this.inflightRegistry = inflightRegistry;
        this.prefillTimePredictor = prefillTimePredictor;
    }

    @Autowired(required = false)
    public void setDpBatchReporter(DpBatchReporter dpBatchReporter) {
        this.dpBatchReporter = dpBatchReporter;
    }

    // ============== Submit (called by RouteService) ==============

    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            String key = modelKey(ctx);
            QueuedRequest qr = QueuedRequest.of(ctx, future);

            ConcurrentHashMap<String, DispatchBatcher> modelBatchers =
                    batchers.computeIfAbsent(key, k -> new ConcurrentHashMap<>());

            WorkerStatus prefillWorker = planner.selectPrefillWorker(
                    key, configService.loadBalanceConfig(), ctx,
                    ipPort -> {
                        DispatchBatcher b = modelBatchers.get(ipPort);
                        return b != null ? b.queueSize() : 0;
                    });
            if (prefillWorker == null) {
                future.complete(failResponse("no available prefill worker"));
                return future;
            }

            String batcherKey = prefillWorker.getIpPort();
            DispatchBatcher batcher = modelBatchers.compute(batcherKey, (k, existing) -> {
                if (existing == null || !existing.isAlive()) {
                    if (existing != null) {
                        Logger.warn("Batcher thread dead for {}, recreating", k);
                    }
                    return createBatcher(key, prefillWorker);
                }
                return existing;
            });
            batcher.offer(qr);
        } catch (Throwable t) {
            Logger.error("DpBatchScheduler.submit threw before request was queued; "
                    + "failing future to avoid leaking a hung Mono", t);
            future.completeExceptionally(t);
        }
        return future;
    }


    private DispatchBatcher createBatcher(String model, WorkerStatus prefillWorker) {
        int dpSize = (int) prefillWorker.getDpSize();
        String type = configService.loadBalanceConfig().getDpBatcherType();
        if ("SIMPLE".equalsIgnoreCase(type)) {
            SimpleDpBatcher b = new SimpleDpBatcher(model, dpSize, configService, planner,
                    this::dispatchBatch, dpBatchReporter, prefillWorker);
            b.start();
            return b;
        }
        SloBudgetBatcher b = new SloBudgetBatcher(model, configService, planner,
                this::dispatchBatch, prefillTimePredictor, dpBatchReporter,
                prefillWorker, dpSize);
        b.start();
        return b;
    }

    private static Response failResponse(String message) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(org.flexlb.dao.loadbalance.StrategyErrorType.NO_PREFILL_WORKER.getErrorCode());
        r.setErrorMessage(message);
        return r;
    }

    // ============== Dispatch (called by batchers via callback) ==============

    void dispatchBatch(DispatchBatch batch) {
        long batchId = batchIdGen.incrementAndGet();
        List<List<PendingRequest>> ranked = batch.rankedRequests();

        EngineRpcService.BatchEnqueueRequestPB pb = buildPb(batchId, ranked, batch.dpSize());
        reportBatchComposition(ranked, batch.dpSize());
        inflightRegistry.register(batchId, batch);

        Logger.info("BatchDispatch batchId={} dpSize={} realRequests={} totalInputs={}",
                batchId, batch.dpSize(), batch.size(), pb.getInputsCount());

        long deadlineMs = configService.loadBalanceConfig().getDpBatchEnqueueDeadlineMs();
        grpcClient.enqueue(batch.prefillIp(), batch.prefillGrpcPort(), pb, deadlineMs)
                .whenComplete((ack, err) -> handleAck(batch, batchId, ranked, ack, err));
    }

    private void reportBatchComposition(List<List<PendingRequest>> ranked, int dpSize) {
        if (dpBatchReporter == null || dpSize <= 0) {
            return;
        }
        int filledRanks = 0;
        for (int rank = 0; rank < ranked.size(); rank++) {
            if (!ranked.get(rank).isEmpty()) {
                dpBatchReporter.reportDpRankHit(rank);
                filledRanks++;
            }
        }
        dpBatchReporter.reportFakePadSlots(dpSize - filledRanks, dpSize);
    }

    @Scheduled(fixedRate = 1000L)
    public void reportSchedulerStats() {
        if (dpBatchReporter == null) {
            return;
        }
        dpBatchReporter.reportInflightStats(
                inflightRegistry.sizeBatches(),
                inflightRegistry.sizeRequests(),
                inflightRegistry.getEvictedCount());
    }

    private void handleAck(DispatchBatch batch, long batchId, List<List<PendingRequest>> ranked,
                           EngineRpcService.BatchEnqueueResponsePB ack, Throwable err) {
        if (err != null || ack == null) {
            String msg = err != null ? err.getMessage() : "no ack";
            Logger.warn("Master.BatchEnqueue transport failed batch={} on {}:{} err={}",
                    batchId, batch.prefillIp(), batch.prefillGrpcPort(), msg);

            for (PendingRequest r : batch.requests()) {
                if (inflightRegistry.getState(r.requestId()) == InflightBatchRegistry.RequestState.CANCELLED) {
                    cascadeEngineCancel(r);
                }
            }
            inflightRegistry.remove(batchId);
            failAll(batch, new RuntimeException("Master.BatchEnqueue transport failed: " + msg));
            return;
        }

        java.util.Map<Long, EngineRpcService.EnqueueResponsePB> ackByReqId =
                new java.util.HashMap<>(ack.getAcksCount() * 2);
        for (EngineRpcService.EnqueueResponsePB slot : ack.getAcksList()) {
            ackByReqId.put(slot.getRequestId(), slot);
        }

        for (int rank = 0; rank < ranked.size(); rank++) {
            for (PendingRequest req : ranked.get(rank)) {
                try {
                    processSlotAck(req, rank, batchId, batch, ackByReqId);
                } catch (Throwable t) {
                    Logger.error("handleAck failed for request {} in batch {}",
                            req.requestId(), batchId, t);
                    inflightRegistry.removeRequest(req.requestId());
                    retryOrFail(req, t);
                }
            }
        }
    }

    private void processSlotAck(PendingRequest req, int rank, long batchId,
                                DispatchBatch batch,
                                Map<Long, EngineRpcService.EnqueueResponsePB> ackByReqId) {
        long reqId = req.requestId();
        EngineRpcService.EnqueueResponsePB slotAck = ackByReqId.get(reqId);
        if (slotAck == null) {
            inflightRegistry.removeRequest(reqId);
            retryOrFail(req, new RuntimeException(
                    "No ack slot for request " + reqId + " in batch " + batchId));
            return;
        }
        long errorCode = slotAck.getErrorInfo().getErrorCode();
        if (errorCode != 0L) {
            inflightRegistry.removeRequest(reqId);
            retryOrFail(req, new RuntimeException("Engine rejected request " + reqId
                    + " errorCode=" + errorCode + " msg=" + slotAck.getErrorInfo().getErrorMessage()));
            return;
        }
        boolean activated = inflightRegistry.markActive(reqId);
        if (!activated) {
            cascadeEngineCancel(req);
            req.future().completeExceptionally(new java.util.concurrent.CancellationException(
                    "Request " + reqId + " cancelled during PENDING_ACK"));
            inflightRegistry.removeRequest(reqId);
            return;
        }
        req.future().complete(buildSuccessResponse(req, rank));
    }

    private void cascadeEngineCancel(PendingRequest req) {
        try {
            grpcClient.cancelPrefill(req.prefill().getServerIp(), req.prefill().getGrpcPort(), req.requestId());
        } catch (RuntimeException e) {
            Logger.warn("Cancel cascade prefill failed for request {}", req.requestId(), e);
        }
        try {
            grpcClient.cancelDecode(req.decode().getServerIp(), req.decode().getGrpcPort(), req.requestId());
        } catch (RuntimeException e) {
            Logger.warn("Cancel cascade decode failed for request {}", req.requestId(), e);
        }
    }

    private Response buildSuccessResponse(PendingRequest req, int dpRank) {
        Response resp = new Response();
        resp.setSuccess(true);
        resp.setCode(200);
        resp.setEnqueuedByMaster(true);

        ServerStatus prefill = copyOf(req.prefill());
        prefill.setDpRank(dpRank);
        prefill.setRequestId(req.requestId());
        prefill.setDebugInfo(buildDebugInfo(req));
        applyDpRankAddress(prefill, dpRank);

        ServerStatus decode = copyOf(req.decode());
        decode.setRequestId(req.requestId());

        List<ServerStatus> list = new ArrayList<>(2);
        list.add(prefill);
        list.add(decode);
        resp.setServerStatus(list);
        return resp;
    }

    // ============== Cancel ==============

    public void cancel(long requestId) {
        for (ConcurrentHashMap<String, DispatchBatcher> modelMap : batchers.values()) {
            for (DispatchBatcher b : modelMap.values()) {
                if (b.cancelInQueue(requestId)) {
                    return;
                }
            }
        }
        InflightBatchRegistry.RequestEntry entry = inflightRegistry.lookupByRequest(requestId);
        if (entry == null) {
            Logger.debug("cancel({}) — no in-flight entry, ignoring", requestId);
            return;
        }
        InflightBatchRegistry.RequestState prev = inflightRegistry.markCancelled(requestId);
        if (prev == null || prev == InflightBatchRegistry.RequestState.CANCELLED) {
            return;
        }
        if (prev == InflightBatchRegistry.RequestState.PENDING_ACK) {
            Logger.info("Cancel arrived during PENDING_ACK for request {} (batch {}); "
                    + "tombstoned — handleAck will cascade after Engine ack",
                    requestId, entry.batchId());
            return;
        }
        try {
            grpcClient.cancelPrefill(entry.prefill().getServerIp(), entry.prefill().getGrpcPort(), requestId);
            grpcClient.cancelDecode(entry.decode().getServerIp(), entry.decode().getGrpcPort(), requestId);
        } catch (RuntimeException e) {
            Logger.warn("Cancel cascade failed for request {}", requestId, e);
        } finally {
            inflightRegistry.removeRequest(requestId);
        }
    }

    // ============== helpers ==============

    private void retryOrFail(PendingRequest req, Throwable cause) {
        if (req.future().isDone()) {
            return;
        }
        BalanceContext ctx = req.ctx();
        FlexlbConfig cfg = configService.loadBalanceConfig();

        int maxRetry = cfg.getMaxDpBatchRetryCount();
        long minBudget = cfg.getMinRetryBudgetMs();
        long elapsedMs = req.waitMicros() / 1000;
        long sloMs = cfg.getDpTtftSloMs();
        long remaining = sloMs - elapsedMs;

        if (ctx.getRetryCount() >= maxRetry || remaining < minBudget) {
            req.future().completeExceptionally(cause);
            return;
        }

        ctx.incrementRetryCount();
        String key = modelKey(ctx);
        ConcurrentHashMap<String, DispatchBatcher> modelBatchers =
                batchers.computeIfAbsent(key, k -> new ConcurrentHashMap<>());

        WorkerStatus prefillWorker = planner.selectPrefillWorker(
                key, cfg, ctx,
                ipPort -> {
                    DispatchBatcher b = modelBatchers.get(ipPort);
                    return b != null ? b.queueSize() : 0;
                });
        if (prefillWorker == null) {
            req.future().completeExceptionally(cause);
            return;
        }

        Logger.info("Retrying request {} (attempt {}), remaining budget {}ms",
                req.requestId(), ctx.getRetryCount(), remaining);

        QueuedRequest qr = QueuedRequest.forRetry(ctx, req.future(), req.enqueuedAtMicros());
        String batcherKey = prefillWorker.getIpPort();
        modelBatchers.compute(batcherKey, (k, existing) -> {
            if (existing == null || !existing.isAlive()) {
                return createBatcher(key, prefillWorker);
            }
            return existing;
        }).offer(qr);
    }

    private void failAll(DispatchBatch batch, Throwable cause) {
        for (PendingRequest r : batch.requests()) {
            retryOrFail(r, cause);
        }
    }

    private static String modelKey(BalanceContext ctx) {
        if (ctx == null || ctx.getRequest() == null || ctx.getRequest().getModel() == null) {
            return "";
        }
        return ctx.getRequest().getModel();
    }

    private EngineRpcService.BatchEnqueueRequestPB buildPb(long batchId,
                                                            List<List<PendingRequest>> ranked,
                                                            int dpSize) {
        EngineRpcService.BatchEnqueueRequestPB.Builder b = EngineRpcService.BatchEnqueueRequestPB.newBuilder()
                .setBatchId(batchId);

        for (int rank = 0; rank < ranked.size(); rank++) {
            List<PendingRequest> rankReqs = ranked.get(rank);
            if (rankReqs.isEmpty()) {
                b.addInputs(buildFakeInputPb(batchId, rank));
                continue;
            }
            for (PendingRequest req : rankReqs) {
                Request reqDto = req.ctx() != null ? req.ctx().getRequest() : null;
                String b64 = reqDto != null ? reqDto.getGenerateInputPbB64() : null;

                EngineRpcService.GenerateInputPB.Builder ib;
                if (b64 != null && !b64.isEmpty()) {
                    try {
                        byte[] bytes = java.util.Base64.getDecoder().decode(b64);
                        ib = EngineRpcService.GenerateInputPB.parseFrom(bytes).toBuilder();
                    } catch (com.google.protobuf.InvalidProtocolBufferException
                             | IllegalArgumentException e) {
                        Logger.error("Bad generate_input_pb_b64 for request {}; falling back to bare PB",
                                req.requestId(), e);
                        ib = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(req.requestId());
                    }
                } else {
                    ib = EngineRpcService.GenerateInputPB.newBuilder().setRequestId(req.requestId());
                }

                ib.setDpRank(com.google.protobuf.Int32Value.of(rank));
                ib.setBatchGroupId(com.google.protobuf.Int64Value.of(batchId));
                ib.setBatchGroupSize(rankReqs.size());
                if (reqDto != null && reqDto.getBlockCacheKeys() != null) {
                    ib.clearCacheHashKey().addAllCacheHashKey(reqDto.getBlockCacheKeys());
                }

                EngineRpcService.GenerateConfigPB.Builder gcb = ib.getGenerateConfigBuilder();
                gcb.setForceBatch(com.google.protobuf.Int32Value.of(1));
                gcb.setBatchGroupTimeout(com.google.protobuf.Int32Value.of(100));
                gcb.clearRoleAddrs();
                ServerStatus prefillSs = req.prefill();
                if (prefillSs != null) {
                    gcb.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                            .setRole(EngineRpcService.RoleAddrPB.RoleType.PREFILL)
                            .setIp(prefillSs.getServerIp())
                            .setHttpPort(prefillSs.getHttpPort())
                            .setGrpcPort(prefillSs.getGrpcPort())
                            .build());
                }
                ServerStatus decodeSs = req.decode();
                if (decodeSs != null) {
                    gcb.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                            .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                            .setIp(decodeSs.getServerIp())
                            .setHttpPort(decodeSs.getHttpPort())
                            .setGrpcPort(decodeSs.getGrpcPort())
                            .build());
                }
                b.addInputs(ib.build());
            }
        }

        return b.build();
    }

    private EngineRpcService.GenerateInputPB buildFakeInputPb(long batchId, int dpRank) {
        long fakeRequestId = -(batchId * 1024L + dpRank);
        EngineRpcService.GenerateInputPB.Builder ib = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(fakeRequestId)
                .setIsFakeQuery(true)
                .setDpRank(com.google.protobuf.Int32Value.of(dpRank))
                .setBatchGroupId(com.google.protobuf.Int64Value.of(batchId));
        ib.getGenerateConfigBuilder().setForceBatch(com.google.protobuf.Int32Value.of(1));
        return ib.build();
    }

    private void applyDpRankAddress(ServerStatus prefill, int dpRank) {
        if (prefill == null || dpRank < 0 || prefill.getServerIp() == null) {
            return;
        }
        String ipPort = prefill.getServerIp() + ":" + prefill.getHttpPort();
        Map<String, WorkerStatus> roleMap =
                engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, prefill.getGroup());
        if (roleMap == null) {
            return;
        }
        WorkerStatus ws = roleMap.get(ipPort);
        if (ws == null || ws.getDpStatuses() == null || ws.getDpStatuses().isEmpty()) {
            return;
        }
        for (DpRankStatus drs : ws.getDpStatuses()) {
            if (drs.dpRank() == dpRank) {
                prefill.setServerIp(drs.ip());
                prefill.setGrpcPort(drs.grpcPort());
                return;
            }
        }
        Logger.warn("applyDpRankAddress: dpRank {} not found in dpStatuses for {} (size={})",
                dpRank, ipPort, ws.getDpStatuses().size());
    }

    private static DebugInfo buildDebugInfo(PendingRequest req) {
        DebugInfo info = new DebugInfo();
        if (req.ctx() != null) {
            info.setHitCacheLen(req.ctx().getCacheMatchedTokens());
        }
        return info;
    }

    private static ServerStatus copyOf(ServerStatus src) {
        if (src == null) return null;
        ServerStatus s = new ServerStatus();
        s.setRole(src.getRole());
        s.setServerIp(src.getServerIp());
        s.setHttpPort(src.getHttpPort());
        s.setGrpcPort(src.getGrpcPort());
        s.setGroup(src.getGroup());
        s.setSuccess(src.isSuccess());
        s.setRequestId(src.getRequestId());
        s.setDpRank(src.getDpRank());
        s.setPrefillTime(src.getPrefillTime());
        return s;
    }

    @PreDestroy
    public void shutdown() {
        for (ConcurrentHashMap<String, DispatchBatcher> modelMap : batchers.values()) {
            for (DispatchBatcher b : modelMap.values()) {
                b.shutdown();
            }
        }
    }
}
