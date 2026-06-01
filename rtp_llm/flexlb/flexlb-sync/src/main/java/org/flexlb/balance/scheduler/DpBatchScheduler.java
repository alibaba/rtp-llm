package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.DispatchPlanner;
import org.flexlb.balance.dp.DpAssignStrategy;
import org.flexlb.balance.dp.GlobalPrefillBatcher;
import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.PrefillBatch;
import org.flexlb.balance.dp.PrefillTimePredictor;
import org.flexlb.balance.dp.QueuedRequest;
import org.flexlb.balance.dp.RankAssignment;
import org.flexlb.balance.dp.RoundRobinAssign;
import org.flexlb.balance.dp.SloBudgetBatcher;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
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
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DpBatchScheduler {

    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final DispatchPlanner planner;
    private final Map<String, DpAssignStrategy> assignStrategies;
    private final DpGrpcClient grpcClient;
    private final InflightBatchRegistry inflightRegistry;
    private final CacheAwareService cacheAwareService;
    private final PrefillTimePredictor prefillTimePredictor;
    private DpBatchReporter dpBatchReporter;

    private final Map<String, GlobalPrefillBatcher> batchers = new ConcurrentHashMap<>();
    private final Map<String, ConcurrentHashMap<String, SloBudgetBatcher>> sloBatchers = new ConcurrentHashMap<>();

    private final ScheduledExecutorService timerExecutor = Executors.newScheduledThreadPool(2, r -> {
        Thread t = new Thread(r, "dp-batch-timer");
        t.setDaemon(true);
        return t;
    });

    private final AtomicLong batchIdGen = new AtomicLong(0);

    public DpBatchScheduler(ConfigService configService,
                            EngineWorkerStatus engineWorkerStatus,
                            DispatchPlanner planner,
                            List<DpAssignStrategy> allAssignStrategies,
                            DpGrpcClient grpcClient,
                            InflightBatchRegistry inflightRegistry,
                            CacheAwareService cacheAwareService,
                            PrefillTimePredictor prefillTimePredictor) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.planner = planner;
        this.assignStrategies = allAssignStrategies.stream()
                .collect(Collectors.toMap(DpAssignStrategy::name, Function.identity()));
        this.grpcClient = grpcClient;
        this.inflightRegistry = inflightRegistry;
        this.cacheAwareService = cacheAwareService;
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
            int dpSize = peekModelDpSize();
            QueuedRequest qr = QueuedRequest.of(ctx, future);
            if (dpSize == 1) {
                WorkerStatus group = planner.selectPrefillWorker(key, configService.loadBalanceConfig(), 1);
                if (group == null) {
                    future.complete(failResponse("no available group"));
                    return future;
                }
                String groupKey = group.getGroup();
                ConcurrentHashMap<String, SloBudgetBatcher> modelBatchers =
                        sloBatchers.computeIfAbsent(key, k -> new ConcurrentHashMap<>());
                modelBatchers.computeIfAbsent(groupKey, k -> newSloBatcher(key, group)).offer(qr);
            } else {
                batchers.computeIfAbsent(key, this::newBatcher).offer(qr);
            }
        } catch (Throwable t) {
            Logger.error("DpBatchScheduler.submit threw before request was queued; "
                    + "failing future to avoid leaking a hung Mono", t);
            future.completeExceptionally(t);
        }
        return future;
    }

    private int peekModelDpSize() {
        Map<String, WorkerStatus> workers =
                engineWorkerStatus.selectModelWorkerStatus(RoleType.PREFILL, null);
        if (workers == null || workers.isEmpty()) {
            return 0;
        }
        for (WorkerStatus w : workers.values()) {
            if (w != null && w.isAlive()) {
                return (int) w.getDpSize();
            }
        }
        return 0;
    }

    private GlobalPrefillBatcher newBatcher(String model) {
        return new GlobalPrefillBatcher(model, configService, engineWorkerStatus,
                planner, this::dispatchBatch, timerExecutor, cacheAwareService, dpBatchReporter);
    }

    private SloBudgetBatcher newSloBatcher(String model, WorkerStatus prefillWorker) {
        SloBudgetBatcher b = new SloBudgetBatcher(model, configService, planner,
                this::dispatchBatch, prefillTimePredictor, dpBatchReporter, prefillWorker);
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

    // ============== Dispatch (called by GlobalPrefillBatcher) ==============

    void dispatchBatch(PrefillBatch batch) {
        long batchId = batchIdGen.incrementAndGet();

        String strategyName = configService.loadBalanceConfig().getDpAssignStrategy();
        DpAssignStrategy strategy = assignStrategies.getOrDefault(strategyName,
                assignStrategies.get(RoundRobinAssign.NAME));
        if (strategy == null) {
            strategy = assignStrategies.values().iterator().next();
        }

        List<RankAssignment> assignments;
        try {
            assignments = strategy.assign(batch);
        } catch (RuntimeException e) {
            Logger.error("DP rank assignment failed for batch of {} requests", batch.size(), e);
            failAll(batch, e);
            return;
        }

        EngineRpcService.BatchEnqueueRequestPB pb = buildPb(batchId, assignments, batch.dpSize());
        reportBatchComposition(assignments, batch.dpSize());
        inflightRegistry.register(batchId, batch);

        Logger.info("BatchDispatch batchId={} dpSize={} realRequests={} totalInputs={}",
                batchId, batch.dpSize(), assignments.size(), pb.getInputsCount());

        long deadlineMs = configService.loadBalanceConfig().getDpBatchEnqueueDeadlineMs();
        grpcClient.enqueue(batch.prefillIp(), batch.prefillGrpcPort(), pb, deadlineMs)
                .whenComplete((ack, err) -> handleAck(batch, batchId, assignments, ack, err));
    }

    private void reportBatchComposition(List<RankAssignment> assignments, int dpSize) {
        if (dpBatchReporter == null || dpSize <= 0) {
            return;
        }
        int[] requestsPerRank = new int[dpSize];
        for (RankAssignment ra : assignments) {
            if (ra.dpRank() >= 0 && ra.dpRank() < dpSize) {
                requestsPerRank[ra.dpRank()]++;
            }
        }
        int filledRanks = 0;
        for (int rank = 0; rank < dpSize; rank++) {
            if (requestsPerRank[rank] > 0) {
                dpBatchReporter.reportDpRankHit(rank);
                filledRanks++;
            }
        }
        dpBatchReporter.reportFakePadSlots(dpSize - filledRanks, dpSize);
    }

    /**
     * Periodic gauge scrape for InflightBatchRegistry health + active batcher counts.
     * Decoupled from inline reporting because these are steady-state observables
     * rather than per-event signals.
     */
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

    private void handleAck(PrefillBatch batch, long batchId, List<RankAssignment> assignments,
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

        for (RankAssignment ra : assignments) {
            PendingRequest req = ra.request();
            long reqId = req.requestId();
            EngineRpcService.EnqueueResponsePB slotAck = ackByReqId.get(reqId);

            if (slotAck == null) {
                Logger.warn("Master.BatchEnqueue returned no ack for request {} in batch {} on {}:{}",
                        reqId, batchId, batch.prefillIp(), batch.prefillGrpcPort());
                if (inflightRegistry.getState(reqId) == InflightBatchRegistry.RequestState.CANCELLED) {
                    cascadeEngineCancel(req);
                }
                req.future().completeExceptionally(
                        new RuntimeException("Master.BatchEnqueue missing per-slot ack for request " + reqId));
                inflightRegistry.removeRequest(reqId);
                continue;
            }

            long errorCode = slotAck.getErrorInfo().getErrorCode();
            if (errorCode != 0L) {
                String slotMsg = slotAck.getErrorInfo().getErrorMessage();
                Logger.warn("Master.Enqueue rejected slot req={} batch={} on {}:{} code={} msg={}",
                        reqId, batchId, batch.prefillIp(), batch.prefillGrpcPort(), errorCode, slotMsg);
                if (inflightRegistry.getState(reqId) == InflightBatchRegistry.RequestState.CANCELLED) {
                    cascadeEngineCancel(req);
                }
                req.future().completeExceptionally(
                        new RuntimeException("Master.Enqueue rejected slot: " + slotMsg));
                inflightRegistry.removeRequest(reqId);
                continue;
            }

            boolean activated = inflightRegistry.markActive(reqId);
            if (!activated) {
                Logger.info("request {} cancelled before Enqueue ack (batch {}); "
                        + "cascading Cancel to engine after-the-fact", reqId, batchId);
                cascadeEngineCancel(req);
                req.future().completeExceptionally(
                        new java.util.concurrent.CancellationException(
                                "Cancelled before Master.Enqueue ack"));
                inflightRegistry.removeRequest(reqId);
                continue;
            }
            req.future().complete(buildSuccessResponse(req, ra.dpRank()));
        }
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
        // Frontend's FetchResponse uses prefill.serverIp + prefill.grpcPort. The
        // request's response_registry entry lives on the DP that received the
        // real (non-fake) Enqueue slot — for any dpRank > 0 that is a peer DP,
        // not the BatchEnqueue receiver. Without this remap, frontend would
        // always hit DP0's registry and get NOT_FOUND for cross-DP requests.
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
        for (GlobalPrefillBatcher b : batchers.values()) {
            if (b.cancelInQueue(requestId)) {
                return;
            }
        }
        for (ConcurrentHashMap<String, SloBudgetBatcher> modelMap : sloBatchers.values()) {
            for (SloBudgetBatcher b : modelMap.values()) {
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

    private void failAll(PrefillBatch batch, Throwable cause) {
        for (PendingRequest r : batch.requests()) {
            r.future().completeExceptionally(cause);
        }
    }

    private static String modelKey(BalanceContext ctx) {
        if (ctx == null || ctx.getRequest() == null || ctx.getRequest().getModel() == null) {
            return "";
        }
        return ctx.getRequest().getModel();
    }

    private EngineRpcService.BatchEnqueueRequestPB buildPb(long batchId,
                                                            List<RankAssignment> assignments,
                                                            int dpSize) {
        EngineRpcService.BatchEnqueueRequestPB.Builder b = EngineRpcService.BatchEnqueueRequestPB.newBuilder()
                .setBatchId(batchId);

        int[] requestsPerRank = dpSize > 0 ? new int[dpSize] : null;
        if (requestsPerRank != null) {
            for (RankAssignment ra : assignments) {
                int rank = ra.dpRank();
                if (rank >= 0 && rank < dpSize) {
                    requestsPerRank[rank]++;
                }
            }
        }

        for (RankAssignment ra : assignments) {
            PendingRequest req = ra.request();
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

            int localGroupSize = (requestsPerRank != null && ra.dpRank() >= 0 && ra.dpRank() < dpSize)
                    ? requestsPerRank[ra.dpRank()] : assignments.size();

            ib.setDpRank(com.google.protobuf.Int32Value.of(ra.dpRank()));
            ib.setBatchGroupId(com.google.protobuf.Int64Value.of(batchId));
            ib.setBatchGroupSize(localGroupSize);
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

        if (requestsPerRank != null) {
            for (int rank = 0; rank < dpSize; rank++) {
                if (requestsPerRank[rank] > 0) {
                    continue;
                }
                b.addInputs(buildFakeInputPb(batchId, rank));
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
        for (ConcurrentHashMap<String, SloBudgetBatcher> modelMap : sloBatchers.values()) {
            for (SloBudgetBatcher b : modelMap.values()) {
                b.shutdown();
            }
        }
        timerExecutor.shutdownNow();
    }

}
