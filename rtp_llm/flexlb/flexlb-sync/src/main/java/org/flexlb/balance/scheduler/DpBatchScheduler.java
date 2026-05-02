package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.DpAssignStrategy;
import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.PrefillBatch;
import org.flexlb.balance.dp.PrefillQueue;
import org.flexlb.balance.dp.RankAssignment;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.dp.DpGrpcClient;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
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

/**
 * V1-α DP-batching path: receives single-request submissions from
 * {@link org.flexlb.service.RouteService}, buckets them into per-prefill-cluster
 * {@link PrefillQueue}s, and on flush:
 * <ol>
 *   <li>Assigns each request a {@code dp_rank} via {@link DpAssignStrategy} (RR for V1-α)</li>
 *   <li>Builds a {@link EngineRpcService.BatchGenerateInputPB} and fires {@code Master.Enqueue} async</li>
 *   <li>On Ack=accepted, completes each pending {@link CompletableFuture} with a
 *       {@link Response} carrying {@code enqueued_by_master=true} so frontend
 *       switches to {@code Decode.FetchResponse}</li>
 *   <li>On Ack=rejected or RPC failure, fails every future and rolls back local
 *       prefill cache reservation</li>
 * </ol>
 *
 * <p>Cancellation: {@link #cancel(long)} looks up the in-flight batch metadata in
 * {@link InflightBatchRegistry} and cascades a {@code Cancel} RPC to both the
 * Prefill leader and the Decode worker.
 *
 * <p>Concurrency: per-prefill-cluster queues are independent; flushBatch runs on the
 * shared timer executor (or the offer thread for size-trigger flushes). gRPC is async,
 * so neither path blocks. Per-batch ack arrives on the gRPC executor and fans out
 * sink completions there — keep that work tiny.
 */
@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DpBatchScheduler {

    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final DpAssignStrategy assignStrategy;
    private final DpGrpcClient grpcClient;
    private final InflightBatchRegistry inflightRegistry;

    /** key = "<model>|<prefillIp:port>" */
    private final Map<String, PrefillQueue> queues = new ConcurrentHashMap<>();

    private final ScheduledExecutorService timerExecutor = Executors.newScheduledThreadPool(2, r -> {
        Thread t = new Thread(r, "dp-batch-timer");
        t.setDaemon(true);
        return t;
    });

    private final AtomicLong batchIdGen = new AtomicLong(0);

    public DpBatchScheduler(ConfigService configService,
                            EngineWorkerStatus engineWorkerStatus,
                            DpAssignStrategy assignStrategy,
                            DpGrpcClient grpcClient,
                            InflightBatchRegistry inflightRegistry) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.assignStrategy = assignStrategy;
        this.grpcClient = grpcClient;
        this.inflightRegistry = inflightRegistry;
    }

    // ============== Submit (called by RouteService) ==============

    /**
     * Pick prefill + decode for {@code ctx}, queue the request for batch flush.
     * Returns a future that will be completed once the batch flushes (success
     * or failure).
     */
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        FlexlbConfig cfg = configService.loadBalanceConfig();
        CompletableFuture<Response> future = new CompletableFuture<>();

        // 1. Pick prefill via configured strategy (typically ShortestTTFT)
        LoadBalancer prefillSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                cfg.getStrategyForRoleType(RoleType.PREFILL));
        ServerStatus prefill = prefillSelector.select(ctx, RoleType.PREFILL, null);
        if (!prefill.isSuccess()) {
            future.complete(failure(StrategyErrorType.NO_PREFILL_WORKER, prefill.getMessage()));
            return future;
        }

        // 2. Pick decode in the same group (so KV transfer path is valid)
        LoadBalancer decodeSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                cfg.getStrategyForRoleType(RoleType.DECODE));
        ServerStatus decode = decodeSelector.select(ctx, RoleType.DECODE, prefill.getGroup());
        if (!decode.isSuccess()) {
            // Roll back prefill local-cache reservation made by ShortestTTFT
            prefillSelector.rollBack(ipPort(prefill), ctx.getRequestId());
            future.complete(failure(StrategyErrorType.NO_DECODE_WORKER, decode.getMessage()));
            return future;
        }

        // 3. Bucket into per-cluster queue
        int dpSize = resolveDpSize(cfg, prefill);
        if (dpSize <= 1) {
            // No DP barrier needed — degrade to immediate "single batch" flush.
            // RouteService should normally have caught this in shouldUseDpBatch,
            // but defend in depth.
            PendingRequest single = PendingRequest.of(ctx, prefill, decode, future);
            flushBatch(new PrefillBatch(prefill, List.of(single), 1));
            return future;
        }

        int batchSize = cfg.getDpBatchSizeMax() > 0 ? cfg.getDpBatchSizeMax() : dpSize;
        String key = queueKey(ctx, prefill);
        PrefillQueue queue = queues.computeIfAbsent(key, k ->
                new PrefillQueue(prefill, dpSize, batchSize,
                        cfg.getDpBatchWindowMs(),
                        cfg.getDpBatchTimeoutMs(),
                        timerExecutor,
                        this::flushBatch));
        queue.offer(PendingRequest.of(ctx, prefill, decode, future));
        return future;
    }

    // ============== Flush (called by PrefillQueue) ==============

    void flushBatch(PrefillBatch batch) {
        long batchId = batchIdGen.incrementAndGet();
        List<RankAssignment> assignments;
        try {
            assignments = assignStrategy.assign(batch);
        } catch (RuntimeException e) {
            Logger.error("DP rank assignment failed for batch of {} requests", batch.size(), e);
            failAll(batch, e);
            return;
        }

        EngineRpcService.BatchGenerateInputPB pb = buildPb(batchId, assignments);
        inflightRegistry.register(batchId, batch);

        grpcClient.enqueue(batch.prefillIp(), batch.prefillGrpcPort(), pb)
                .whenComplete((ack, err) -> handleAck(batch, batchId, assignments, ack, err));
    }

    private void handleAck(PrefillBatch batch, long batchId, List<RankAssignment> assignments,
                           EngineRpcService.EnqueueAckPB ack, Throwable err) {
        if (err != null || ack == null || !ack.getAccepted()) {
            String msg = err != null ? err.getMessage()
                    : (ack != null ? ack.getErrorMessage() : "no ack");
            Logger.warn("Master.Enqueue failed batch={} on {}:{} err={}",
                    batchId, batch.prefillIp(), batch.prefillGrpcPort(), msg);

            // Rollback local prefill cache reservation for every request in the batch
            FlexlbConfig cfg = configService.loadBalanceConfig();
            LoadBalancer prefillSelector = LoadBalanceStrategyFactory.getLoadBalancer(
                    cfg.getStrategyForRoleType(RoleType.PREFILL));
            for (PendingRequest r : batch.requests()) {
                try {
                    prefillSelector.rollBack(ipPort(r.prefill()), r.requestId());
                } catch (Throwable t) {
                    Logger.warn("rollBack threw for request {}", r.requestId(), t);
                }
            }
            inflightRegistry.remove(batchId);
            failAll(batch, new RuntimeException("Master.Enqueue rejected: " + msg));
            return;
        }

        // Success: complete each future with a Response carrying enqueued_by_master=true
        for (RankAssignment ra : assignments) {
            PendingRequest req = ra.request();
            Response resp = buildSuccessResponse(req, ra.dpRank());
            req.future().complete(resp);
        }
    }

    private Response buildSuccessResponse(PendingRequest req, int dpRank) {
        Response resp = new Response();
        resp.setSuccess(true);
        resp.setCode(200);
        resp.setEnqueuedByMaster(true);

        // Decorate the prefill ServerStatus with the assigned dp_rank
        ServerStatus prefill = copyOf(req.prefill());
        prefill.setDpRank(dpRank);
        prefill.setRequestId(req.requestId());

        ServerStatus decode = copyOf(req.decode());
        decode.setRequestId(req.requestId());

        List<ServerStatus> list = new ArrayList<>(2);
        list.add(prefill);
        list.add(decode);
        resp.setServerStatus(list);
        return resp;
    }

    // ============== Cancel (called by RouteService.cancel / HTTP /rtp_llm/cancel) ==============

    /** Cascade cancel to Prefill + Decode for one request. Best-effort, fire-and-forget. */
    public void cancel(long requestId) {
        InflightBatchRegistry.RequestEntry entry = inflightRegistry.lookupByRequest(requestId);
        if (entry == null) {
            Logger.debug("cancel({}) — no in-flight entry, ignoring", requestId);
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

    private static Response failure(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }

    private static String ipPort(ServerStatus s) {
        return s.getServerIp() + ":" + s.getHttpPort();
    }

    private static String queueKey(BalanceContext ctx, ServerStatus prefill) {
        String model = ctx.getRequest() != null && ctx.getRequest().getModel() != null
                ? ctx.getRequest().getModel() : "";
        return model + "|" + prefill.getServerIp() + ":" + prefill.getGrpcPort();
    }

    private int resolveDpSize(FlexlbConfig cfg, ServerStatus prefill) {
        if (cfg.getDpBatchSizeMax() > 0) {
            return cfg.getDpBatchSizeMax();
        }
        Map<String, WorkerStatus> map = engineWorkerStatus.selectModelWorkerStatus(
                RoleType.PREFILL, prefill.getGroup());
        if (map == null) {
            return 1;
        }
        WorkerStatus ws = map.get(ipPort(prefill));
        return ws != null && ws.getDpSize() > 0 ? (int) ws.getDpSize() : 1;
    }

    private static EngineRpcService.BatchGenerateInputPB buildPb(long batchId, List<RankAssignment> assignments) {
        EngineRpcService.BatchGenerateInputPB.Builder b = EngineRpcService.BatchGenerateInputPB.newBuilder().setBatchId(batchId);
        for (RankAssignment ra : assignments) {
            PendingRequest req = ra.request();
            // V1-α: opaque_input is not yet plumbed end-to-end (frontend → MasterRequest).
            // For now Master only conveys request_id + dp_rank + cache_hash_key (cache hash
            // wiring also TBD). C++ engine will obtain the full GenerateInputPB by other
            // means (existing GenerateStreamCall path during phase-in), or future MasterRequest
            // expansion will fill opaque_input.
            EngineRpcService.GenerateInputPB.Builder ib = EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(req.requestId())
                    .setDpRank(ra.dpRank());
            if (req.ctx() != null && req.ctx().getRequest() != null
                    && req.ctx().getRequest().getBlockCacheKeys() != null) {
                ib.addAllCacheHashKey(req.ctx().getRequest().getBlockCacheKeys());
            }
            b.addInputs(ib.build());
        }
        return b.build();
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
        timerExecutor.shutdownNow();
    }

    /** Test/observability hook. */
    public int queueCount() {
        return queues.size();
    }
}
