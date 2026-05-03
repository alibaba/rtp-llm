package org.flexlb.balance.scheduler;

import org.flexlb.balance.dp.DispatchPlanner;
import org.flexlb.balance.dp.DpAssignStrategy;
import org.flexlb.balance.dp.GlobalPrefillBatcher;
import org.flexlb.balance.dp.InflightBatchRegistry;
import org.flexlb.balance.dp.PendingRequest;
import org.flexlb.balance.dp.PrefillBatch;
import org.flexlb.balance.dp.QueuedRequest;
import org.flexlb.balance.dp.RankAssignment;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.dp.DpGrpcClient;
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
 * V1 DP-batching orchestrator.
 *
 * <pre>
 *  Request                       Per-model
 *  ─► submit() ──► QueuedRequest ─► GlobalPrefillBatcher ──► drain ──► DispatchPlanner
 *                                                                       │
 *                                                                       ▼
 *                                                            PrefillBatch (group, requests, dpSize)
 *                                                                       │
 *                                                                       ▼
 *                                          dispatchBatch() ──► assign ranks ──► register inflight
 *                                                          ──► Master.Enqueue (async) ──► handleAck
 * </pre>
 *
 * <p><b>Group selection happens at drain time, not submit time.</b> A request
 * does not learn which DP group it will land on until {@code dpSize} other
 * requests have arrived (or the window timer fires). This decouples batching
 * efficiency from the ratio of QPS-to-group-count and shifts the RR cursor
 * from "per request" to "per batch" — the unit of fairness now matches the
 * unit of dispatch.
 */
@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DpBatchScheduler {

    private final ConfigService configService;
    private final EngineWorkerStatus engineWorkerStatus;
    private final DispatchPlanner planner;
    private final DpAssignStrategy assignStrategy;
    private final DpGrpcClient grpcClient;
    private final InflightBatchRegistry inflightRegistry;

    /** One batcher per model so different models never share a batch. */
    private final Map<String /*model*/, GlobalPrefillBatcher> batchers = new ConcurrentHashMap<>();

    private final ScheduledExecutorService timerExecutor = Executors.newScheduledThreadPool(2, r -> {
        Thread t = new Thread(r, "dp-batch-timer");
        t.setDaemon(true);
        return t;
    });

    private final AtomicLong batchIdGen = new AtomicLong(0);

    public DpBatchScheduler(ConfigService configService,
                            EngineWorkerStatus engineWorkerStatus,
                            DispatchPlanner planner,
                            DpAssignStrategy assignStrategy,
                            DpGrpcClient grpcClient,
                            InflightBatchRegistry inflightRegistry) {
        this.configService = configService;
        this.engineWorkerStatus = engineWorkerStatus;
        this.planner = planner;
        this.assignStrategy = assignStrategy;
        this.grpcClient = grpcClient;
        this.inflightRegistry = inflightRegistry;
    }

    // ============== Submit (called by RouteService) ==============

    /**
     * Enqueue a request into its model's global batcher. The returned future
     * completes when the batch is flushed and Master.Enqueue is acked, or
     * earlier if the request is failed/cancelled before dispatch.
     *
     * <p>Any exception escaping the offer path (config service throw, executor
     * rejection during shutdown, etc.) is converted into an exceptionally
     * completed future so callers never see a hung {@code CompletableFuture}.
     */
    public CompletableFuture<Response> submit(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        try {
            String key = modelKey(ctx);
            GlobalPrefillBatcher batcher = batchers.computeIfAbsent(key, this::newBatcher);
            batcher.offer(QueuedRequest.of(ctx, future));
        } catch (Throwable t) {
            Logger.error("DpBatchScheduler.submit threw before request was queued; "
                    + "failing future to avoid leaking a hung Mono", t);
            future.completeExceptionally(t);
        }
        return future;
    }

    private GlobalPrefillBatcher newBatcher(String model) {
        return new GlobalPrefillBatcher(model, configService, engineWorkerStatus,
                planner, this::dispatchBatch, timerExecutor);
    }

    // ============== Dispatch (called by GlobalPrefillBatcher) ==============

    /**
     * Take an assembled {@link PrefillBatch}, pin each request to a dp_rank,
     * register the inflight entries for cancel cascading, and fire the async
     * {@code Master.Enqueue} RPC.
     */
    void dispatchBatch(PrefillBatch batch) {
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

            // Cancel-before-ack tombstones still need engine-side cleanup IF the Engine
            // happened to receive the Enqueue (the failure could be a deadline that
            // fired AFTER engine started processing). Best-effort cascade for tombstones.
            for (PendingRequest r : batch.requests()) {
                if (inflightRegistry.getState(r.requestId()) == InflightBatchRegistry.RequestState.CANCELLED) {
                    cascadeEngineCancel(r);
                }
            }
            inflightRegistry.remove(batchId);
            failAll(batch, new RuntimeException("Master.Enqueue rejected: " + msg));
            return;
        }

        // Engine accepted Enqueue. Per-request: any request tombstoned during
        // PENDING_ACK must NOT be reported as success; instead, send Cancel to
        // engine NOW (engine just accepted the request and can act on it) and
        // fail the corresponding future. Other requests transition to ACTIVE
        // and complete successfully.
        for (RankAssignment ra : assignments) {
            PendingRequest req = ra.request();
            boolean activated = inflightRegistry.markActive(req.requestId());
            if (!activated) {
                Logger.info("request {} cancelled before Enqueue ack (batch {}); "
                        + "cascading Cancel to engine after-the-fact", req.requestId(), batchId);
                cascadeEngineCancel(req);
                req.future().completeExceptionally(
                        new java.util.concurrent.CancellationException(
                                "Cancelled before Master.Enqueue ack"));
                inflightRegistry.removeRequest(req.requestId());
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

        ServerStatus decode = copyOf(req.decode());
        decode.setRequestId(req.requestId());

        List<ServerStatus> list = new ArrayList<>(2);
        list.add(prefill);
        list.add(decode);
        resp.setServerStatus(list);
        return resp;
    }

    // ============== Cancel (called by RouteService.cancel / HTTP /rtp_llm/cancel) ==============

    /**
     * Cancel a request. Three lifecycles to cover:
     * <ol>
     *   <li><b>Still queued</b> (not yet drained): yank from the model's
     *       {@link GlobalPrefillBatcher}. {@code RouteService} completes the future
     *       exceptionally; we just stop the request from being dispatched.</li>
     *   <li><b>In flight</b> (registered, ack received, ACTIVE): cascade
     *       {@code Cancel} RPC to Prefill + Decode.</li>
     *   <li><b>In flight, ack pending</b> (PENDING_ACK): tombstone — {@link #handleAck}
     *       sees it and cascades after Engine actually has the request.</li>
     * </ol>
     * Re-entrant and tolerant of unknown ids (no-op).
     */
    public void cancel(long requestId) {
        // (1) try the queue first
        for (GlobalPrefillBatcher b : batchers.values()) {
            if (b.cancelInQueue(requestId)) {
                return;
            }
        }
        // (2,3) registry-based handling
        InflightBatchRegistry.RequestEntry entry = inflightRegistry.lookupByRequest(requestId);
        if (entry == null) {
            Logger.debug("cancel({}) — no in-flight entry, ignoring", requestId);
            return;
        }
        InflightBatchRegistry.RequestState prev = inflightRegistry.markCancelled(requestId);
        if (prev == null) {
            return;
        }
        if (prev == InflightBatchRegistry.RequestState.CANCELLED) {
            return;
        }
        if (prev == InflightBatchRegistry.RequestState.PENDING_ACK) {
            Logger.info("Cancel arrived during PENDING_ACK for request {} (batch {}); "
                    + "tombstoned — handleAck will cascade after Engine ack",
                    requestId, entry.batchId());
            return;
        }
        // ACTIVE: standard cascade
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

    private EngineRpcService.BatchGenerateInputPB buildPb(long batchId, List<RankAssignment> assignments) {
        EngineRpcService.BatchGenerateInputPB.Builder b = EngineRpcService.BatchGenerateInputPB.newBuilder()
                .setBatchId(batchId);
        for (RankAssignment ra : assignments) {
            PendingRequest req = ra.request();
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

    private static Response failureResponse(StrategyErrorType type, String detail) {
        Response r = new Response();
        r.setSuccess(false);
        r.setCode(type.getErrorCode());
        r.setErrorMessage(type.getErrorMsg() + (detail != null ? ": " + detail : ""));
        return r;
    }

    @PreDestroy
    public void shutdown() {
        timerExecutor.shutdownNow();
    }

    /** Test/observability. */
    public int batcherCount() {
        return batchers.size();
    }

    /** Test/observability. */
    public int totalQueueDepth() {
        int sum = 0;
        for (GlobalPrefillBatcher b : batchers.values()) {
            sum += b.queueSize();
        }
        return sum;
    }
}
