package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A single inference request queued for batch dispatch.
 *
 * <p>Extracted from {@link BatchScheduler} to reduce coupling
 * with {@link WorkerBatcher}.
 *
 * <p>Carries direct {@link PrefillEndpoint} / {@link DecodeEndpoint} references
 * so downstream operations (commit, rollback, ack) avoid repeated
 * {@code EndpointRegistry} lookups by ip+port.
 *
 * <p>The item owns its own terminal transitions: success
 * ({@link #completeSuccess}) and the error paths ({@link #failExpired},
 * {@link #failOffer}, {@link #failDispatch}, {@link #failTimeout}) complete
 * the future and release EP-side resources without going through the
 * scheduler. All paths are idempotent: the future completes at most once,
 * the decode reservation is rolled back at most once ({@link #rolledBack}
 * CAS), and prefill batch repack uses {@code computeIfPresent}.
 *
 * <p>{@link #sortKey} is mutable — {@link FixedWindowBatcherAlgorithm} computes it
 * inside {@link WorkerBatcher#offer(BatchItem)} via {@link FixedWindowBatcherAlgorithm#computeSortKey}.
 */
public final class BatchItem {

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final Response routeResponse;
    private final ServerStatus prefill;
    private final ServerStatus decode;
    private final PrefillEndpoint prefillEp;
    private final DecodeEndpoint decodeEp;
    private final long enqueuedAtMs;

    /** Mutable sort key set by the batcher algorithm at offer time. */
    private volatile long sortKey;

    /**
     * Batch ID assigned in {@link PrefillEndpoint#submitBatch} just before
     * dispatch. Used for stale-ACK detection (the endpoint compares the
     * incoming batch ID against this field). 0 = not yet dispatched.
     */
    private volatile long assignedBatchId;

    /**
     * Wall-clock timestamp (ms) set just before the endpoint hands the batch
     * to the dispatch executor. Used for the dispatch-to-ACK latency metric.
     * 0 = not yet dispatched.
     */
    private volatile long dispatchedAtMs;

    /**
     * CAS flag ensuring the decode reservation is rolled back at most once
     * ({@link #rollbackOnce}).
     */
    final AtomicBoolean rolledBack = new AtomicBoolean(false);

    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     long enqueuedAtMs) {
        this.ctx = ctx;
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.decode = decode;
        this.prefillEp = prefillEp;
        this.decodeEp = decodeEp;
        this.enqueuedAtMs = enqueuedAtMs;
    }

    // -- accessors --

    public BalanceContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return future; }
    public Response routeResponse() { return routeResponse; }
    public ServerStatus prefill() { return prefill; }
    public ServerStatus decode() { return decode; }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public DecodeEndpoint decodeEp() { return decodeEp; }
    public long enqueuedAtMs() { return enqueuedAtMs; }

    /** Priority queue sort key. */
    public long sortKey() { return sortKey; }

    /** Set by {@link WorkerBatcher#offer} after {@link FixedWindowBatcherAlgorithm#computeSortKey}. */
    public void setSortKey(long sortKey) { this.sortKey = sortKey; }

    /** Batch ID assigned at dispatch time; 0 = not yet dispatched. */
    public long assignedBatchId() { return assignedBatchId; }

    /** Set by {@link PrefillEndpoint#submitBatch} before dispatch. */
    public void setAssignedBatchId(long assignedBatchId) { this.assignedBatchId = assignedBatchId; }

    /** Wall-clock dispatch timestamp (ms); 0 = not yet dispatched. */
    public long dispatchedAtMs() { return dispatchedAtMs; }

    /** Set by {@link PrefillEndpoint#submitBatch} just before async dispatch. */
    public void setDispatchedAtMs(long dispatchedAtMs) { this.dispatchedAtMs = dispatchedAtMs; }

    // -- derived accessors --

    public long requestId() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getRequestId() : 0;
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getSeqLen() : 0;
    }

    /** Cache-hit tokens on the assigned prefill endpoint. */
    public long hitCache() {
        return hitCacheOf(prefill);
    }

    /** Extract cache-hit length from a {@link ServerStatus} debug info. */
    private static long hitCacheOf(ServerStatus ss) {
        return ss != null && ss.getDebugInfo() != null
                ? ss.getDebugInfo().getHitCacheLen() : 0;
    }

    // ==================== Terminal transitions ====================

    /**
     * Complete the request successfully after the engine ACKed the enqueue.
     *
     * @param globalQueueLength current global inflight count, echoed to the client
     */
    public void completeSuccess(int globalQueueLength) {
        if (future.isDone()) {
            return;
        }
        Response success = copyResponse(routeResponse);
        success.setSuccess(true);
        success.setCode(200);
        success.setEnqueuedByMaster(true);
        success.setQueueLength(globalQueueLength);
        future.complete(success);
    }

    /**
     * Terminal path: the item expired in the batcher queue before dispatch
     * (queue deadline exceeded).
     */
    public void failExpired() {
        if (future.isDone()) {
            return;
        }
        rollbackOnce();
        removeFromPrefillBatch();
        completeError(StrategyErrorType.BATCH_SLO_EXPIRED,
                "batch SLO expired before dispatch");
    }

    /**
     * Terminal path: the batcher rejected the item at offer time (stopped,
     * queue full, oversized request, shutdown drain). The item was never
     * committed to a prefill batch, so no repack is needed.
     */
    public void failOffer(Throwable error) {
        if (future.isDone()) {
            return;
        }
        rollbackOnce();
        completeError(StrategyErrorType.BATCH_DISPATCH_FAILED,
                "Batcher offer failed: " + error.getMessage());
    }

    /** Terminal path: dispatch to the engine failed (build/reject/missing-ack/network). */
    public void failDispatch(Throwable error) {
        if (future.isDone()) {
            return;
        }
        rollbackOnce();
        removeFromPrefillBatch();
        completeError(StrategyErrorType.BATCH_DISPATCH_FAILED, error.getMessage());
    }

    /** Terminal path: the EnqueueBatch gRPC deadline elapsed before an acknowledgement. */
    public void failTimeout(Throwable error) {
        if (future.isDone()) {
            return;
        }
        rollbackOnce();
        removeFromPrefillBatch();
        completeError(StrategyErrorType.BATCH_SLO_EXPIRED,
                "EnqueueBatch deadline exceeded: " + error.getMessage());
    }

    /**
     * Release the decode reservation at most once (CAS-guarded).
     * Also invoked by the scheduler's {@code whenComplete} safety net;
     * overlapping with the explicit terminal paths is safe.
     */
    public void rollbackOnce() {
        if (rolledBack.compareAndSet(false, true)
                && decodeEp != null && decode != null) {
            decodeEp.release(decode.getRequestId());
        }
    }

    /**
     * Remove a failed or timed-out request from its prefill batch entry.
     * Uses {@link PrefillEndpoint#repackBatch} which:
     * <ul>
     *   <li>Single-request batch → removes the entire entry (batch becomes empty)</li>
     *   <li>Multi-request batch → keeps survivors, removes only this request</li>
     *   <li>Batch already removed (calibrate or releaseBatch ran first) → no-op</li>
     * </ul>
     * Safe to call multiple times (idempotent via ConcurrentHashMap.computeIfPresent).
     */
    public void removeFromPrefillBatch() {
        long batchId = assignedBatchId;
        if (batchId <= 0) {
            return;
        }
        if (prefillEp != null) {
            prefillEp.repackBatch(batchId, Set.of(requestId()));
            Logger.info("FlexLB remove from prefill batch: request_id={} batch_id={} engine={}",
                    requestId(), batchId, prefillEp.getIp());
        }
    }

    private void completeError(StrategyErrorType errorType, String message) {
        if (future.isDone()) {
            return;
        }
        Response errorResp = Response.error(errorType);
        errorResp.setErrorMessage(message == null ? errorType.getErrorMsg() : message);
        future.complete(errorResp);
    }

    // ==================== Static copy utilities ====================

    /** Deep-enough copy of a route response for per-request result isolation. */
    public static Response copyResponse(Response src) {
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

    public static ServerStatus copyOf(ServerStatus src) {
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
}
