package org.flexlb.dao;

import com.google.protobuf.ByteString;
import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Data
@ToString
public class BalanceContext {

    //======================== Basic =======================//

    private FlexlbConfig config;

    private Request request;

    private Response response;

    @ToString.Exclude
    private ByteString generateInputPb;

    //======================== Queue ========================//

    private CompletableFuture<Response> future;

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    /** Monotonic timestamp captured when server-side request processing starts. */
    private long serviceStartNanos = System.nanoTime();

    /**
     * Timestamp (ms) when the request entered the gRPC server pipeline,
     * recorded by {@code GrpcServerTimingInterceptor}. Used to split the
     * total arrival delay into network delay and gRPC server processing time.
     * Remains 0 if the interceptor did not set it (e.g. non-gRPC code path).
     */
    private long grpcEntryTime;

    /** Monotonic counterpart of {@link #grpcEntryTime} for duration measurements. */
    private long grpcEntryNanos;

    /** Monotonic timestamp immediately before the request enters its worker batcher. */
    private long routeSubmittedNanos;

    /** Monotonic timestamp immediately before the batch is dispatched to the engine. */
    private long batchDispatchedNanos;

    private long enqueueTime;

    /**
     * Timestamp (ms) when the engine acknowledges the batch in BATCH mode.
     * Set when RequestScheduler confirms the EnqueueBatch acknowledgement.
     * Used to compute ack_to_response_time_ms in FlexlbServiceImpl.completeSchedule().
     * Remains 0 for non-BATCH paths or when ACK was not received.
     */
    private long ackAtMs;

    /** Monotonic counterpart of {@link #ackAtMs}. */
    private long ackAtNanos;

    private boolean success = true;

    private String errorMessage;

    //===================== Scheduling =================//

    /**
     * Immutable per-request scheduling metadata. The expiration is an
     * absolute caller-supplied Unix timestamp and is shared by DIRECT, QUEUE,
     * and BATCH without being reset on retry or rescue.
     */
    private SchedulingMetadata schedulingMetadata;

    /**
     * priority scheduling plan type that finally placed the request:
     * normal / prefill_evict / decode_evict. Empty when not applicable.
     */
    private String planType = "";

    /** Cost of the committed eviction plan; 0 for normal placement. */
    private long planCost;

    /** Victims preempted to place this request; 0 for normal placement. */
    private int victimCount;

    //===================== Method ===================//

    public long getRequestId() {
        return request.getRequestId();
    }

    /**
     * Normalized priority of the request (1-100, higher = more important).
     * Immutable scheduling metadata is authoritative; the request fallback
     * supports manually constructed internal contexts.
     */
    public int getPriority() {
        return schedulingMetadata != null
                ? schedulingMetadata.priority()
                : request.getPriority();
    }

    /**
     * Absolute request expiration timestamp in Unix epoch milliseconds.
     */
    public long getRequestExpiresAtMs() {
        if (schedulingMetadata != null) {
            return schedulingMetadata.expiresAtMs();
        }
        if (config != null && config.isQueue()) {
            return config.queueScheduler().resolveExpiresAtMs(startTime);
        }
        return Long.MAX_VALUE;
    }

    public boolean requestExpired(long nowMs) {
        long expiresAtMs = getRequestExpiresAtMs();
        return expiresAtMs <= 0 || nowMs >= expiresAtMs;
    }

    /** Direct accessor for immutable scheduling metadata. */
    public SchedulingMetadata schedulingMetadata() {
        return schedulingMetadata;
    }

}
