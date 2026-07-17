package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

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
    private byte[] generateInputPbBytes;

    private volatile ScheduleModeEnum scheduleMode = ScheduleModeEnum.AUTO;

    //======================== Queue ========================//

    private CompletableFuture<Response> future;

    private final AtomicInteger retryCount = new AtomicInteger(0);

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

    private long dequeueTime;

    /**
     * Timestamp (ms) when the engine acknowledges the batch in BATCH mode.
     * Set by FlexlbBatchScheduler.onSuccess() when the ACK is received.
     * Used to compute ack_to_response_time_ms in FlexlbServiceImpl.completeSchedule().
     * Remains 0 for non-BATCH paths or when ACK was not received.
     */
    private long ackAtMs;

    /** Monotonic counterpart of {@link #ackAtMs}. */
    private long ackAtNanos;

    private long sequenceId;

    private boolean success = true;

    private String errorMessage;

    //===================== Cancellation ===================//

    /**
     * Cancel flag set by gRPC CancellationListener when the client disconnects.
     * Read by batcher algorithms (processQueue / pickFirstN) to skip cancelled
     * items and by flushItems to prevent dispatch of cancelled requests.
     */
    private final java.util.concurrent.atomic.AtomicBoolean cancelled =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    /** Mark request as cancelled. */
    public void cancel() {
        cancelled.compareAndSet(false, true);
    }

    /** Check if request has been cancelled. */
    public boolean isCancelled() {
        return cancelled.get();
    }

    //===================== Decode Release ===================//

    /**
     * Callback to release the decode KV reservation for DIRECT/QUEUE paths.
     * Set by CostBasedDecodeStrategy.select() after a successful DECODE reserve.
     * Executed by resource cleanup so each path releases its own reservation
     * directly, without going through FlexlbBatchScheduler.
     *
     * <p>Uses Runnable (not DecodeEndpoint) because BalanceContext lives in
     * flexlb-common which cannot depend on flexlb-sync classes.
     */
    @ToString.Exclude
    private volatile Runnable decodeReleaseCallback;

    //===================== Method ===================//

    public long getRequestId() {
        return request.getRequestId();
    }

    /**
     * Increment retry count
     * @return the new retry count after incrementing
     */
    public int incrementRetryCount() {
        return retryCount.incrementAndGet();
    }

    /**
     * Get current retry count
     */
    public int getRetryCount() {
        return retryCount.get();
    }
}
