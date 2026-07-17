package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
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

    private AtomicBoolean cancelled = new AtomicBoolean(false);

    private final AtomicInteger retryCount = new AtomicInteger(0);

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    private long enqueueTime;

    private long dequeueTime;

    private long sequenceId;

    private boolean success = true;

    private String errorMessage;

    //===================== Decode Release ===================//

    /**
     * Callback to release the decode KV reservation for DIRECT/QUEUE paths.
     * Set by CostBasedDecodeStrategy.select() after a successful DECODE reserve.
     * Executed by RouteService.cancel() so each path releases its own reservation
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
     * Mark request as cancelled
     */
    public void cancel() {
        cancelled.compareAndSet(false, true);
    }

    /**
     * Check if request has been cancelled
     */
    public boolean isCancelled() {
        return cancelled.get();
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
