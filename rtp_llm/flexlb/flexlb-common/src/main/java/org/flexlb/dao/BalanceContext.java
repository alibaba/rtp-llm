package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;

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

    //======================== Queue ========================//

    private CompletableFuture<Response> future;

    private AtomicBoolean cancelled = new AtomicBoolean(false);

    private final AtomicInteger retryCount = new AtomicInteger(0);

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    private long enqueueTime;

    private long dequeueTime;

    private long sequenceId;

    private long requestArrivalDelayMs;

    private long requestBodyReadAndDeserializeTimeUs;

    private long blockHashQueueWaitTimeUs;

    private long blockHashExecutionTimeUs;

    private long kvcmQueryTimeUs;

    private int kvcmQueryCount;

    private boolean success = true;

    private String errorMessage;

    //===================== Method ===================//

    public String getRequestId() {
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

    public void recordRequestTiming(long requestTimeMs, long bodyReadAndDeserializeTimeUs) {
        if (requestTimeMs > 0) {
            this.requestArrivalDelayMs = startTime - requestTimeMs;
        }
        this.requestBodyReadAndDeserializeTimeUs = bodyReadAndDeserializeTimeUs;
    }

    public void recordBlockHashTiming(long queueWaitTimeUs, long executionTimeUs) {
        this.blockHashQueueWaitTimeUs = queueWaitTimeUs;
        this.blockHashExecutionTimeUs = executionTimeUs;
    }

    public void recordKvcmQuery(long queryTimeUs) {
        this.kvcmQueryTimeUs += queryTimeUs;
        this.kvcmQueryCount++;
    }
}
