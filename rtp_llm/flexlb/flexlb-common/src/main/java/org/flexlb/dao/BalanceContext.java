package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author zjw
 * description:
 * date: 2025/3/11
 */
@Data
@ToString
public class BalanceContext {

    //======================== Basic =======================//

    private WhaleMasterConfig config;

    private Request request;

    private Response response;

    //======================== Queue ========================//

    private CompletableFuture<Response> future;

    private AtomicBoolean cancelled = new AtomicBoolean(false);

    private int retryCount = 0;

    //======================== Meters =======================//

    private long startTime = System.currentTimeMillis();

    private long enqueueTime;

    private long dequeueTime;

    private long sequenceId;

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
     */
    public void incrementRetryCount() {
        retryCount++;
    }
}
