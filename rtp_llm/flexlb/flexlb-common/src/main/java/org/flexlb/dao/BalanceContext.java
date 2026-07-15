package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.pv.ShortestTtftDecision;
import org.flexlb.dao.route.RoleType;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
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

    private final long startTimeNs = System.nanoTime();

    private long totalTimeUs;

    private long enqueueTime;

    private long dequeueTime;

    private long sequenceId;

    private long requestArrivalDelayMs;

    private long requestBodyReadAndDeserializeTimeUs;

    private long blockHashQueueWaitTimeUs;

    private long blockHashExecutionTimeUs;

    private long cacheMatchQueryTimeUs;

    private int cacheMatchQueryCount;

    private String cacheMatchSource;

    private final Map<RoleType, CacheMatchSelection> cacheMatchSelectionByRole = new EnumMap<>(RoleType.class);

    private Map<RoleType, ShortestTtftDecision> shortestTtftDecisionByRole;

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

    public void finishRequestTiming() {
        this.totalTimeUs = (System.nanoTime() - startTimeNs) / 1_000;
    }

    public void recordBlockHashTiming(long queueWaitTimeUs, long executionTimeUs) {
        this.blockHashQueueWaitTimeUs = queueWaitTimeUs;
        this.blockHashExecutionTimeUs = executionTimeUs;
    }

    public void recordCacheMatch(
            String source,
            long queryTimeUs,
            RoleType role,
            String selectedIp,
            long hitCacheTokens) {
        this.cacheMatchSource = source;
        this.cacheMatchQueryTimeUs += queryTimeUs;
        this.cacheMatchQueryCount++;
        this.cacheMatchSelectionByRole.put(
                role, new CacheMatchSelection(role, selectedIp, hitCacheTokens));
    }

    public void recordShortestTtftDecision(ShortestTtftDecision decision) {
        if (this.shortestTtftDecisionByRole == null) {
            this.shortestTtftDecisionByRole = new EnumMap<>(RoleType.class);
        }
        this.shortestTtftDecisionByRole.put(decision.role(), decision);
    }

    public Map<RoleType, ShortestTtftDecision> getShortestTtftDecisionByRole() {
        return this.shortestTtftDecisionByRole == null
                ? Collections.emptyMap()
                : this.shortestTtftDecisionByRole;
    }

    public record CacheMatchSelection(RoleType role, String selectedIp, long hitCacheTokens) {
    }
}
