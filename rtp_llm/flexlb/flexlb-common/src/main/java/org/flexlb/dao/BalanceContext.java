package org.flexlb.dao;

import lombok.Data;
import lombok.ToString;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.trace.NoopSpanImpl;
import org.flexlb.trace.WhaleSpan;

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

    //===================== trace and log ===================//

    private static final WhaleSpan NOOP_SPAN = new NoopSpanImpl();

    private WhaleSpan span = NOOP_SPAN;

    private String otlpTraceParent;

    private String otlpTraceState;

    //===================== Method ===================//

    public String getRequestId() {
        return request.getRequestId();
    }

    /**
     * 标记请求为已取消
     */
    public void cancel() {
        cancelled.compareAndSet(false, true);
    }

    /**
     * 检查请求是否已取消
     */
    public boolean isCancelled() {
        return cancelled.get();
    }

    /**
     * 递增重试计数
     */
    public void incrementRetryCount() {
        retryCount++;
    }
}
