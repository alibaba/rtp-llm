package org.flexlb.dao.loadbalance;

import lombok.Data;
import org.flexlb.dao.BalanceContext;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 排队请求封装
 *
 * @author saichen.sm
 * @since 2025/12/23
 */
@Data
public class QueuedRequest {

    private final CompletableFuture<MasterResponse> future;
    private BalanceContext balanceContext;
    private final long enqueueTime;
    private final AtomicBoolean cancelled = new AtomicBoolean(false);

    // 重试相关字段
    private int retryCount = 0;

    public QueuedRequest(BalanceContext balanceContext, CompletableFuture<MasterResponse> future) {
        this.future = future;
        this.balanceContext = balanceContext;
        this.enqueueTime = System.currentTimeMillis();
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
