package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.QueuedRequest;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.LoggingUtils;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeoutException;

import reactor.core.publisher.Mono;

/**
 * 请求排队管理器
 *
 * @author saichen.sm
 * @since 2025/12/22
 */
@Component
public class QueueManager {

    private final RoutingQueueReporter metrics;

    // 请求队列
    private static BlockingDeque<QueuedRequest> queue = null;

    public QueueManager(RoutingQueueReporter routingQueueReporter, ConfigService configService) {
        this.metrics = routingQueueReporter;
        queue = new LinkedBlockingDeque<>(configService.loadBalanceConfig().getMaxQueueSize());
    }

    /**
     * 尝试路由请求（异步版本）
     * <p>
     * 如果资源不足则排队并异步等待
     *
     * @param balanceContext 负载均衡上下文
     * @return 路由结果
     */
    public Mono<MasterResponse> tryRouteAsync(BalanceContext balanceContext) {
        String modelName = balanceContext.getMasterRequest().getModel();

        CompletableFuture<MasterResponse> future = new CompletableFuture<>();
        QueuedRequest queuedRequest = new QueuedRequest(balanceContext, future);
        balanceContext.setQueuedRequest(queuedRequest);

        // 放入队尾
        boolean added = queue.offerLast(queuedRequest);
        if (!added) {
            LoggingUtils.warn("Queue is full for request id: {}, current size: {}", balanceContext.getInterRequestId(), queue.size());
            metrics.reportRejected();
            return Mono.just(MasterResponse.error(StrategyErrorType.QUEUE_FULL));
        }

        return Mono.fromFuture(future)
                .timeout(Duration.ofMillis(balanceContext.getMasterRequest().getGenerateTimeout()))
                .onErrorResume(e -> handleQueueException(modelName, queue, queuedRequest, e))
                .doFinally(signalType -> {
                    long waitTimeMs = System.currentTimeMillis() - queuedRequest.getEnqueueTime();
                    metrics.reportQueueWaitingMetric(waitTimeMs);
                });
    }

    /**
     * 请求队首出队，若队列为空会阻塞等待
     * <p>
     * 会检查已取消和已超时的请求
     *
     * @return 队首请求
     * @throws InterruptedException 如果等待被中断
     */
    public QueuedRequest dequeue() throws InterruptedException {

        while (true) {
            QueuedRequest request = queue.takeFirst(); // 从队首出队

            // 检查是否已取消
            if (request.isCancelled()) {
                continue;
            }

            // 检查是否超时
            long waitTimeMs = System.currentTimeMillis() - request.getEnqueueTime();
            long maxQueueWaitTimeMs = request.getBalanceContext().getMasterRequest().getGenerateTimeout();
            if (waitTimeMs > maxQueueWaitTimeMs) {
                request.getFuture().completeExceptionally(new TimeoutException("Request timeout in queue"));
                continue;
            }

            return request;
        }
    }

    /**
     * 在队首重新入队
     *
     * @param request 需要重新入队的请求
     * @return 是否成功入队
     */
    public boolean reEnqueueAtFront(QueuedRequest request) {
        // 重试请求放回插入队首，尽力保证 FIFO
        boolean added = queue.offerFirst(request);

        if (added) {
            LoggingUtils.info("Request requeued at front for retry, id: {}, retry count: {}",
                    request.getBalanceContext().getInterRequestId(),
                    request.getRetryCount());
        } else {
            LoggingUtils.warn("Queue is full, cannot requeue request id: {}, current size: {}",
                    request.getBalanceContext().getInterRequestId(), queue.size());
        }

        return added;
    }

    /**
     * 取消指定的请求
     *
     * @param balanceContext 负载均衡上下文
     */
    public void cancelRequest(BalanceContext balanceContext) {
        QueuedRequest queuedRequest = balanceContext.getQueuedRequest();
        queuedRequest.cancel();
        queuedRequest.getFuture().completeExceptionally(new CancellationException("Request cancelled by client"));
    }

    private void handleTimeout(BlockingQueue<QueuedRequest> queue, QueuedRequest queuedRequest) {
        boolean removed = queue.remove(queuedRequest);
        if (!removed) {
            LoggingUtils.error("Failed to remove timeout request:{} from queue", queuedRequest.getBalanceContext().getInterRequestId());
        }
        metrics.reportTimeout();

        long waitTimeMs = System.currentTimeMillis() - queuedRequest.getEnqueueTime();
        LoggingUtils.warn("Request timeout in queue for id: {}, wait time: {}ms", queuedRequest.getBalanceContext().getInterRequestId(), waitTimeMs);
    }

    private void handleCanceled(BlockingQueue<QueuedRequest> queue, QueuedRequest queuedRequest) {
        boolean removed = queue.remove(queuedRequest);
        if (!removed) {
            LoggingUtils.error("Failed to remove cancelled request from queue for id: {}", queuedRequest.getBalanceContext().getInterRequestId());
        }
        metrics.reportCancelled();

        long waitTimeMs = System.currentTimeMillis() - queuedRequest.getEnqueueTime();
        LoggingUtils.warn("Request canceled in queue for id: {}, wait time: {}ms", queuedRequest.getBalanceContext().getInterRequestId(), waitTimeMs);
    }

    private void handleInterruption(BlockingQueue<QueuedRequest> queue, QueuedRequest queuedRequest) {
        boolean removed = queue.remove(queuedRequest);
        if (!removed) {
            LoggingUtils.error("Failed to remove interrupted request from queue for id: {}", queuedRequest.getBalanceContext().getInterRequestId());
        }
        Thread.currentThread().interrupt();
        LoggingUtils.error("Request interrupted while waiting in queue for id: {}", queuedRequest.getBalanceContext().getInterRequestId());
    }

    private Mono<MasterResponse> handleQueueException(String modelName, BlockingQueue<QueuedRequest> queue,
                                                       QueuedRequest queuedRequest, Throwable e) {
        // 处理 ExecutionException 包装的异常（与同步版本保持一致）
        Throwable cause = e instanceof ExecutionException ? e.getCause() : e;
        if (cause instanceof TimeoutException) {
            handleTimeout(queue, queuedRequest);
            return Mono.just(MasterResponse.error(StrategyErrorType.QUEUE_TIMEOUT));
        } else if (cause instanceof CancellationException) {
            handleCanceled(queue, queuedRequest);
            return Mono.just(MasterResponse.error(StrategyErrorType.REQUEST_CANCELLED));
        } else if (cause instanceof InterruptedException) {
            handleInterruption(queue, queuedRequest);
            return Mono.just(MasterResponse.error(StrategyErrorType.QUEUE_TIMEOUT));
        }
        // 其他异常：记录日志并返回 NO_AVAILABLE_WORKER（与同步版本一致）
        LoggingUtils.error("Request execution failed for model: {}", modelName, e);
        return Mono.just(MasterResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER));
    }

    public static long getQueueSize() {
        return queue == null ? 0 : queue.size();
    }
}
