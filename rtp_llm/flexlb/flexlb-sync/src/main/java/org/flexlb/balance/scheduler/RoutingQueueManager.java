package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.QueuedRequest;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.LoggingUtils;
import org.springframework.stereotype.Component;

import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * 请求排队管理器
 *
 * @author saichen.sm
 * @since 2025/12/22
 */
@Component
public class RoutingQueueManager {

    private final RoutingQueueReporter metrics;
    private final ConfigService configService;

    // 请求队列
    private final BlockingDeque<QueuedRequest> queue = new LinkedBlockingDeque<>();

    public RoutingQueueManager(RoutingQueueReporter routingQueueReporter, ConfigService configService) {
        this.metrics = routingQueueReporter;
        this.configService = configService;
    }

    /**
     * 尝试路由请求
     * <p>
     * 如果资源不足则排队并同步等待
     *
     * @param balanceContext 负载均衡上下文
     * @return 路由结果
     */
    public MasterResponse tryRoute(BalanceContext balanceContext) {
        return enqueueAndWait(balanceContext);
    }

    /**
     * 请求入队尾并同步等待路由结果
     *
     * @param balanceContext 负载均衡上下文
     * @return 路由结果
     */
    public MasterResponse enqueueAndWait(BalanceContext balanceContext) {
        String modelName = balanceContext.getMasterRequest().getModel();
        WhaleMasterConfig config = configService.loadBalanceConfig();

        // 检查队列容量
        if (queue.size() >= config.getMaxQueueSize()) {
            LoggingUtils.warn("Queue is full for request id: {}, size: {}", balanceContext.getInterRequestId(), queue.size());
            metrics.reportRejected();
            return MasterResponse.error(StrategyErrorType.QUEUE_FULL);
        }

        long enqueueTime = System.currentTimeMillis();
        CompletableFuture<MasterResponse> future = new CompletableFuture<>();
        QueuedRequest queuedRequest = new QueuedRequest(balanceContext, future);
        balanceContext.setQueuedRequest(queuedRequest);

        // 放入队尾
        boolean added = queue.offerLast(queuedRequest);
        if (!added) {
            LoggingUtils.warn("Failed to enqueue request id: {}", balanceContext.getInterRequestId());
            metrics.reportRejected();
            return MasterResponse.error(StrategyErrorType.QUEUE_FULL);
        }

        try {
            // 同步等待,直到路由成功或超时
            return future.get(config.getMaxQueueWaitTimeMs(), TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            handleTimeout(queue, queuedRequest);
            return MasterResponse.error(StrategyErrorType.QUEUE_TIMEOUT);
        } catch (InterruptedException e) {
            handleInterruption(queue, queuedRequest);
            return MasterResponse.error(StrategyErrorType.QUEUE_TIMEOUT);
        } catch (ExecutionException e) {
            // 检查是否是取消异常
            if (e.getCause() instanceof CancellationException) {
                handleCanceled(queue, queuedRequest);
                return MasterResponse.error(StrategyErrorType.REQUEST_CANCELLED);
            }
            // 检查是否是出队时检测到的超时异常
            if (e.getCause() instanceof TimeoutException) {
                handleTimeout(queue, queuedRequest);
                return MasterResponse.error(StrategyErrorType.QUEUE_TIMEOUT);
            }
            LoggingUtils.error("Request execution failed for model: {}", modelName, e);
            return MasterResponse.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        } finally {
            long waitTimeMs = System.currentTimeMillis() - enqueueTime;
            metrics.reportQueueMetric(queue.size(), waitTimeMs);
        }
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

        WhaleMasterConfig config = configService.loadBalanceConfig();
        long maxQueueWaitTimeMs = config.getMaxQueueWaitTimeMs();

        while (true) {
            QueuedRequest request = queue.takeFirst(); // 从队首出队

            // 检查是否已取消
            if (request.isCancelled()) {
                continue;
            }

            // 检查是否超时
            long waitTimeMs = System.currentTimeMillis() - request.getEnqueueTime();
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

        // 检查队列容量
        WhaleMasterConfig config = configService.loadBalanceConfig();
        if (queue.size() >= config.getMaxQueueSize()) {
            LoggingUtils.warn("Queue is full, cannot requeue request id: {}", request.getBalanceContext().getInterRequestId());
            return false;
        }

        // 重试请求放回插入队首，尽力保证 FIFO
        boolean added = queue.offerFirst(request);

        if (added) {
            LoggingUtils.info("Request requeued at front for retry, id: {}, retry count: {}/{}",
                    request.getBalanceContext().getInterRequestId(),
                    request.getRetryCount(),
                    configService.loadBalanceConfig().getMaxRetryCount());
        } else {
            LoggingUtils.error("Failed to requeue request at front for id: {}",
                    request.getBalanceContext().getInterRequestId());
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
}
