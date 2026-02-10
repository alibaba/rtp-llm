package org.flexlb.balance.scheduler;

import lombok.Data;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.QueueSnapshot;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 请求排队管理器
 *
 * @author saichen.sm
 * @since 2025/12/22
 */
@Data
@Component
public class QueueManager {

    private static final String SNAPSHOT_DIR = "/tmp/flexlb-queue-snapshots";

    private final RoutingQueueReporter metrics;

    private final AtomicLong sequenceGenerator = new AtomicLong(0);

    // 请求队列
    private final BlockingDeque<BalanceContext> queue;

    public QueueManager(RoutingQueueReporter routingQueueReporter, ConfigService configService) {
        this.metrics = routingQueueReporter;
        this.queue = new LinkedBlockingDeque<>(configService.loadBalanceConfig().getMaxQueueSize());
    }

    /**
     * 尝试路由请求
     * <p>
     * 如果资源不足则排队并异步等待
     *
     * @param ctx 负载均衡上下文
     * @return 路由结果
     */
    public Mono<Response> tryRouteAsync(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        ctx.setFuture(future);

        // 放入队尾
        ctx.setEnqueueTime(System.currentTimeMillis());
        boolean added = queue.offerLast(ctx);
        if (!added) {
            Logger.warn("Queue is full for request id: {}, current size: {}", ctx.getRequestId(), queue.size());
            metrics.reportRejected();
            return Mono.just(Response.error(StrategyErrorType.QUEUE_FULL));
        }
        ctx.setSequenceId(sequenceGenerator.incrementAndGet());
        metrics.reportQueueEntry();

        return Mono.fromFuture(future)
                .timeout(Duration.ofMillis(ctx.getRequest().getGenerateTimeout()))
                .onErrorResume(e -> handleQueueException(ctx, e))
                .doFinally(signalType -> {
                    long routeExecutionTimeMs = System.currentTimeMillis() - ctx.getDequeueTime();
                    metrics.reportRouteExecutionMetric(routeExecutionTimeMs);
                });
    }

    /**
     * 放回队头（用于失败重试）
     *
     * @param ctx 负载均衡上下文
     */
    public void offerToHead(BalanceContext ctx) {
        queue.offerFirst(ctx);
    }

    /**
     * 从队列取出请求（阻塞/非阻塞）
     *
     * @param isBlock          是否阻塞等待
     * @param blockTimeoutMs   阻塞超时时间（毫秒）
     * @return 请求上下文，队列为空返回null
     */
    public BalanceContext takeRequest(boolean isBlock, long blockTimeoutMs) {
        return takeValidRequest(queue, isBlock, blockTimeoutMs);
    }

    /**
     * 从队列中取出单个有效请求
     * <p>
     * 会检查已取消和已超时的请求，对无效请求完成其 future
     *
     * @param sourceQueue 源队列
     * @return 请求上下文，队列为空返回null
     */
    private BalanceContext takeValidRequest(BlockingQueue<BalanceContext> sourceQueue, boolean isBlock, long blockTimeoutMs) {
        try {
            while (true) {
                BalanceContext ctx = isBlock ? sourceQueue.poll(blockTimeoutMs, TimeUnit.MILLISECONDS) : sourceQueue.poll();
                if (ctx == null) {
                    return null;
                }
                ctx.setDequeueTime(System.currentTimeMillis());
                if (ctx.isCancelled()) {
                    ctx.getFuture().completeExceptionally(new CancellationException("Request cancelled by client"));
                    continue;
                }
                long waitTimeMs = System.currentTimeMillis() - ctx.getEnqueueTime();
                long maxQueueWaitTimeMs = ctx.getRequest().getGenerateTimeout();
                if (waitTimeMs > maxQueueWaitTimeMs) {
                    ctx.getFuture().completeExceptionally(new TimeoutException("Request timeout in queue"));
                    continue;
                }
                long queueWaitTimeMs = ctx.getDequeueTime() - ctx.getEnqueueTime();
                metrics.reportQueueWaitingMetric(queueWaitTimeMs);
                return ctx;
            }
        } catch (Exception e) {
            Logger.error("Failed to take request from queue", e);
            return null;
        }
    }

    private void handleTimeout(BalanceContext ctx) {
        remove(ctx);
        metrics.reportTimeout();

        long waitTimeMs = System.currentTimeMillis() - ctx.getEnqueueTime();
        Logger.warn("Request timeout in queue for id: {}, wait time: {}ms", ctx.getRequestId(), waitTimeMs);
    }

    private void handleCanceled(BalanceContext ctx) {
        remove(ctx);
        metrics.reportCancelled();

        long waitTimeMs = System.currentTimeMillis() - ctx.getEnqueueTime();
        Logger.warn("Request canceled in queue for id: {}, wait time: {}ms", ctx.getRequestId(), waitTimeMs);
    }

    private void handleInterruption(BalanceContext ctx) {
        remove(ctx);
        Thread.currentThread().interrupt();
        Logger.error("Request interrupted while waiting in queue for id: {}", ctx.getRequestId());
    }

    private void remove(BalanceContext ctx) {
        boolean removed = queue.remove(ctx);
        if (!removed) {
            Logger.error("Failed to remove timeout request from queue:{}", ctx.getRequestId());
        }
    }

    private Mono<Response> handleQueueException(BalanceContext BalanceContext, Throwable e) {
        // Handle ExecutionException wrapper (consistent with synchronous version)
        Throwable cause = e instanceof ExecutionException ? e.getCause() : e;
        if (cause instanceof TimeoutException) {
            handleTimeout(BalanceContext);
            return Mono.just(Response.error(StrategyErrorType.QUEUE_TIMEOUT));
        } else if (cause instanceof CancellationException) {
            handleCanceled(BalanceContext);
            return Mono.just(Response.error(StrategyErrorType.REQUEST_CANCELLED));
        } else if (cause instanceof InterruptedException) {
            handleInterruption(BalanceContext);
            return Mono.just(Response.error(StrategyErrorType.QUEUE_TIMEOUT));
        }
<<<<<<< HEAD
        // 其他异常：记录日志并返回 NO_AVAILABLE_WORKER（与同步版本一致）
        Logger.error("Request execution failed for model: {}", modelName, e);
=======
        // Other exceptions: log and return NO_AVAILABLE_WORKER (consistent with synchronous version)
        Logger.error("Request execution failed error: {}", e);
>>>>>>> a09f8ce54 (feature - remove model name in request and worker_status_map)
        return Mono.just(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
    }

    @Scheduled(fixedRate = 1000)
    public void reportQueueSize() {
        metrics.reportQueueSize(queue.size());
    }

    public QueueSnapshotResponse snapshotQueue() {
        List<QueueSnapshot> snapshots = new ArrayList<>();
        long currentTime = System.currentTimeMillis();

        for (BalanceContext ctx : queue.toArray(new BalanceContext[0])) {
            QueueSnapshot snapshot = new QueueSnapshot();
            snapshot.setSequenceId(ctx.getSequenceId());
            snapshot.setRequestId(ctx.getRequestId());
            snapshot.setEnqueueTime(ctx.getEnqueueTime());
            snapshot.setWaitTimeMs(currentTime - ctx.getEnqueueTime());
            snapshot.setRetryCount(ctx.getRetryCount());
            snapshot.setQueueType("main");
            snapshots.add(snapshot);
        }

        try {
            Path dirPath = Paths.get(SNAPSHOT_DIR);
            if (!Files.exists(dirPath)) {
                Files.createDirectories(dirPath);
            }

            long timestamp = System.currentTimeMillis();
            String fileName = "queue-snapshot-" + timestamp + ".json";
            Path filePath = dirPath.resolve(fileName);

            String jsonContent = JsonUtils.toFormattedString(snapshots);
            Files.writeString(filePath, jsonContent);

            QueueSnapshotResponse response = new QueueSnapshotResponse();
            response.setFilePath(filePath.toAbsolutePath().toString());
            response.setTimestamp(timestamp);
            response.setCount(snapshots.size());

            return response;
        } catch (IOException e) {
            throw new RuntimeException("Failed to create queue snapshot", e);
        }
    }
}