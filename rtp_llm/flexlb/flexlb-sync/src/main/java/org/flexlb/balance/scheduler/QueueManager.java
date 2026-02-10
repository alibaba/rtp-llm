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
 * Request queue manager
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

    // Request queue
    private final BlockingDeque<BalanceContext> queue;

    public QueueManager(RoutingQueueReporter routingQueueReporter, ConfigService configService) {
        this.metrics = routingQueueReporter;
        this.queue = new LinkedBlockingDeque<>(configService.loadBalanceConfig().getMaxQueueSize());
    }

    /**
     * Attempt to route request
     * <p>
     * Queue and wait asynchronously if resources are insufficient
     *
     * @param ctx Load balancing context
     * @return Routing result
     */
    public Mono<Response> tryRouteAsync(BalanceContext ctx) {
        CompletableFuture<Response> future = new CompletableFuture<>();
        ctx.setFuture(future);

        // Add to queue tail
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
     * Offer to queue head (for retry on failure)
     *
     * @param ctx Load balancing context
     */
    public void offerToHead(BalanceContext ctx) {
        queue.offerFirst(ctx);
    }

    /**
     * Take request from queue (blocking/non-blocking)
     *
     * @param isBlock          Whether to block and wait
     * @param blockTimeoutMs   Block timeout in milliseconds
     * @return Request context, null if queue is empty
     */
    public BalanceContext takeRequest(boolean isBlock, long blockTimeoutMs) {
        return takeValidRequest(queue, isBlock, blockTimeoutMs);
    }

    /**
     * Take a single valid request from queue
     * <p>
     * Checks for cancelled and timed-out requests, completes future for invalid requests
     *
     * @param sourceQueue Source queue
     * @return Request context, null if queue is empty
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
        // Other exceptions: log and return NO_AVAILABLE_WORKER (consistent with synchronous version)
        Logger.error("Request execution failed error: {}", e);
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