package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
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
@Component
public class QueueManager {

    private static final String SNAPSHOT_DIR = "/tmp/flexlb-queue-snapshots";
    private static final int MAX_SNAPSHOT_FILES = 10;

    private final RoutingQueueReporter metrics;

    private final EndpointRegistry endpointRegistry;

    private final AtomicLong sequenceGenerator = new AtomicLong(0);

    // Request queue
    private final BlockingDeque<BalanceContext> queue;

    public QueueManager(RoutingQueueReporter routingQueueReporter,
                        ConfigService configService,
                        EndpointRegistry endpointRegistry) {
        this.metrics = routingQueueReporter;
        this.endpointRegistry = endpointRegistry;
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
        ctx.setSequenceId(sequenceGenerator.incrementAndGet());
        boolean added = queue.offerLast(ctx);
        if (!added) {
            Logger.warn("Queue is full for request id: {}, current size: {}", ctx.getRequestId(), queue.size());
            metrics.reportRejected();
            return Mono.just(Response.error(StrategyErrorType.QUEUE_FULL));
        }
        metrics.reportQueueEntry();

        return Mono.fromFuture(future)
                .timeout(Duration.ofMillis(ctx.getRequest().getGenerateTimeout()))
                .onErrorResume(e -> handleQueueException(ctx, e))
                .doFinally(signalType -> {
                    if (ctx.getDequeueTime() > 0) {
                        long routeExecutionTimeMs = System.currentTimeMillis() - ctx.getDequeueTime();
                        metrics.reportRouteExecutionMetric(routeExecutionTimeMs);
                    }
                });
    }

    /**
     * Offer to queue head (for retry on failure)
     *
     * @param ctx Load balancing context
     */
    public void offerToHead(BalanceContext ctx) {
        boolean added = queue.offerFirst(ctx);
        if (!added) {
            Logger.warn("Failed to re-queue request id: {} (queue full), completing with error", ctx.getRequestId());
            ctx.getFuture().complete(Response.error(StrategyErrorType.QUEUE_FULL));
        }
    }

    public int queueSize() {
        return queue.size();
    }

    /**
     * Cancel a QUEUE-mode request: complete future exceptionally and release
     * decode KV reservation.
     *
     * <p>First completes the pending {@link CompletableFuture} exceptionally so
     * that the waiting caller unblocks immediately, then releases the decode KV
     * reservation via the callback recorded in {@link BalanceContext} by
     * CostBasedDecodeStrategy.select(). When the callback is not yet set (narrow
     * race window between reserve() and setCallback()), falls back to iterating
     * all decode endpoints — {@code ConcurrentHashMap.remove()} inside release()
     * is idempotent, so this is safe regardless of timing.
     *
     * @param ctx the balance context of the request to cancel
     */
    public void cancel(BalanceContext ctx) {
        CompletableFuture<Response> future = ctx.getFuture();
        if (future != null) {
            future.completeExceptionally(new CancellationException("Request cancelled by client"));
        }
        Runnable releaseCallback = ctx.getDecodeReleaseCallback();
        if (releaseCallback != null) {
            releaseCallback.run();
        } else {
            long rid = ctx.getRequestId();
            for (DecodeEndpoint ep : endpointRegistry.getDecodeEndpoints().values()) {
                ep.release(rid);
            }
        }
    }

    /**
     * Cancel a request by ID when no BalanceContext is available (fallback for
     * cancelByRequestId).
     *
     * <p>Since the mode is unknown, this brute-force releases across all
     * decode endpoints. {@code ConcurrentHashMap.remove()} is idempotent,
     * so this is safe even if the reservation was already released.
     *
     * @param requestId the request ID to cancel
     */
    public void cancelByRequestId(long requestId) {
        for (DecodeEndpoint ep : endpointRegistry.getDecodeEndpoints().values()) {
            ep.release(requestId);
        }
    }

    /**
     * Take request from the queue, waiting up to {@code blockTimeoutMs}.
     *
     * @return request context, or null when no request arrives before the timeout
     */
    public BalanceContext takeRequest(long blockTimeoutMs) {
        return takeValidRequest(queue, blockTimeoutMs);
    }

    /**
     * Take a single valid request from queue
     * <p>
     * Checks for cancelled and timed-out requests, completes future for invalid requests
     *
     * @param sourceQueue Source queue
     * @return Request context, null if queue is empty
     */
    private BalanceContext takeValidRequest(BlockingQueue<BalanceContext> sourceQueue, long blockTimeoutMs) {
        try {
            while (true) {
                BalanceContext ctx = sourceQueue.poll(blockTimeoutMs, TimeUnit.MILLISECONDS);
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

    @Scheduled(fixedRate = 2000L)
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

            // Clean up old snapshots, keep at most MAX_SNAPSHOT_FILES
            cleanOldSnapshots(dirPath);

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

    private void cleanOldSnapshots(Path dirPath) {
        try {
            List<Path> snapshotFiles = Files.list(dirPath)
                    .filter(p -> p.getFileName().toString().startsWith("queue-snapshot-"))
                    .sorted()
                    .collect(java.util.stream.Collectors.toList());
            // Keep at most MAX_SNAPSHOT_FILES - 1 so the new one makes it MAX_SNAPSHOT_FILES
            while (snapshotFiles.size() >= MAX_SNAPSHOT_FILES) {
                Files.deleteIfExists(snapshotFiles.remove(0));
            }
        } catch (IOException e) {
            Logger.warn("Failed to clean old queue snapshots: {}", e.getMessage());
        }
    }
}
