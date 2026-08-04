package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.QueueSnapshot;
import org.flexlb.dao.loadbalance.QueueSnapshotResponse;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.util.JsonUtils;
import org.flexlb.util.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Self-contained queueing component: a bounded FIFO request queue plus the
 * consumer worker pool that drains it.
 *
 * <p>Owned and composed by {@link QueueScheduler} — not a Spring bean; its
 * lifecycle ({@link #start()} / {@link #shutdown()}) follows the owning
 * scheduler. Absorbs the former QueueManager (queue container, snapshot dump)
 * and RequestScheduler (worker loop with permit-gated concurrency control).
 *
 * <p>Worker loop semantics (preserved from the original implementation):
 * <ol>
 *   <li>Acquire a concurrency permit from {@link DynamicWorkerManager}
 *       (500ms timeout — avoids indefinite blocking, allows shutdown checks)</li>
 *   <li>Take a request from the queue (500ms poll timeout); requests whose
 *       queue wait already exceeds their generate timeout are expired with a
 *       {@link TimeoutException} and skipped</li>
 *   <li>Hand the request to the consumer callback (routing + retry policy
 *       live in the owning scheduler)</li>
 *   <li>Release the permit in a {@code finally} block</li>
 * </ol>
 *
 * <p>Retry re-insertion goes through {@link #requeueHead} (queue-head
 * priority); the retry decision itself stays with the consumer.
 */
public class QueueingComponent {

    private static final String SNAPSHOT_DIR = "/tmp/flexlb-queue-snapshots";
    private static final int MAX_SNAPSHOT_FILES = 10;
    private static final long PERMIT_WAIT_MS = 500;
    private static final long TAKE_WAIT_MS = 500;

    private final ConfigService configService;
    private final RoutingQueueReporter metrics;
    private final DynamicWorkerManager dynamicWorkerManager;
    private final Consumer<BalanceContext> consumer;

    private final AtomicLong sequenceGenerator = new AtomicLong(0);
    private final BlockingDeque<BalanceContext> queue;

    private ExecutorService workerExecutor;
    private volatile boolean running = true;

    /**
     * @param configService        config source (queue capacity, worker count)
     * @param metrics              queue metric reporter
     * @param dynamicWorkerManager permit-based concurrency control (capacity
     *                             recalculation stays inside the manager)
     * @param consumer             dequeued-request handler; invoked on worker
     *                             threads with a permit held
     */
    public QueueingComponent(ConfigService configService,
                             RoutingQueueReporter metrics,
                             DynamicWorkerManager dynamicWorkerManager,
                             Consumer<BalanceContext> consumer) {
        this.configService = configService;
        this.metrics = metrics;
        this.dynamicWorkerManager = dynamicWorkerManager;
        this.consumer = consumer;
        this.queue = new LinkedBlockingDeque<>(configService.loadBalanceConfig().getMaxQueueSize());
    }

    // ==================== Lifecycle ====================

    /** Start the consumer worker pool. Call once before submitting requests. */
    public void start() {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.workerExecutor = Executors.newFixedThreadPool(config.getScheduleWorkerSize(), r -> {
            Thread t = new Thread(r, "routing-queue-worker");
            t.setDaemon(true);
            return t;
        });
        for (int i = 0; i < config.getScheduleWorkerSize(); i++) {
            workerExecutor.submit(this::workerLoop);
        }
        Logger.info("QueueingComponent worker pool started, worker count: {}", config.getScheduleWorkerSize());
    }

    public void shutdown() {
        running = false;
        if (workerExecutor != null && !workerExecutor.isShutdown()) {
            workerExecutor.shutdown();
            try {
                if (!workerExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    workerExecutor.shutdownNow();
                }
                Logger.info("QueueingComponent worker pool stopped");
            } catch (InterruptedException e) {
                workerExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    // ==================== Producer side ====================

    /**
     * Enqueue a request at the queue tail. Stamps enqueue time and sequence ID.
     *
     * @return {@code true} if enqueued; {@code false} when the queue is full
     *         (rejection metric reported, completion left to the caller)
     */
    public boolean enqueue(BalanceContext ctx) {
        ctx.setEnqueueTime(System.currentTimeMillis());
        ctx.setSequenceId(sequenceGenerator.incrementAndGet());
        boolean added = queue.offerLast(ctx);
        if (!added) {
            Logger.warn("Queue is full for request id: {}, current size: {}", ctx.getRequestId(), queue.size());
            metrics.reportRejected();
            return false;
        }
        metrics.reportQueueEntry();
        return true;
    }

    /**
     * Re-insert a request at the queue head (retry priority). If the queue is
     * full the request's future is completed with
     * {@link StrategyErrorType#QUEUE_FULL}.
     */
    public void requeueHead(BalanceContext ctx) {
        boolean added = queue.offerFirst(ctx);
        if (!added) {
            Logger.warn("Failed to re-queue request id: {} (queue full), completing with error", ctx.getRequestId());
            ctx.getFuture().complete(Response.error(StrategyErrorType.QUEUE_FULL));
        }
    }

    /**
     * Remove a request from the queue (timeout cleanup driven by the owning
     * scheduler's reactive pipeline).
     */
    public void remove(BalanceContext ctx) {
        boolean removed = queue.remove(ctx);
        if (!removed) {
            Logger.error("Failed to remove timeout request from queue:{}", ctx.getRequestId());
        }
    }

    public int queueSize() {
        return queue.size();
    }

    /** Current total concurrency permits (diagnostic view). */
    public int totalPermits() {
        return dynamicWorkerManager.getTotalPermits();
    }

    // ==================== Consumer side ====================

    /**
     * Worker thread main loop: permit → take → consume, both waits bounded
     * so shutdown is observed within one cycle.
     */
    private void workerLoop() {
        Logger.info("Worker thread started, ready to process requests...");

        while (running && !Thread.currentThread().isInterrupted()) {
            try {
                boolean acquired = dynamicWorkerManager.tryAcquirePermit(PERMIT_WAIT_MS, TimeUnit.MILLISECONDS);
                if (!acquired) {
                    continue;
                }

                try {
                    BalanceContext ctx = takeValidRequest(TAKE_WAIT_MS);
                    if (ctx == null) {
                        continue; // permit released in finally
                    }

                    Logger.debug("Worker processing request id: {}", ctx.getRequestId());
                    consumer.accept(ctx);
                } finally {
                    dynamicWorkerManager.releasePermit();
                }
            } catch (Exception e) {
                Logger.error("Worker thread encountered error", e);
            }
        }

        Logger.info("Worker thread stopped");
    }

    /**
     * Take a single valid request from the queue, waiting up to
     * {@code blockTimeoutMs}. Requests whose queue wait already exceeds their
     * generate timeout are expired in place ({@link TimeoutException} on the
     * raw future) and skipped.
     *
     * @return request context, or {@code null} when no valid request arrives
     *         before the timeout
     */
    private BalanceContext takeValidRequest(long blockTimeoutMs) {
        try {
            while (true) {
                BalanceContext ctx = queue.poll(blockTimeoutMs, TimeUnit.MILLISECONDS);
                if (ctx == null) {
                    return null;
                }
                ctx.setDequeueTime(System.currentTimeMillis());
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

    // ==================== Diagnostics ====================

    /** Report the current queue length gauge. Driven by the owning scheduler. */
    public void reportQueueSize() {
        metrics.reportQueueSize(queue.size());
    }

    /**
     * Dump a snapshot of the queued requests to a JSON file under
     * {@value #SNAPSHOT_DIR} (at most {@value #MAX_SNAPSHOT_FILES} files kept).
     */
    public QueueSnapshotResponse snapshot() {
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
