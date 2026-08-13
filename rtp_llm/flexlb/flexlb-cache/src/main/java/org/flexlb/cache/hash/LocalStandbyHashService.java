package org.flexlb.cache.hash;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.LocalStandbyHashResult;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.constant.MetricConstant.LOCAL_STANDBY_HASH_EXECUTION_TIME_US;
import static org.flexlb.constant.MetricConstant.LOCAL_STANDBY_HASH_QUEUE_WAIT_TIME_US;
import static org.flexlb.constant.MetricConstant.LOCAL_STANDBY_HASH_RESULT;
import static org.flexlb.constant.MetricConstant.LOCAL_STANDBY_HASH_THREAD_POOL_INFO;

/**
 * Calculates the secondary Local Standby hashes without delaying the normal KVCM path.
 */
@Slf4j
@Component
public class LocalStandbyHashService {

    private static final FlexMetricTags SUCCESS_TAGS = FlexMetricTags.of("status", "success");
    private static final FlexMetricTags FAILURE_TAGS = FlexMetricTags.of("status", "failure");
    private static final FlexMetricTags REJECTED_TAGS = FlexMetricTags.of("status", "rejected");
    private static final long RESULT_RETENTION_SECONDS = 60;

    private final boolean enabled;
    private final FlexMonitor monitor;
    private final BlockHashStrategy blockHashStrategy;
    private final ThreadPoolExecutor executor;
    private final Cache<String, CompletableFuture<LocalStandbyHashResult>> tasksByRequestId;

    public LocalStandbyHashService(CacheMatchConfiguration configuration,
                                   FlexMonitor monitor,
                                   BlockHashStrategy blockHashStrategy) {
        LocalStandbyConfig config = configuration.getLocalStandbyConfig();
        this.enabled = configuration.isLocalStandbyEnabled();
        this.monitor = monitor;
        this.blockHashStrategy = blockHashStrategy;
        int threadCount = enabled ? config.getHashThreadCount() : LocalStandbyConfig.DEFAULT_HASH_THREAD_COUNT;
        int queueCapacity = enabled ? config.getHashQueueCapacity() : LocalStandbyConfig.DEFAULT_HASH_QUEUE_CAPACITY;
        AtomicInteger threadNumber = new AtomicInteger();
        this.executor = new ThreadPoolExecutor(
                threadCount,
                threadCount,
                0,
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(queueCapacity),
                runnable -> {
                    Thread thread = new Thread(
                            runnable,
                            "local-standby-hash-" + threadNumber.incrementAndGet());
                    thread.setDaemon(true);
                    thread.setPriority(Math.max(
                            Thread.MIN_PRIORITY,
                            Thread.NORM_PRIORITY - 1));
                    return thread;
                },
                new ThreadPoolExecutor.AbortPolicy());
        this.tasksByRequestId = Caffeine.newBuilder()
                .maximumSize(Math.max(1L, queueCapacity))
                .expireAfterAccess(RESULT_RETENTION_SECONDS, TimeUnit.SECONDS)
                .build();
    }

    @PostConstruct
    public void registerMetrics() {
        monitor.register(LOCAL_STANDBY_HASH_QUEUE_WAIT_TIME_US, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(LOCAL_STANDBY_HASH_EXECUTION_TIME_US, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(LOCAL_STANDBY_HASH_RESULT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(LOCAL_STANDBY_HASH_THREAD_POOL_INFO, FlexMetricType.GAUGE);
    }

    public CompletableFuture<LocalStandbyHashResult> submit(Request request, int[] inputIds, long blockSize, int lookaheadTokens) {
        if (!enabled) {
            return CompletableFuture.completedFuture(LocalStandbyHashResult.empty());
        }
        if (request == null || request.getRequestId() == null) {
            throw new IllegalArgumentException("request and requestId must not be null");
        }

        CompletableFuture<LocalStandbyHashResult> task = new CompletableFuture<>();
        CompletableFuture<LocalStandbyHashResult> existing =
                tasksByRequestId.asMap().putIfAbsent(request.getRequestId(), task);
        if (existing != null) {
            // Reuse the existing in-flight or recently completed task for the same request.
            return existing;
        }

        long submittedAt = System.nanoTime();
        try {
            executor.execute(() -> calculate(
                    request,
                    inputIds,
                    blockSize,
                    lookaheadTokens,
                    submittedAt,
                    task));
        } catch (RejectedExecutionException e) {
            log.warn("Local Standby hash queue is full, requestId={}", request.getRequestId());
            monitor.report(LOCAL_STANDBY_HASH_RESULT, REJECTED_TAGS, 1.0);
            complete(request, task, LocalStandbyHashResult.empty());
        }
        return task;
    }

    /**
     * Gets the available Local Standby hashes or the asynchronous task calculating them.
     */
    public CompletableFuture<LocalStandbyHashResult> getHashResult(String requestId,
                                                                   List<Long> existingBlockCacheKeys,
                                                                   long blockSize) {
        // Hashes are already available when the primary result is reused or the async task has completed.
        if (existingBlockCacheKeys != null) {
            return CompletableFuture.completedFuture(new LocalStandbyHashResult(existingBlockCacheKeys, blockSize));
        }

        // A different Local Standby block size is calculated asynchronously and indexed by request ID.
        CompletableFuture<LocalStandbyHashResult> task =
                requestId == null ? null : tasksByRequestId.getIfPresent(requestId);
        if (task != null) {
            return task;
        }

        // No hashes or registered calculation are available for this request.
        return CompletableFuture.completedFuture(LocalStandbyHashResult.empty());
    }

    private void calculate(Request request, int[] inputIds, long blockSize, int lookaheadTokens,
                           long submittedAt, CompletableFuture<LocalStandbyHashResult> task) {
        long startedAt = System.nanoTime();
        monitor.report(LOCAL_STANDBY_HASH_QUEUE_WAIT_TIME_US, (startedAt - submittedAt) / 1_000.0);
        try {
            List<Long> keys = blockHashStrategy.calculate(inputIds, blockSize, lookaheadTokens);
            request.setLocalStandbyCacheableBlockCacheKeys(
                    blockHashStrategy.cacheablePrefix(
                            keys, inputIds.length, blockSize, lookaheadTokens));
            complete(request, task, new LocalStandbyHashResult(keys, blockSize));
            monitor.report(LOCAL_STANDBY_HASH_RESULT, SUCCESS_TAGS, 1.0);
        } catch (RuntimeException e) {
            log.warn("Failed to calculate Local Standby hashes, requestId={}", request.getRequestId(), e);
            complete(request, task, LocalStandbyHashResult.empty());
            monitor.report(LOCAL_STANDBY_HASH_RESULT, FAILURE_TAGS, 1.0);
        } finally {
            monitor.report(LOCAL_STANDBY_HASH_EXECUTION_TIME_US, (System.nanoTime() - startedAt) / 1_000.0);
        }
    }

    private void complete(Request request, CompletableFuture<LocalStandbyHashResult> task, LocalStandbyHashResult result) {
        request.setLocalStandbyBlockCacheKeys(result.blockCacheKeys());
        if (result.blockCacheKeys().isEmpty()) {
            request.setLocalStandbyCacheableBlockCacheKeys(result.blockCacheKeys());
        }
        task.complete(result);
    }

    @Scheduled(fixedRate = 2000)
    void reportThreadPoolMetrics() {
        reportThreadPoolMetric("executingTaskThreadSize", executor.getActiveCount());
        reportThreadPoolMetric("queueSize", executor.getQueue().size());
        reportThreadPoolMetric("remainingQueueCapacity", executor.getQueue().remainingCapacity());
        reportThreadPoolMetric("threadPoolSize", executor.getPoolSize());
    }

    private void reportThreadPoolMetric(String type, int value) {
        monitor.report(LOCAL_STANDBY_HASH_THREAD_POOL_INFO, FlexMetricTags.of("type", type), value);
    }

    @PreDestroy
    public void shutdown() {
        executor.shutdown();
        tasksByRequestId.invalidateAll();
    }
}
