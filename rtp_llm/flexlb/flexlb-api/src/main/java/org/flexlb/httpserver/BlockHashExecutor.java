package org.flexlb.httpserver;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.flexlb.util.BlockCacheKeyCalculator;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.flexlb.constant.MetricConstant.BLOCK_HASH_EXECUTION_TIME_US;
import static org.flexlb.constant.MetricConstant.BLOCK_HASH_QUEUE_WAIT_TIME_US;
import static org.flexlb.constant.MetricConstant.BLOCK_HASH_RESULT;
import static org.flexlb.constant.MetricConstant.BLOCK_HASH_THREAD_POOL_INFO;

/**
 * Runs block hash calculations outside Reactor Netty event-loop threads.
 */
@Component
public class BlockHashExecutor {

    private static final FlexMetricTags SUCCESS_TAGS = FlexMetricTags.of("status", "success");
    private static final FlexMetricTags FAILURE_TAGS = FlexMetricTags.of("status", "failure");
    private static final FlexMetricTags REJECTED_TAGS = FlexMetricTags.of("status", "rejected");

    private final FlexMonitor monitor;
    private final ThreadPoolExecutor executor;
    private final Scheduler scheduler;

    public BlockHashExecutor(
            FlexMonitor monitor,
            @Value("${flexlb.block-hash.core-thread-count:8}") int coreThreadCount,
            @Value("${flexlb.block-hash.max-thread-count:32}") int maxThreadCount,
            @Value("${flexlb.block-hash.keep-alive-seconds:60}") long keepAliveSeconds,
            @Value("${flexlb.block-hash.queue-capacity:16384}") int queueCapacity) {
        if (coreThreadCount <= 0) {
            throw new IllegalArgumentException("flexlb.block-hash.core-thread-count must be positive");
        }
        if (maxThreadCount < coreThreadCount) {
            throw new IllegalArgumentException(
                    "flexlb.block-hash.max-thread-count must be at least core-thread-count");
        }
        if (keepAliveSeconds <= 0) {
            throw new IllegalArgumentException("flexlb.block-hash.keep-alive-seconds must be positive");
        }
        if (queueCapacity <= 0) {
            throw new IllegalArgumentException("flexlb.block-hash.queue-capacity must be positive");
        }

        this.monitor = monitor;
        this.executor = new ThreadPoolExecutor(
                coreThreadCount,
                maxThreadCount,
                keepAliveSeconds,
                TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(queueCapacity),
                new NamedThreadFactory("block-hash"),
                new ThreadPoolExecutor.AbortPolicy());
        this.scheduler = Schedulers.fromExecutor(executor);
    }

    @PostConstruct
    public void registerMetrics() {
        monitor.register(BLOCK_HASH_QUEUE_WAIT_TIME_US, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(BLOCK_HASH_EXECUTION_TIME_US, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        monitor.register(BLOCK_HASH_RESULT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(BLOCK_HASH_THREAD_POOL_INFO, FlexMetricType.GAUGE);
    }

    public Mono<BlockHashCalculationResult> calculate(int[] inputIds, long blockSize) {
        return calculate(inputIds, blockSize, 0);
    }

    public Mono<BlockHashCalculationResult> calculate(
            int[] inputIds,
            long blockSize,
            int lookaheadTokens) {
        return submitTimed(() -> BlockCacheKeyCalculator.calculate(
                        inputIds, blockSize, lookaheadTokens))
                .map(result -> new BlockHashCalculationResult(
                        result.value(),
                        result.queueWaitTimeUs(),
                        result.executionTimeUs()));
    }

    <T> Mono<T> submit(Callable<T> task) {
        return submitTimed(task).map(TimedTaskResult::value);
    }

    private <T> Mono<TimedTaskResult<T>> submitTimed(Callable<T> task) {
        return Mono.defer(() -> {
            long submittedAt = System.nanoTime();
            return Mono.fromCallable(() -> {
                        long startedAt = System.nanoTime();
                        long queueWaitTimeUs = (startedAt - submittedAt) / 1_000;
                        monitor.report(
                                BLOCK_HASH_QUEUE_WAIT_TIME_US,
                                queueWaitTimeUs);
                        try {
                            T value = task.call();
                            long executionTimeUs = (System.nanoTime() - startedAt) / 1_000;
                            return new TimedTaskResult<>(value, queueWaitTimeUs, executionTimeUs);
                        } finally {
                            monitor.report(
                                    BLOCK_HASH_EXECUTION_TIME_US,
                                    (System.nanoTime() - startedAt) / 1_000.0);
                        }
                    })
                    .subscribeOn(scheduler)
                    .doOnSuccess(ignored -> monitor.report(BLOCK_HASH_RESULT, SUCCESS_TAGS, 1.0))
                    .doOnError(error -> monitor.report(
                            BLOCK_HASH_RESULT,
                            error instanceof RejectedExecutionException ? REJECTED_TAGS : FAILURE_TAGS,
                            1.0))
                    // Keep downstream routing callbacks from occupying a block hash worker.
                    .publishOn(Schedulers.parallel());
        });
    }

    private record TimedTaskResult<T>(T value, long queueWaitTimeUs, long executionTimeUs) {
    }

    @Scheduled(fixedRate = 2000)
    void reportThreadPoolMetrics() {
        reportThreadPoolMetric("executingTaskThreadSize", executor.getActiveCount());
        reportThreadPoolMetric("queueSize", executor.getQueue().size());
        reportThreadPoolMetric("remainingQueueCapacity", executor.getQueue().remainingCapacity());
        reportThreadPoolMetric("corePoolSize", executor.getCorePoolSize());
        reportThreadPoolMetric("maxPoolSize", executor.getMaximumPoolSize());
        reportThreadPoolMetric("currentThreadSizeInPool", executor.getPoolSize());
        reportThreadPoolMetric("largestThreadSizeInPool", executor.getLargestPoolSize());
    }

    private void reportThreadPoolMetric(String type, int value) {
        monitor.report(BLOCK_HASH_THREAD_POOL_INFO, FlexMetricTags.of("type", type), value);
    }

    @PreDestroy
    public void shutdown() {
        scheduler.dispose();
        executor.shutdown();
    }
}
