package org.flexlb.cache.hash;

import org.flexlb.cache.domain.BlockHashCalculationResult;
import org.flexlb.dao.loadbalance.TokenIds;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.core.Disposable;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.constant.MetricConstant.BLOCK_HASH_EXECUTION_TIME_US;
import static org.flexlb.constant.MetricConstant.BLOCK_HASH_QUEUE_WAIT_TIME_US;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.util.ReflectionTestUtils.getField;

class BlockHashExecutorTest {

    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private BlockHashExecutor executor;
    private BlockHashStrategy strategy;
    private final Map<TokenIds, Callable<?>> hashTasks = new ConcurrentHashMap<>();

    @BeforeEach
    void setUp() {
        strategy = spy(new VllmBlockHashStrategy());
        doAnswer(invocation -> {
            Callable<?> task = hashTasks.get(invocation.getArgument(0));
            if (task == null) {
                return invocation.callRealMethod();
            }
            task.call();
            return List.of(1L);
        }).when(strategy).calculate(any(TokenIds.class), eq(1L), eq(0));
        executor = new BlockHashExecutor(monitor, strategy, 1, 2, 60, 1);
    }

    @AfterEach
    void tearDown() {
        executor.shutdown();
    }

    @Test
    void allocatesResourcesOnlyWhenHashCalculationIsSubscribed() {
        Mono<BlockHashCalculationResult> calculation =
                executor.calculate(TokenIds.wrap(new int[]{1, 2, 3, 4}), 4, 0);
        executor.reportThreadPoolMetrics();

        assertNull(getField(executor, "executor"));
        assertNull(getField(executor, "scheduler"));
        verifyNoInteractions(monitor, strategy);

        assertNotNull(calculation.block(Duration.ofSeconds(5)));
        assertNotNull(getField(executor, "executor"));
        assertNotNull(getField(executor, "scheduler"));
    }

    @Test
    void shutdownBeforeFirstSubscriptionDoesNotAllocateResources() {
        Mono<BlockHashCalculationResult> calculation =
                executor.calculate(TokenIds.wrap(new int[]{1}), 1, 0);
        executor.shutdown();
        executor.shutdown();
        executor.reportThreadPoolMetrics();

        StepVerifier.create(calculation)
                .expectError(RejectedExecutionException.class)
                .verify(Duration.ofSeconds(5));
        assertNull(getField(executor, "executor"));
        assertNull(getField(executor, "scheduler"));
        verifyNoInteractions(strategy);
    }

    @Test
    void concurrentFirstSubscriptionsShareOneThreadPool() throws Exception {
        Set<Thread> workers = ConcurrentHashMap.newKeySet();
        BlockHashStrategy concurrentStrategy = mock(BlockHashStrategy.class);
        TokenIds inputIds = TokenIds.wrap(new int[]{1});
        when(concurrentStrategy.calculate(inputIds, 1, 0)).thenAnswer(invocation -> {
            workers.add(Thread.currentThread());
            return List.of(1L);
        });
        BlockHashExecutor concurrentExecutor =
                new BlockHashExecutor(monitor, concurrentStrategy, 1, 1, 60, 64);
        CountDownLatch start = new CountDownLatch(1);
        try (var callers = Executors.newFixedThreadPool(16)) {
            List<Future<BlockHashCalculationResult>> results = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                results.add(callers.submit(() -> {
                    assertTrue(start.await(5, TimeUnit.SECONDS));
                    return concurrentExecutor.calculate(inputIds, 1, 0).block(Duration.ofSeconds(5));
                }));
            }
            start.countDown();
            for (Future<BlockHashCalculationResult> result : results) {
                assertEquals(List.of(1L), result.get(10, TimeUnit.SECONDS).blockCacheKeys());
            }
            assertEquals(1, workers.size());
            concurrentExecutor.shutdown();
            assertThrows(RejectedExecutionException.class,
                    () -> concurrentExecutor.calculate(inputIds, 1, 0).block(Duration.ofSeconds(5)));
        } finally {
            start.countDown();
            concurrentExecutor.shutdown();
        }
    }

    @Test
    void initializedSchedulerDoesNotWaitForLifecycleMonitor() throws Exception {
        TokenIds inputIds = TokenIds.wrap(new int[]{1, 2, 3, 4});
        assertNotNull(executor.calculate(inputIds, 4, 0).block(Duration.ofSeconds(5)));

        try (var caller = Executors.newSingleThreadExecutor()) {
            synchronized (executor) {
                Future<BlockHashCalculationResult> result = caller.submit(
                        () -> executor.calculate(inputIds, 4, 0).block(Duration.ofSeconds(5)));
                assertEquals(List.of(2164874634404590027L),
                        result.get(5, TimeUnit.SECONDS).blockCacheKeys());
            }
        }
    }

    @Test
    void runsCpuTaskOnDedicatedThreadAndReportsLatency() {
        AtomicReference<String> threadName = new AtomicReference<>();
        submitHashTask(() -> {
            threadName.set(Thread.currentThread().getName());
            return null;
        }).block();

        assertTrue(threadName.get().startsWith("block-hash"));
        verify(monitor).report(eq(BLOCK_HASH_QUEUE_WAIT_TIME_US), anyDouble());
        verify(monitor).report(eq(BLOCK_HASH_EXECUTION_TIME_US), anyDouble());
    }

    @Test
    void returnsPerRequestHashTimings() {
        BlockHashCalculationResult result = executor.calculate(TokenIds.wrap(new int[]{1, 2, 3, 4}), 4, 0).block();

        assertNotNull(result);
        assertEquals(List.of(2164874634404590027L), result.blockCacheKeys());
        assertTrue(result.queueWaitTimeUs() >= 0);
        assertTrue(result.executionTimeUs() >= 0);
    }

    @Test
    void usesConfiguredBlockHashStrategy() {
        BlockHashStrategy strategy = mock(BlockHashStrategy.class);
        TokenIds inputIds = TokenIds.wrap(new int[]{1, 2, 3, 4, 5});
        when(strategy.calculate(inputIds, 4, 0))
                .thenReturn(List.of(11L, 22L));
        BlockHashExecutor configuredExecutor =
                new BlockHashExecutor(monitor, strategy, 1, 2, 60, 1);

        try {
            BlockHashCalculationResult result =
                    configuredExecutor.calculate(inputIds, 4, 0).block();

            assertNotNull(result);
            assertEquals(List.of(11L, 22L), result.blockCacheKeys());
            verify(strategy).calculate(inputIds, 4, 0);
        } finally {
            configuredExecutor.shutdown();
        }
    }

    @Test
    void calculatesSglangHashChainForCompletePages() {
        BlockHashExecutor sglangExecutor =
                new BlockHashExecutor(monitor, new SglangBlockHashStrategy(), 1, 2, 60, 1);

        try {
            BlockHashCalculationResult result =
                    sglangExecutor.calculate(TokenIds.wrap(new int[]{1, 2, 3, 4, 5}), 4, 0).block();

            assertNotNull(result);
            assertEquals(
                    List.of(-3488128144981237669L),
                    result.blockCacheKeys());
        } finally {
            sglangExecutor.shutdown();
        }
    }

    @Test
    void calculatesSglangEagleBigramHash() {
        BlockHashExecutor sglangExecutor =
                new BlockHashExecutor(monitor, new SglangBlockHashStrategy(), 1, 2, 60, 1);

        try {
            BlockHashCalculationResult result =
                    sglangExecutor.calculate(TokenIds.wrap(new int[]{1, 2, 3, 4, 5, 6}), 4, 1).block();

            assertNotNull(result);
            assertEquals(
                    List.of(-638950109823820341L),
                    result.blockCacheKeys());
        } finally {
            sglangExecutor.shutdown();
        }
    }

    @Test
    void rejectsWhenAllThreadsAndQueueSlotsAreOccupied() throws Exception {
        CountDownLatch firstTaskStarted = new CountDownLatch(1);
        CountDownLatch releaseFirstTask = new CountDownLatch(1);
        CountDownLatch queuedTaskCompleted = new CountDownLatch(1);
        AtomicReference<Throwable> backgroundError = new AtomicReference<>();

        Disposable runningTask = submitHashTask(() -> {
                    firstTaskStarted.countDown();
                    releaseFirstTask.await(5, TimeUnit.SECONDS);
                    return "running";
                })
                .subscribe(ignored -> { }, backgroundError::set);
        assertTrue(firstTaskStarted.await(5, TimeUnit.SECONDS));

        Disposable queuedTask = submitHashTask(() -> {
                    releaseFirstTask.await(5, TimeUnit.SECONDS);
                    return "queued";
                })
                .doFinally(ignored -> queuedTaskCompleted.countDown())
                .subscribe(ignored -> { }, backgroundError::set);

        Disposable secondRunningTask = submitHashTask(() -> {
                    releaseFirstTask.await(5, TimeUnit.SECONDS);
                    return "second-running";
                })
                .subscribe(ignored -> { }, backgroundError::set);

        StepVerifier.create(submitHashTask(() -> "rejected"))
                .expectError(RejectedExecutionException.class)
                .verify(Duration.ofSeconds(5));

        releaseFirstTask.countDown();
        assertTrue(queuedTaskCompleted.await(5, TimeUnit.SECONDS));
        assertNull(backgroundError.get());
        runningTask.dispose();
        queuedTask.dispose();
        secondRunningTask.dispose();
    }

    @Test
    void expandsBeyondCoreThreadsWhenTheQueueIsFull() throws Exception {
        CountDownLatch releaseCoreTask = new CountDownLatch(1);
        CountDownLatch coreTaskStarted = new CountDownLatch(1);
        CountDownLatch expandedTaskCompleted = new CountDownLatch(1);

        Disposable coreTask = submitHashTask(() -> {
                    coreTaskStarted.countDown();
                    releaseCoreTask.await(5, TimeUnit.SECONDS);
                    return "core";
                })
                .subscribe();
        assertTrue(coreTaskStarted.await(5, TimeUnit.SECONDS));
        Disposable queuedTask = submitHashTask(() -> "queued").subscribe();
        Disposable expandedTask = submitHashTask(() -> "expanded")
                .doFinally(ignored -> expandedTaskCompleted.countDown())
                .subscribe();

        try {
            assertTrue(expandedTaskCompleted.await(5, TimeUnit.SECONDS));
        } finally {
            releaseCoreTask.countDown();
            coreTask.dispose();
            queuedTask.dispose();
            expandedTask.dispose();
        }
    }
    private Mono<BlockHashCalculationResult> submitHashTask(Callable<?> task) {
        TokenIds inputIds = TokenIds.wrap(new int[]{1});
        hashTasks.put(inputIds, task);
        return executor.calculate(inputIds, 1, 0);
    }
}
