package org.flexlb.httpserver;

import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.core.Disposable;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.constant.MetricConstant.BLOCK_HASH_EXECUTION_TIME_US;
import static org.flexlb.constant.MetricConstant.BLOCK_HASH_QUEUE_WAIT_TIME_US;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class BlockHashExecutorTest {

    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private BlockHashExecutor executor;

    @BeforeEach
    void setUp() {
        executor = new BlockHashExecutor(monitor, 1, 2, 60, 1);
    }

    @AfterEach
    void tearDown() {
        executor.shutdown();
    }

    @Test
    void runsCpuTaskOnDedicatedThreadAndReportsLatency() {
        String threadName = executor.submit(() -> Thread.currentThread().getName()).block();

        assertTrue(threadName.startsWith("block-hash"));
        verify(monitor).report(eq(BLOCK_HASH_QUEUE_WAIT_TIME_US), anyDouble());
        verify(monitor).report(eq(BLOCK_HASH_EXECUTION_TIME_US), anyDouble());
    }

    @Test
    void returnsPerRequestHashTimings() {
        BlockHashCalculationResult result = executor.calculate(List.of(1L, 2L, 3L, 4L), 4).block();

        assertNotNull(result);
        assertEquals(List.of(2164874634404590027L), result.blockCacheKeys());
        assertTrue(result.queueWaitTimeUs() >= 0);
        assertTrue(result.executionTimeUs() >= 0);
    }

    @Test
    void rejectsWhenAllThreadsAndQueueSlotsAreOccupied() throws Exception {
        CountDownLatch firstTaskStarted = new CountDownLatch(1);
        CountDownLatch releaseFirstTask = new CountDownLatch(1);
        CountDownLatch queuedTaskCompleted = new CountDownLatch(1);
        AtomicReference<Throwable> backgroundError = new AtomicReference<>();

        Disposable runningTask = executor.submit(() -> {
                    firstTaskStarted.countDown();
                    releaseFirstTask.await(5, TimeUnit.SECONDS);
                    return "running";
                })
                .subscribe(ignored -> { }, backgroundError::set);
        assertTrue(firstTaskStarted.await(5, TimeUnit.SECONDS));

        Disposable queuedTask = executor.submit(() -> {
                    releaseFirstTask.await(5, TimeUnit.SECONDS);
                    return "queued";
                })
                .doFinally(ignored -> queuedTaskCompleted.countDown())
                .subscribe(ignored -> { }, backgroundError::set);

        Disposable secondRunningTask = executor.submit(() -> {
                    releaseFirstTask.await(5, TimeUnit.SECONDS);
                    return "second-running";
                })
                .subscribe(ignored -> { }, backgroundError::set);

        StepVerifier.create(executor.submit(() -> "rejected"))
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

        Disposable coreTask = executor.submit(() -> {
                    coreTaskStarted.countDown();
                    releaseCoreTask.await(5, TimeUnit.SECONDS);
                    return "core";
                })
                .subscribe();
        assertTrue(coreTaskStarted.await(5, TimeUnit.SECONDS));
        Disposable queuedTask = executor.submit(() -> "queued").subscribe();
        Disposable expandedTask = executor.submit(() -> "expanded")
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
}
