package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RequestWorkQueueTest {
    @Test
    @Timeout(10)
    void duplicateRunnerCannotExecuteTheNextSettlementConcurrently() throws Exception {
        var pool = new RequestTaskPool();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7101L);
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var next = new CountDownLatch(1);
        var driver = new AtomicReference<Runnable>();
        var workers = Executors.newFixedThreadPool(2);
        var dispatch = mock(ExecutorService.class);
        doAnswer(
                        call -> {
                            var task = call.getArgument(0, Runnable.class);
                            driver.set(task);
                            workers.execute(task);
                            return null;
                        })
                .when(dispatch)
                .execute(any());
        ReflectionTestUtils.setField(pool, "requestWorkers", dispatch);
        try {
            try (var accepted = pool.registerRequestWork(context)) {
                accepted.submitClose(
                        () -> {
                            entered.countDown();
                            RequestProtocolTestSupport.await(release);
                        },
                        false);
            }
            assertTrue(entered.await(5, TimeUnit.SECONDS));
            try (var accepted = pool.registerRequestWork(context)) {
                accepted.submitClose(next::countDown, false);
            }
            workers.submit(driver.get()).get(5, TimeUnit.SECONDS);
            assertEquals(
                    1, next.getCount(), "a fallback runner cannot pass the executing settlement");
            release.countDown();
            assertTrue(next.await(5, TimeUnit.SECONDS));
            pool.awaitIdle();
            pool.throwIfFailed();
        } finally {
            release.countDown();
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
            when(dispatch.isTerminated()).thenReturn(true);
            pool.close();
        }
    }

    @Test
    void failedStartRejectsCountedWorkAndAllowsTheNextNotification() {
        var pool = new RequestTaskPool();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7102L);
        var failure = new IllegalStateException("all executors rejected");
        var dispatch = mock(ExecutorService.class);
        doThrow(failure).when(dispatch).execute(any());
        when(dispatch.isTerminated()).thenReturn(true);
        ReflectionTestUtils.setField(pool, "requestWorkers", dispatch);
        var ran = new AtomicInteger();
        try {
            try (var accepted = pool.registerRequestWork(context)) {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> accepted.submitClose(ran::incrementAndGet, false)));
                assertTrue(accepted.canRetrySubmission());
                assertFalse(accepted.hasTaskOwner());
            }
            pool.awaitIdle();
            assertSame(failure, assertThrows(IllegalStateException.class, pool::throwIfFailed));
            assertEquals(0, ran.get());
            try (var accepted = pool.registerRequestWork(context)) {
                accepted.submitClose(ran::incrementAndGet, true);
            }
            pool.awaitIdle();
            assertEquals(1, ran.get(), "failed handoff cannot strand the request lane");
        } finally {
            pool.close();
        }
    }

    @Test
    void pendingNotificationsCoalesceAndAnIdleLaneDoesNotRetainRequestIdentity() {
        var pool = new RequestTaskPool();
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 7103L);
        var driver = new AtomicReference<Runnable>();
        var dispatch = mock(ExecutorService.class);
        doAnswer(
                        call -> {
                            driver.set(call.getArgument(0));
                            return null;
                        })
                .when(dispatch)
                .execute(any());
        when(dispatch.isTerminated()).thenReturn(true);
        ReflectionTestUtils.setField(pool, "requestWorkers", dispatch);
        var ran = new AtomicInteger();
        try {
            for (int i = 0; i < 101; i++) {
                try (var accepted = pool.registerRequestWork(context)) {
                    accepted.submitClose(ran::incrementAndGet, false);
                    assertEquals(i == 0, accepted.hasTaskOwner());
                }
            }
            driver.get().run();
            pool.awaitIdle();
            assertEquals(1, ran.get());
            var queue = ReflectionTestUtils.getField(pool, "requestWorkQueue");
            assertEquals(
                    0, ((java.util.Map<?, ?>) ReflectionTestUtils.getField(queue, "lanes")).size());
            try (var accepted = pool.registerRequestWork(context)) {
                accepted.submitClose(ran::incrementAndGet, true);
            }
            pool.awaitIdle();
            pool.throwIfFailed();
            assertEquals(2, ran.get());
        } finally {
            pool.close();
        }
    }
}
