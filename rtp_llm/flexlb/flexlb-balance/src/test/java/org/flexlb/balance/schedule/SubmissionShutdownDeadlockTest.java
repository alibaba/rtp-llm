package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

/**
 * Real submit/stop/shutdown paths; only endpoint selection and thread timing are isolated.
 */
class SubmissionShutdownDeadlockTest {
    @ParameterizedTest(name = "queued={0}")
    @ValueSource(booleans = {false, true})
    void stoppedSubmissionHoldingAnExternalLockMustNotWaitBehindDrain(boolean queued)
            throws Exception {
        try (Fixture f = new Fixture(queued)) {
            var repository = f.runtime.requests();
            var beforeAcceptingReturn = new CountDownLatch(1);
            var resumeAcceptingReturn = new CountDownLatch(1);
            var paused = new AtomicBoolean();
            var shutdownResult = new CompletableFuture<Void>();
            var externalLock = new ReentrantLock();
            var activeEntered = new CountDownLatch(1);
            var beginExternalWait = new CountDownLatch(1);
            var lateReturned = new CountDownLatch(1);
            var activeResult = new CompletableFuture<CompletableFuture<Response>>();
            var lateResult = new CompletableFuture<CompletableFuture<Response>>();
            var activeContext = f.context(951L);
            var lateContext = f.context(952L);

            Runnable waitForExternalLock =
                    () -> {
                        activeEntered.countDown();
                        try {
                            if (!beginExternalWait.await(5, TimeUnit.SECONDS)) {
                                throw new AssertionError(
                                        "active submission never reached the external lock");
                            }
                            // Interruptible only to release the reproduced cycle during test
                            // cleanup.
                            externalLock.lockInterruptibly();
                            externalLock.unlock();
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            throw new IllegalStateException(
                                    "release blocked submission during cleanup", interrupted);
                        }
                    };
            when(f.router.select(any(), any()))
                    .thenAnswer(
                            call -> {
                                if (!queued) {
                                    waitForExternalLock.run();
                                }
                                return RequestResult.rejected(
                                        Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
                            });
            Thread active =
                    Thread.ofPlatform()
                            .daemon()
                            .name("active-submission")
                            .unstarted(
                                    () -> {
                                        try {
                                            activeResult.complete(
                                                    queued
                                                            ? SchedulerTestServices.entry(
                                                                            f.scheduler)
                                                                    .submit(
                                                                            activeContext,
                                                                            waitForExternalLock)
                                                            : SchedulerTestServices.entry(
                                                                            f.scheduler)
                                                                    .submit(activeContext));
                                        } catch (Throwable failure) {
                                            activeResult.completeExceptionally(failure);
                                        }
                                    });
            Thread late =
                    Thread.ofPlatform()
                            .daemon()
                            .name("late-submission")
                            .unstarted(
                                    () -> {
                                        externalLock.lock();
                                        try {
                                            lateResult.complete(
                                                    SchedulerTestServices.entry(f.scheduler)
                                                            .submit(lateContext));
                                        } catch (Throwable failure) {
                                            lateResult.completeExceptionally(failure);
                                        } finally {
                                            externalLock.unlock();
                                            lateReturned.countDown();
                                        }
                                    });
            doAnswer(
                            call -> {
                                boolean accepting = (boolean) call.callRealMethod();
                                if (Thread.currentThread() == late
                                        && paused.compareAndSet(false, true)) {
                                    beforeAcceptingReturn.countDown();
                                    RequestProtocolTestSupport.await(resumeAcceptingReturn);
                                }
                                return accepting;
                            })
                    .when(repository)
                    .isStopping();
            Thread shutdown =
                    Thread.ofPlatform()
                            .daemon()
                            .name("runtime-shutdown")
                            .unstarted(
                                    () -> {
                                        try {
                                            SchedulerTestSupport.shutdown(f.runtime);
                                            shutdownResult.complete(null);
                                        } catch (Throwable failure) {
                                            shutdownResult.completeExceptionally(failure);
                                        }
                                    });
            try {
                active.start();
                assertTrue(activeEntered.await(3, TimeUnit.SECONDS));
                late.start();
                // Pause after the first intake check; registration must recheck the repository
                // gate.
                assertTrue(beforeAcceptingReturn.await(3, TimeUnit.SECONDS));
                assertNull(
                        lateContext.getRequestScheduler(),
                        "late submission has not registered yet");

                f.runtime.requests().stopAccepting();
                shutdown.start();
                RequestProtocolTestSupport.awaitCondition(() -> f.runtime.requests().isClosed());
                if (!queued) {
                    assertThrows(
                            java.util.concurrent.TimeoutException.class,
                            () -> shutdownResult.get(100, TimeUnit.MILLISECONDS),
                            "shutdown must await live DIRECT admission");
                }
                beginExternalWait.countDown();
                RequestProtocolTestSupport.awaitCondition(externalLock::hasQueuedThreads);
                resumeAcceptingReturn.countDown();

                assertTrue(
                        lateReturned.await(1, TimeUnit.SECONDS),
                        "stopped submit must reject and release the external lock; otherwise active submit "
                                + "waits for that lock, drain waits for active submit, and late submit waits behind drain");
                assertEquals(
                        StrategyErrorType.SCHEDULER_UNAVAILABLE.getErrorCode(),
                        lateResult.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS).getCode());
                assertNull(
                        lateContext.getRequestScheduler(),
                        "stopped submission must remain unregistered");
                activeResult.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS);
                shutdownResult.get(3, TimeUnit.SECONDS);

            } finally {
                // Break the cycle even on a red test, so Maven and fixture shutdown never hang.
                beginExternalWait.countDown();
                resumeAcceptingReturn.countDown();
                active.interrupt();
                active.join(3_000);
                late.join(3_000);
                shutdown.join(3_000);
                assertFalse(active.isAlive(), "active submission must be cleaned up");
                assertFalse(late.isAlive(), "late submission must be cleaned up");
            }
        }
    }

    /**
     * Starts a direct or queued scheduler with shared request services so submissions can race shutdown under an external lock.
     */
    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final RequestRouteSelector router = mock(RequestRouteSelector.class);
        final SchedulerTestServices runtime;
        final AbstractRequestScheduler scheduler;

        Fixture(boolean queued) {
            if (queued) {
                SchedulingTestConfig.useFifoQueue(config);
            } else {
                config.setScheduler(ConfigTestSupport.direct());
            }
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            runtime =
                    spy(
                            new SchedulerTestServices(
                                    spy(new RequestRepository()),
                                    mock(EndpointRegistry.class),
                                    reporter,
                                    mock(RequestSchedulerReporter.class),
                                    SchedulerTestSupport.dispatcher(),
                                    service,
                                    mock(EngineGrpcClient.class)));
            scheduler =
                    org.flexlb.balance.schedule.SchedulerTestSupport.createStartedScheduler(
                            runtime.newLifecycle(
                                    config,
                                    SchedulingTestConfig.capacity(config),
                                    org.mockito.Mockito.mock(
                                            org.flexlb.balance.requestlog.TheoryCacheHitReporter
                                                    .class)),
                            config,
                            router,
                            mock(DecodePreemption.class),
                            new CapacityNotifier(),
                            new OrderedRequestQueue(config.isPriorityOrdering()));
            runtime.registerScheduler(scheduler);
        }

        @Override
        public void close() {
            SchedulerTestSupport.shutdown(runtime);
        }

        RequestContext context(long id) {
            return RequestProtocolTestSupport.context(config, id);
        }
    }
}
