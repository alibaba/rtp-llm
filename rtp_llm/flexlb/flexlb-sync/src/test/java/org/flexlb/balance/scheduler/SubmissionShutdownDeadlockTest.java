package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/** Real submit/stop/shutdown paths; only endpoint selection and thread timing are isolated. */
class SubmissionShutdownDeadlockTest {
    @ParameterizedTest(name = "queued={0}")
    @ValueSource(booleans = {false, true})
    void stoppedSubmissionHoldingAnExternalLockMustNotWaitBehindDrain(boolean queued) throws Exception {
        try (Fixture f = new Fixture(queued)) {
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

            Runnable waitForExternalLock = () -> {
                activeEntered.countDown();
                try {
                    if (!beginExternalWait.await(5, TimeUnit.SECONDS)) {
                        throw new AssertionError("active submission never reached the external lock");
                    }
                    // Interruptible only to release the reproduced cycle during test cleanup.
                    externalLock.lockInterruptibly();
                    externalLock.unlock();
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("release blocked submission during cleanup", interrupted);
                }
            };
            when(f.router.select(any(), any())).thenAnswer(call -> {
                if (!queued) { waitForExternalLock.run(); }
                return PlacementResult.rejected(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED));
            });
            Thread active = Thread.ofPlatform().daemon().name("active-submission").unstarted(() -> {
                try {
                    activeResult.complete(queued ? f.scheduler.submit(activeContext, waitForExternalLock)
                            : f.scheduler.submit(activeContext));
                } catch (Throwable failure) { activeResult.completeExceptionally(failure); }
            });
            Thread late = Thread.ofPlatform().daemon().name("late-submission").unstarted(() -> {
                externalLock.lock();
                try { lateResult.complete(f.scheduler.submit(lateContext)); }
                catch (Throwable failure) { lateResult.completeExceptionally(failure); }
                finally {
                    externalLock.unlock();
                    lateReturned.countDown();
                }
            });
            doAnswer(call -> {
                boolean accepting = (boolean) call.callRealMethod();
                if (Thread.currentThread() == late && paused.compareAndSet(false, true)) {
                    beforeAcceptingReturn.countDown();
                    RequestProtocolTestSupport.await(resumeAcceptingReturn);
                }
                return accepting;
            }).when(f.runtime).isAccepting();
            Thread shutdown = Thread.ofPlatform().daemon().name("runtime-shutdown").unstarted(() -> {
                try { f.runtime.shutdown(); shutdownResult.complete(null); }
                catch (Throwable failure) { shutdownResult.completeExceptionally(failure); }
            });
            try {
                active.start();
                assertTrue(activeEntered.await(3, TimeUnit.SECONDS));
                late.start();
                // Pause after the first intake check; registration must recheck the Runtime gate.
                assertTrue(beforeAcceptingReturn.await(3, TimeUnit.SECONDS));
                assertNull(lateContext.scheduler(), "late submission has not registered yet");

                f.runtime.stopAccepting();
                shutdown.start();
                RequestProtocolTestSupport.awaitCondition(() -> f.runtime.requests().isClosed());
                if (!queued) {
                    assertThrows(java.util.concurrent.TimeoutException.class,
                            () -> shutdownResult.get(100, TimeUnit.MILLISECONDS), "shutdown must await live DIRECT admission");
                }
                beginExternalWait.countDown();
                RequestProtocolTestSupport.awaitCondition(() -> externalLock.hasQueuedThread(active));
                resumeAcceptingReturn.countDown();

                assertTrue(lateReturned.await(1, TimeUnit.SECONDS),
                        "stopped submit must reject and release the external lock; otherwise active submit "
                                + "waits for that lock, drain waits for active submit, and late submit waits behind drain");
                assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                        lateResult.get(3, TimeUnit.SECONDS).get(3, TimeUnit.SECONDS).getCode());
                assertNull(lateContext.scheduler(), "stopped submission must remain unregistered");
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

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        final SchedulerRuntime runtime;
        final AbstractRequestScheduler scheduler;

        Fixture(boolean queued) {
            if (queued) { SchedulingTestConfig.useFifoQueue(config); }
            else { config.setScheduler(SchedulerConfig.direct()); }
            SchedulingTestConfig.useNonBatchDispatcher(config);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            runtime = spy(new SchedulerRuntime(new RequestRepository(), mock(EndpointRegistry.class), reporter,
                    mock(RequestSchedulerReporter.class), mock(DefaultBatchDispatcher.class), service,
                    mock(RecentCacheKeyTraceReporter.class), mock(EngineCancelChannel.class)));
            scheduler = (AbstractRequestScheduler) PlacementConfiguration.create(runtime, config, router,
                    reporter, mock(DecodeCapacityAcquirer.class), new PlacementAvailability());
            runtime.initializeScheduler(scheduler);
        }
        RequestContext context(long id) { return RequestProtocolTestSupport.context(config, id); }
        @Override public void close() { runtime.shutdown(); }
    }
}
