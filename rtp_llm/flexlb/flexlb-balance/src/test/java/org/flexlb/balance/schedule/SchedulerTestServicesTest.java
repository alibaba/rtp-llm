package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.ResponseCompletionExecutor.Registration;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.scheduler.SchedulerConfig;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks shared test-runtime startup, request ownership, and shutdown draining through real lifecycle components.
 */
class SchedulerTestServicesTest {
    @Test
    void drainEscalatesToImmediateAndFreezesAdmissionAfterItsDeadline() {
        var config = SchedulingTestConfig.newConfig();
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var runtime =
                new SchedulerTestServices(
                        new RequestRepository(),
                        mock(EndpointRegistry.class),
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        SchedulerTestSupport.dispatcher(),
                        service,
                        mock(EngineGrpcClient.class));
        var scheduler = mock(AbstractRequestScheduler.class);
        var pending = RequestProtocolTestSupport.context(config, 88001L);
        var owner =
                runtime.newLifecycle(
                        config,
                        SchedulingTestConfig.capacity(config),
                        mock(TheoryCacheHitReporter.class));
        scheduler = spy(owner);
        scheduler.submit(pending, () -> {});
        runtime.registerScheduler(scheduler);

        ShutdownResult result = runtime.shutdown(ShutdownMode.DRAIN, Duration.ZERO);

        assertFalse(result.drained());
        assertTrue(result.forced());
        verify(scheduler).close();
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void deferredCloseCannotBlockItsRequestWorkSource() throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9111L);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            var sourceReturned = new java.util.concurrent.CompletableFuture<Void>();
            try {
                f.runtime
                        .tasks()
                        .execute(
                                () -> {
                                    {
                                        SchedulerTestSupport.submitClose(
                                                f.runtime.tasks,
                                                context,
                                                () -> {
                                                    entered.countDown();
                                                    RequestProtocolTestSupport.await(release);
                                                },
                                                false,
                                                null,
                                                false);
                                    }
                                    sourceReturned.complete(null);
                                });
                assertTrue(entered.await(2, java.util.concurrent.TimeUnit.SECONDS));
                sourceReturned.get(2, java.util.concurrent.TimeUnit.SECONDS);
            } finally {
                release.countDown();
                SchedulerTestSupport.awaitRequestWork(f.runtime);
            }
        }
    }

    @Test
    void executorErrorTransfersWorkOnceAndDoesNotLeakItsDrainOwner() throws Exception {
        var f = new Fixture();
        var workers = SchedulerTestSupport.requestWorkers(f.runtime);
        var rejected = mock(java.util.concurrent.ExecutorService.class);
        var error = new AssertionError("executor failed before accepting work");
        doThrow(error).when(rejected).execute(any(Runnable.class));
        var executions = new java.util.concurrent.atomic.AtomicInteger();
        try {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    f.runtime.tasks(), "requestWorkers", rejected);
            assertThrows(
                    AssertionError.class,
                    () ->
                            f.runtime
                                    .tasks()
                                    .execute(
                                            () -> {
                                                executions.incrementAndGet();
                                            }));
            assertTimeoutPreemptively(
                    java.time.Duration.ofSeconds(5),
                    () -> SchedulerTestSupport.awaitRequestWork(f.runtime));
            assertEquals(0, executions.get());
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    f.runtime.tasks(), "requestWorkers", workers);
            assertDoesNotThrow(
                    () -> f.runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO));
        }
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void externalFutureCancellationRemainsSynchronousAndBlocksResponseExecutorClose()
            throws Exception {
        try (var f = new Fixture(false, false)) {
            var scheduler =
                    SchedulerTestSupport.lifecycle(
                            SchedulerTestSupport.initializedScheduler(f.runtime));
            var context = RequestProtocolTestSupport.context(f.config, 9103L);
            var result = RequestProtocolTestSupport.submitForManualPlacement(scheduler, context);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            var cancelled = new java.util.concurrent.CompletableFuture<Boolean>();
            var callback =
                    result.whenComplete(
                            (response, failure) -> {
                                assertFalse(Thread.holdsLock(context));
                                entered.countDown();
                                RequestProtocolTestSupport.await(release);
                            });
            var source =
                    new Thread(
                            () -> {
                                try {
                                    cancelled.complete(result.cancel(false));
                                } catch (Throwable failure) {
                                    cancelled.completeExceptionally(failure);
                                }
                            });
            var closer =
                    new Thread(() -> SchedulerTestSupport.responseCompletions(f.runtime).close());
            try {
                source.start();
                assertTrue(entered.await(5, java.util.concurrent.TimeUnit.SECONDS));
                assertTrue(
                        result.isCancelled(),
                        "Future.cancel selects and publishes its result on the caller");
                assertFalse(
                        cancelled.isDone(),
                        "synchronous dependent callbacks still belong to that caller");
                closer.start();
                RequestProtocolTestSupport.awaitCondition(
                        () -> closer.getState() == Thread.State.WAITING);
                assertTrue(
                        closer.isAlive(), "response executor waits for the synchronous callback");
            } finally {
                release.countDown();
                source.join(5_000);
                closer.join(5_000);
            }
            assertTrue(cancelled.get(5, java.util.concurrent.TimeUnit.SECONDS));
            assertTrue(callback.isCompletedExceptionally());
            assertFalse(closer.isAlive());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void directStartupUsesOneSchedulerAndRejectsQueueRequests() {
        try (var f = new Fixture(false)) {
            var scheduler =
                    SchedulerTestSupport.lifecycle(
                            SchedulerTestSupport.initializedScheduler(f.runtime));
            assertInstanceOf(
                    DirectRequestScheduler.class,
                    org.flexlb.balance.schedule.SchedulerTestServices.scheduler(scheduler));
            assertSame(
                    org.flexlb.balance.schedule.SchedulerTestServices.scheduler(scheduler),
                    SchedulerTestSupport.initializedScheduler(f.runtime));
            var wrongMode = new RequestContext(SchedulingTestConfig.newConfig());
            wrongMode.setRequest(
                    org.flexlb.worker.endpoint.Request.builder()
                            .requestId("991")
                            .seqLen(1L)
                            .build());

            wrongMode.setSchedulingMetadata(
                    org.flexlb.balance.schedule.SchedulingMetadata.explicit(50, Long.MAX_VALUE));
            var rejected =
                    SchedulerTestServices.entry(
                                    (AbstractRequestScheduler)
                                            org.flexlb.balance.schedule.SchedulerTestServices
                                                    .scheduler(scheduler))
                            .submit(wrongMode)
                            .join();
            assertFalse(rejected.isSuccess());
            assertEquals(
                    "request configuration does not match scheduler mode",
                    rejected.getErrorMessage());
            assertEquals(0, f.runtime.requests().liveRequestCount());
            assertSame(f.config, new RequestContext(f.config).getConfig());
            assertThrows(
                    IllegalStateException.class,
                    () -> f.runtime.registerScheduler(mock(AbstractRequestScheduler.class)));
        }
    }

    @Test
    void queueStartupUsesOneScheduler() {
        try (var f = new Fixture(true)) {
            assertInstanceOf(
                    QueuedRequestScheduler.class,
                    SchedulerTestSupport.initializedScheduler(f.runtime));
            assertSame(
                    SchedulerTestSupport.initializedScheduler(f.runtime),
                    SchedulerTestSupport.initializedScheduler(f.runtime));
            assertThrows(
                    IllegalStateException.class,
                    () -> f.runtime.registerScheduler(mock(AbstractRequestScheduler.class)));
        }
    }

    @Test
    void schedulerCannotBeRegisteredAfterShutdown() {
        var f = new Fixture(false);
        f.close();
        assertThrows(
                IllegalStateException.class,
                () -> f.runtime.registerScheduler(mock(AbstractRequestScheduler.class)));
    }

    @Test
    void stoppingRejectsRegistrationWhileAcceptedRequestWorkCanContinue() throws Exception {
        try (var f = new Fixture()) {
            var scheduler =
                    SchedulerTestSupport.lifecycle(
                            SchedulerTestSupport.initializedScheduler(f.runtime));
            var accepted = RequestProtocolTestSupport.context(f.config, 9101L);
            var result = RequestProtocolTestSupport.submitForManualPlacement(scheduler, accepted);
            assertFalse(result.isDone());

            f.runtime.requests().stopAccepting();

            assertFalse(f.runtime.isAccepting());
            assertFalse(f.runtime.requests().isClosed());
            assertSame(accepted, f.runtime.requests().findActive(accepted.getRequestId()));
            var rejected = RequestProtocolTestSupport.context(f.config, 9102L);
            var rejectedResult =
                    RequestProtocolTestSupport.submitForManualPlacement(scheduler, rejected);
            assertFalse(rejectedResult.join().isSuccess());
            assertNull(rejected.getRequestScheduler());

            var continued = new java.util.concurrent.CompletableFuture<Void>();
            f.runtime.tasks().execute(() -> continued.complete(null));
            continued.get(1, java.util.concurrent.TimeUnit.SECONDS);
            assertFalse(result.isDone());
        }
    }

    @Test
    void cancellingOneTimeoutLeavesOtherTimeoutsAvailable() throws Exception {
        try (var f = new Fixture()) {
            var cancelledWork = new java.util.concurrent.CompletableFuture<Void>();
            var cancellation =
                    f.runtime.taskTimer().schedule(() -> cancelledWork.complete(null), 60_000L);
            cancellation.cancel(false);

            var otherWork = new java.util.concurrent.CompletableFuture<Void>();
            f.runtime.taskTimer().schedule(() -> otherWork.complete(null), 0L);
            otherWork.get(1, java.util.concurrent.TimeUnit.SECONDS);
            assertFalse(cancelledWork.isDone());
            assertTrue(SchedulerTestSupport.cleanupExecutor(f.runtime).getQueue().isEmpty());
        }
    }

    @Test
    void shutdownCancelsEveryExactOwnerInTheSharedRequestDirectory() {
        try (var f = new Fixture()) {
            var installed =
                    SchedulerTestSupport.lifecycle(
                            SchedulerTestSupport.initializedScheduler(f.runtime));
            var other =
                    SchedulerTestServices.direct(
                            mock(RequestRouteSelector.class),
                            f.runtime.newLifecycle(
                                    f.config,
                                    SchedulingTestConfig.capacity(f.config),
                                    mock(TheoryCacheHitReporter.class)));
            var installedResult =
                    RequestProtocolTestSupport.submitForManualPlacement(
                            installed, RequestProtocolTestSupport.context(f.config, 9103L));
            var otherResult =
                    RequestProtocolTestSupport.submitForManualPlacement(
                            other, RequestProtocolTestSupport.context(f.config, 9104L));

            f.runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);

            assertFalse(installedResult.join().isSuccess());
            assertFalse(otherResult.join().isSuccess());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void failedCancellationDoesNotPreventOtherOwnersFromSettling() {
        try (var f = new Fixture()) {
            var installed =
                    SchedulerTestSupport.lifecycle(
                            SchedulerTestSupport.initializedScheduler(f.runtime));
            var other =
                    spy(
                            SchedulerTestServices.direct(
                                    mock(RequestRouteSelector.class),
                                    f.runtime.newLifecycle(
                                            f.config,
                                            SchedulingTestConfig.capacity(f.config),
                                            mock(TheoryCacheHitReporter.class))));
            var lifecycle = SchedulerTestSupport.spyLifecycle(other);
            var otherContext = RequestProtocolTestSupport.context(f.config, 9105L);
            var otherResult =
                    RequestProtocolTestSupport.submitForManualPlacement(other, otherContext);
            var installedResult =
                    RequestProtocolTestSupport.submitForManualPlacement(
                            installed, RequestProtocolTestSupport.context(f.config, 9106L));
            var expected = new IllegalStateException("cancellation failed");
            doThrow(expected).when(lifecycle).cancel(otherContext, CancelReason.SHUTDOWN);

            assertSame(
                    expected,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    f.runtime.shutdown(
                                            ShutdownMode.IMMEDIATE, java.time.Duration.ZERO)));

            assertFalse(installedResult.join().isSuccess());
            assertFalse(otherResult.join().isSuccess());
            assertEquals(0, f.runtime.requests().liveRequestCount());
        }
    }

    @Test
    void shutdownFinalizesAnUnregisteredOwnerAfterCancellationFailure() {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var endpoints = mock(EndpointRegistry.class);
        var runtime =
                org.mockito.Mockito.spy(
                        new SchedulerTestServices(
                                new RequestRepository(),
                                endpoints,
                                mock(DeliveryMetricsReporter.class),
                                mock(RequestSchedulerReporter.class),
                                SchedulerTestSupport.dispatcher(),
                                service,
                                mock(EngineGrpcClient.class)));
        try {
            var owner =
                    spy(
                            SchedulerTestServices.direct(
                                    mock(RequestRouteSelector.class),
                                    runtime.newLifecycle(
                                            config,
                                            SchedulingTestConfig.capacity(config),
                                            mock(TheoryCacheHitReporter.class))));
            var lifecycle = SchedulerTestSupport.spyLifecycle(owner);
            var context = RequestProtocolTestSupport.context(config, 9107L);
            var result = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
            var expected =
                    new IllegalStateException("cancellation failed before scheduler registration");
            doThrow(expected).when(lifecycle).cancel(context, CancelReason.SHUTDOWN);

            assertSame(
                    expected,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    runtime.shutdown(
                                            ShutdownMode.IMMEDIATE, java.time.Duration.ZERO)));

            assertFalse(result.join().isSuccess());
            assertEquals(0, runtime.requests().liveRequestCount());
            verify(endpoints).close();
            assertTrue(SchedulerTestSupport.cleanupExecutor(runtime).isTerminated());
        } finally {
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void ownedWorkRejectsShutdownBeforeWaitingForTheShutdownMonitor(boolean responseCallback)
            throws Exception {
        try (var f = new Fixture()) {
            var context = RequestProtocolTestSupport.context(f.config, 9202L);
            var rejected = new java.util.concurrent.CompletableFuture<Void>();
            Runnable reenterShutdown =
                    () -> {
                        try {
                            var failure =
                                    assertThrows(
                                            IllegalStateException.class,
                                            () ->
                                                    f.runtime.shutdown(
                                                            ShutdownMode.IMMEDIATE, Duration.ZERO));
                            assertTrue(
                                    failure.getMessage()
                                            .contains("own request work or response callback"));
                            rejected.complete(null);
                        } catch (Throwable failure) {
                            rejected.completeExceptionally(failure);
                        }
                    };
            // Model shutdown holding its monitor while waiting for this registered owner.
            try {
                synchronized (f.runtime) {
                    if (responseCallback) {
                        var response =
                                new ResponseCompletionExecutor.RequestFuture(
                                        (kind, value, failure) -> false);
                        response.thenRun(reenterShutdown);
                        new Registration(
                                        f.runtime.responseCompletions(),
                                        context,
                                        response,
                                        () -> true)
                                .submit(() -> response.completeOwned(null));
                    } else {
                        f.runtime.tasks().execute(reenterShutdown);
                    }
                    rejected.get(5, java.util.concurrent.TimeUnit.SECONDS);
                    assertTrue(f.runtime.isAccepting(), "rejection must not start shutdown");
                }
            } finally {
                // Release the modeled shutdown lock before cleanup, including on a red test.
                rejected.get(5, java.util.concurrent.TimeUnit.SECONDS);
            }
        }
    }

    /**
     * Assembles the shared scheduler test runtime and closes it after each scenario.
     */
    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final EndpointRegistry endpoints = mock(EndpointRegistry.class);
        final SchedulerTestServices runtime;

        Fixture() {
            this(false);
        }

        Fixture(boolean queued) {
            this(queued, true);
        }

        Fixture(boolean queued, boolean startScheduling) {
            config.setScheduler(queued ? new SchedulerConfig() : ConfigTestSupport.direct());
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            runtime =
                    new SchedulerTestServices(
                            new RequestRepository(),
                            endpoints,
                            reporter,
                            mock(RequestSchedulerReporter.class),
                            SchedulerTestSupport.dispatcher(),
                            service,
                            mock(EngineGrpcClient.class));
            if (!startScheduling) {
                runtime.registerScheduler(
                        runtime.newLifecycle(
                                config,
                                SchedulingTestConfig.capacity(config),
                                mock(org.flexlb.balance.requestlog.TheoryCacheHitReporter.class)));
                return;
            }
            SchedulerTestSupport.create(
                    runtime,
                    service,
                    reporter,
                    mock(TheoryCacheHitReporter.class),
                    mock(RequestRouteSelector.class),
                    mock(org.flexlb.balance.schedule.DecodePreemption.class),
                    new CapacityNotifier());
        }

        public void close() {
            runtime.shutdown(ShutdownMode.IMMEDIATE, java.time.Duration.ZERO);
        }
    }
}
