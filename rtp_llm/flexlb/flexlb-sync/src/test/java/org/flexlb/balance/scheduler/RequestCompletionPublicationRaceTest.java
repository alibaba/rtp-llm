package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.await;
import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Frontend result selection precedes unlocked publication and cannot undo TTL cleanup. */
class RequestCompletionPublicationRaceTest {

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void nestedSynchronousPublicationKeepsOuterCloseReentrant(boolean asyncOuter) throws Exception {
        var responseExecutor = new ResponseCompletionExecutor(1);
        var outerRegistration = responseExecutor.tryRegister();
        var innerRegistration = responseExecutor.tryRegister();
        var outer = new RequestContext.RequestFuture((completion, response, failure, interrupt) -> false);
        var inner = new RequestContext.RequestFuture((completion, response, failure, interrupt) -> false);
        var response = new Response();
        var innerCallback = inner.thenRun(responseExecutor::close);
        var outerCallback = outer.thenRun(() -> {
            assertTrue(responseExecutor.completeNow(innerRegistration, () -> inner.completeOwned(response)));
            innerCallback.join();
            // Inner publication has returned, but this outer callback still owns its execution registration.
            responseExecutor.close();
        });
        try {
            var execution = CompletableFuture.runAsync(() -> {
                if (asyncOuter) {
                    responseExecutor.submit(outerRegistration, () -> outer.completeOwned(response));
                } else {
                    assertTrue(responseExecutor.completeNow(outerRegistration, () -> outer.completeOwned(response)));
                }
            });
            outerCallback.get(3, TimeUnit.SECONDS);
            execution.get(3, TimeUnit.SECONDS);
            responseExecutor.close();
            var executor = (ExecutorService) org.springframework.test.util.ReflectionTestUtils
                    .getField(responseExecutor, "completionWorkers");
            assertTrue(executor.isTerminated());
        } finally {
            innerRegistration.close();
            outerRegistration.close();
            responseExecutor.close();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void foreignCompletionRejectionReturnsTheOriginalOwnersRegistration(boolean async) {
        try (var owner = new ResponseCompletionExecutor(1);
             var other = new ResponseCompletionExecutor(1)) {
            var registration = owner.tryRegister();
            try {
                java.util.function.BooleanSupplier forbiddenOperation = () -> {
                    throw new AssertionError("foreign completion must not execute");
                };
                assertThrows(IllegalStateException.class, () -> {
                    if (async) {
                        other.submit(registration, forbiddenOperation);
                    } else {
                        other.completeNow(registration, forbiddenOperation);
                    }
                });
                assertTrue(((java.util.Set<?>) org.springframework.test.util.ReflectionTestUtils
                        .getField(owner, "registrations")).isEmpty());
            } finally {
                registration.close();
            }
        }
    }

    @Test
    void selectedCancellationCompletesWhenCompletionWorkersRejectSubmission() throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service,
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            CompletableFuture<Response> future = registry.register(RequestProtocolTestSupport.context(config, 503L), StrategyErrorType.BATCH_SLO_EXPIRED);
            var responseExecutor = (ResponseCompletionExecutor) org.springframework.test.util.ReflectionTestUtils
                    .getField(registry, "responseCompletions");
            var executor = (ExecutorService) org.springframework.test.util.ReflectionTestUtils
                    .getField(responseExecutor, "completionWorkers");
            executor.shutdown();
            CompletableFuture<Thread> callbackThread = future.thenApply(ignored -> Thread.currentThread());

            registry.cancel(503L, 0L, CancelReason.CLIENT_CANCELLED);
            // Waiting on the source future can help run its dependent callback on this thread.
            assertFalse(callbackThread.get(2L, TimeUnit.SECONDS) == Thread.currentThread());
            assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(),
                    future.get(2L, TimeUnit.SECONDS).getCode());
            assertEquals(RequestState.Phase.CANCELLED, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(503L, 0L).state());
        } finally {
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
            }
        }
    }

    @Test
    @Timeout(15)
    void callbackCompletionsDrainBeforeReentrantExecutorClose() throws Exception {
        var config = spy(SchedulingTestConfig.batchConfig());
        var runtime = spy(config.getInternalRuntime());
        when(runtime.getBatchDispatchCompletionThreads()).thenReturn(1);
        when(config.getInternalRuntime()).thenReturn(runtime);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        AbstractRequestScheduler scheduler = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service,
                mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        var responseExecutor = (ResponseCompletionExecutor) org.springframework.test.util.ReflectionTestUtils
                .getField(scheduler, "responseCompletions");
        int count = 256;
        var callbacks = new java.util.ArrayList<CompletableFuture<Void>>(count);
        var depth = new java.util.concurrent.atomic.AtomicInteger();
        var completed = new java.util.concurrent.atomic.AtomicInteger();
        try {
            for (int index = 0; index < count; index++) {
                long requestId = 10_000L + index;
                var context = RequestProtocolTestSupport.context(config, requestId);
                var future = RequestProtocolTestSupport.register(scheduler, context);
                boolean last = index == count - 1;
                callbacks.add(future.thenAccept(response -> {
                    assertFalse(Thread.holdsLock(context));
                    assertTrue(Thread.currentThread().getName().startsWith("response-completion-"));
                    assertEquals(1, depth.incrementAndGet(), "asynchronous publications must not recurse");
                    try {
                        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());
                        completed.incrementAndGet();
                        if (last) {
                            responseExecutor.close();
                        } else {
                            scheduler.cancel(requestId + 1, 0L, CancelReason.CLIENT_CANCELLED);
                        }
                    } finally {
                        depth.decrementAndGet();
                    }
                }));
            }
            scheduler.cancel(10_000L, 0L, CancelReason.CLIENT_CANCELLED);
            CompletableFuture.allOf(callbacks.toArray(CompletableFuture[]::new)).get(10L, TimeUnit.SECONDS);
            responseExecutor.close();
            assertEquals(count, completed.get());
            assertEquals(0, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler).liveRequestCount());
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(scheduler);
            scheduler.closeOutstandingAndTerminalize();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler).timer().close();
            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler).closeRequestExecutors();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = { false, true })
    @Timeout(15)
    void concurrentExecutorCloseSharesResultAndPreservesInterrupt(boolean shutdownFails) throws Exception {
        var responseExecutor = new ResponseCompletionExecutor(1);
        var executor = (ExecutorService) org.springframework.test.util.ReflectionTestUtils
                .getField(responseExecutor, "completionWorkers");
        RuntimeException failure = shutdownFails ? new IllegalStateException("shutdown failed") : null;
        var failedExecutor = shutdownFails ? mock(java.util.concurrent.ThreadPoolExecutor.class) : null;
        if (shutdownFails) {
            executor.shutdown();
            org.mockito.Mockito.doThrow(failure).when(failedExecutor).shutdown();
            org.springframework.test.util.ReflectionTestUtils.setField(responseExecutor, "completionWorkers", failedExecutor);
        }
        var registration = responseExecutor.tryRegister();
        assertNotNull(registration);
        var owner = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var follower = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var interruptPreserved = new java.util.concurrent.atomic.AtomicBoolean();
        var closers = Executors.newFixedThreadPool(2);
        try {
            var first = closers.submit(() -> {
                owner.set(Thread.currentThread());
                try {
                    responseExecutor.close();
                    return null;
                } catch (Throwable problem) {
                    return problem;
                }
            });
            RequestProtocolTestSupport.awaitCondition(() -> owner.get() != null
                    && owner.get().getState() == Thread.State.WAITING);
            assertNull(responseExecutor.tryRegister(),
                    "close must reject new reservations while an earlier publication is pending");
            var second = closers.submit(() -> {
                follower.set(Thread.currentThread());
                Thread.currentThread().interrupt();
                try {
                    responseExecutor.close();
                    return null;
                } catch (Throwable problem) {
                    return problem;
                } finally {
                    interruptPreserved.set(Thread.currentThread().isInterrupted());
                }
            });
            RequestProtocolTestSupport.awaitCondition(() -> follower.get() != null
                    && follower.get().getState() == Thread.State.WAITING);
            assertFalse(first.isDone());
            assertFalse(second.isDone());
            registration.close();
            assertSame(failure, first.get(5, TimeUnit.SECONDS));
            assertSame(failure, second.get(5, TimeUnit.SECONDS));
            assertTrue(interruptPreserved.get());
            assertTrue(((ExecutorService) org.springframework.test.util.ReflectionTestUtils
                    .getField(responseExecutor, "rejectedTaskWorker")).isShutdown());
            if (shutdownFails) {
                assertSame(failure, assertThrows(IllegalStateException.class, responseExecutor::close));
                verify(failedExecutor, org.mockito.Mockito.times(1)).shutdown();
            } else {
                responseExecutor.close();
                assertTrue(executor.isTerminated());
            }
        } finally {
            registration.close();
            closers.shutdownNow();
            assertTrue(closers.awaitTermination(5, TimeUnit.SECONDS));
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @Timeout(15)
    void inactivityWinsWhileAcknowledgementReportingAndTerminalCleanupAreBothPaused() throws Exception {
        var config = spy(SchedulingTestConfig.batchConfig());
        long timeoutMs = TimeUnit.HOURS.toMillis(1L);
        config.getRequestLifecycle().getRequest().setTimeoutMs(timeoutMs);
        var runtime = spy(config.getInternalRuntime());
        when(runtime.getBatchDispatchCompletionThreads()).thenReturn(1);
        when(config.getInternalRuntime()).thenReturn(runtime);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        DeliveryMetricsReporter reporter = mock(DeliveryMetricsReporter.class);
        AbstractRequestScheduler registry = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, reporter,
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        ExecutorService operations = Executors.newFixedThreadPool(2);
        CountDownLatch reportingEntered = new CountDownLatch(1);
        CountDownLatch resumeReporting = new CountDownLatch(1);
        CountDownLatch cleanupEntered = new CountDownLatch(1);
        CountDownLatch resumeCleanup = new CountDownLatch(1);
        try {
            RequestContext context = RequestProtocolTestSupport.context(config, 501L);
            CompletableFuture<Response> future = RequestProtocolTestSupport.register(registry, context);
            RequestContext requestContext = registry.findRequestContext(501L);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            when(prefill.getIp()).thenReturn("prefill");
            DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
            var reservation = new DecodeResources.ReservationHandle(1L, 501L, 1L);
            context.setFuture(future);
            RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null,
                    prefill, decode, reservation, requestContext.createdAtMs());
            RequestProtocolTestSupport.bind(registry,
                    new RequestProtocolTestSupport.Registered(item, future));
            DeliveryClaim claim = RequestProtocolTestSupport.claimBatch(
                    registry, item, 601L, () -> true);
            assertNotNull(claim);

            doAnswer(invocation -> {
                assertFalse(Thread.holdsLock(requestContext));
                reportingEntered.countDown();
                await(resumeReporting);
                return null;
            }).when(reporter).reportLatency(org.mockito.ArgumentMatchers.eq(DeliveryMetricsReporter.Latency.DISPATCH_ACK), anyString(), anyString(), anyLong());
            doAnswer(invocation -> {
                assertFalse(Thread.holdsLock(requestContext));
                cleanupEntered.countDown();
                await(resumeCleanup);
                return DecodeResources.ReservationReleaseResult.RELEASED;
            }).when(decode).release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            when(SchedulerTestSupport.cancelChannel(registry).cancel(any(), anyLong(), any(), anyLong()))
                    .thenReturn(CompletableFuture.completedFuture(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED));
            assertTrue(claim.item.ctx().scheduler().tryStartSend(claim));
            var completions = new java.util.concurrent.atomic.AtomicInteger();
            CompletableFuture<Void> callback = future.thenAccept(response -> {
                assertFalse(Thread.holdsLock(requestContext), "frontend callbacks must not hold the context lock");
                completions.incrementAndGet();
            });

            Future<?> acknowledgement = operations.submit(() -> claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.delivered()));
            assertTrue(reportingEntered.await(2L, TimeUnit.SECONDS));
            assertEquals(RequestState.Phase.ACKNOWLEDGED, requestContext.snapshot().state());
            assertFalse(future.isDone());

            long handoffAtMs = (long) org.springframework.test.util.ReflectionTestUtils
                    .getField(requestContext, "batchEnqueueStartedAtMs");
            Future<?> expiry = operations.submit(() ->
                    RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handoffAtMs + timeoutMs));
            assertTrue(cleanupEntered.await(2L, TimeUnit.SECONDS));
            assertEquals(RequestState.Phase.TIMED_OUT, requestContext.snapshot().state(),
                    "selected terminal is visible before endpoint cleanup completes");
            assertEquals(RequestContext.RequestStage.FINALIZING, requestContext.stage());
            assertSame(requestContext, registry.findRequestContext(501L));
            assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                    registry.register(RequestProtocolTestSupport.context(config, 501L), StrategyErrorType.BATCH_SLO_EXPIRED).join().getCode());
            resumeReporting.countDown();
            acknowledgement.get(2L, TimeUnit.SECONDS);

            // A second response proves the obsolete ACK has run. The terminal response
            // may finish while cleanup is pending, but cannot archive the live identity.
            var barrier = registry.register(RequestProtocolTestSupport.context(config, 502L), StrategyErrorType.BATCH_SLO_EXPIRED);
            registry.cancel(502L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(barrier.get(2L, TimeUnit.SECONDS).isSuccess());
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess(),
                    "an obsolete success permit cannot win after TTL claims cleanup");
            assertSame(requestContext, registry.findRequestContext(501L));
            assertEquals(RequestContext.RequestStage.FINALIZING, requestContext.stage());

            resumeCleanup.countDown();
            expiry.get(2L, TimeUnit.SECONDS);
            registry.runtime.continuations().awaitIdle();
            Response expired = future.get(2L, TimeUnit.SECONDS);
            callback.get(2L, TimeUnit.SECONDS);
            assertFalse(expired.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), expired.getCode());
            assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    expired.getAdmissionRejectReason());
            assertEquals(RequestState.Phase.TIMED_OUT, requestContext.snapshot().state());
            assertEquals(RequestContext.RequestStage.FINISHED, requestContext.stage());
            assertNull(registry.findRequestContext(501L));
            assertEquals(RequestState.Phase.TIMED_OUT, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(501L, 0L).state());
            assertEquals(1, completions.get());
            verify(decode).release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            verify(prefill).releaseRequest(item);
        } finally {
            resumeReporting.countDown();
            resumeCleanup.countDown();
            operations.shutdownNow();
            operations.awaitTermination(5L, TimeUnit.SECONDS);
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).timer().close();
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry).closeRequestExecutors();
            }
        }
    }

    @Test
    @Timeout(15)
    void terminalInvalidatesAcknowledgementQueuedBehindAnotherRequestsCallback() throws Exception {
        try (var completions = new ResponseCompletionExecutor(1)) {
            Fixture fixture = fixture(completions);
            var entered = new CountDownLatch(1);
            var resume = new CountDownLatch(1);
            var blockedFuture = new RequestContext.RequestFuture((kind, response, failure, interrupt) -> false);
            var blockedCallback = blockedFuture.thenRun(() -> {
                entered.countDown();
                await(resume);
            });
            var registration = completions.tryRegister();
            completions.submit(registration, () -> blockedFuture.completeOwned(new Response()));
            try {
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                fixture.scheduler().deliveryEffects(fixture.requestContext(), fixture.delivery(), null).run();
                assertNull(fixture.requestContext().selectedResponse(), "queued ACK must remain unselected");
                assertFalse(fixture.requestContext().future().isDone());
                Response failure = new Response();
                failure.setSuccess(false);
                TerminalAction terminal;
                synchronized (fixture.requestContext()) {
                    terminal = RequestProtocolTestSupport.claimTerminal(fixture.scheduler(), fixture.requestContext(),
                            TerminalOutcome.fail("worker failed while response executor was busy"), failure, true);
                    assertNotNull(terminal.publication());
                    fixture.scheduler().commitTerminalRecord(fixture.requestContext(), terminal);
                }
                var selected = AbstractRequestScheduler.selectPublication(fixture.requestContext(), terminal.publication(),
                        RequestContext.ResponseCompletion.RESPONSE, failure, null, false);
                completions.submit(terminal.publication().registration, () -> AbstractRequestScheduler.completeFutureResult(terminal.publication(), selected));
                resume.countDown();
                blockedCallback.get(2, TimeUnit.SECONDS);
                assertSame(failure, fixture.requestContext().future().get(2, TimeUnit.SECONDS));
            } finally {
                resume.countDown();
                completions.close();
                assertTrue(((java.util.Set<?>) org.springframework.test.util.ReflectionTestUtils
                        .getField(completions, "registrations")).isEmpty());
                fixture.scheduler().runtime.timer().close();
                fixture.scheduler().runtime.closeRequestExecutors();
            }
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void rejectedLockHeldCompletionReturnsExecutionRegistration(boolean asynchronous) {
        try (var responseExecutor = new ResponseCompletionExecutor(1)) {
            Fixture fixture = fixture(responseExecutor);
            var selected = AbstractRequestScheduler.selectPublication(fixture.requestContext(), fixture.delivery().publication(),
                    RequestContext.ResponseCompletion.RESPONSE, new Response(), null, false);
            try {
                synchronized (fixture.requestContext()) {
                    assertThrows(IllegalStateException.class, () -> org.springframework.test.util.ReflectionTestUtils
                            .invokeMethod(fixture.scheduler(), asynchronous ? "submitResponse" : "completeResponseNow", fixture.delivery().publication(), selected));
                }
                assertFalse(fixture.requestContext().future().isDone(), "invalid lock-held calls must not run callbacks");
                assertTrue(((java.util.Set<?>) org.springframework.test.util.ReflectionTestUtils
                        .getField(responseExecutor, "registrations")).isEmpty());
            } finally {
                fixture.scheduler().runtime.timer().close();
                fixture.scheduler().runtime.closeRequestExecutors();
            }
        }
    }

    @Test
    void deliverySelectedBeforeExpiryKeepsItsResponseEvenBeforeTheFutureIsCompleted() {
        Fixture fixture = fixture();
        Response success = new Response();
        success.setSuccess(true);
        RequestContext.ResponseResult publishSuccess = AbstractRequestScheduler.selectPublication(fixture.requestContext(), fixture.delivery().publication(), RequestContext.ResponseCompletion.RESPONSE, success, null, false);
        assertFalse(fixture.requestContext().future().isDone());
        CompletableFuture<Void> callback = fixture.requestContext().future().thenAccept(response ->
                assertFalse(Thread.holdsLock(fixture.requestContext())));
        synchronized (fixture.requestContext()) {
            RequestProtocolTestSupport.recordCancellation(fixture.scheduler(), fixture.requestContext(), CancelReason.DEADLINE_EXCEEDED, "request inactive");
            TerminalAction terminal = RequestProtocolTestSupport.claimTerminal(fixture.scheduler(), fixture.requestContext(), TerminalOutcome.timeout("request inactive"), new Response(), true);
            assertNotNull(terminal);
            assertNull(terminal.publication(), "an already selected delivery owns the frontend result");
            fixture.scheduler().commitTerminalRecord(fixture.requestContext(), terminal);
            assertEquals(RequestState.Phase.TIMED_OUT, fixture.requestContext().snapshot().state());
        }
        assertTrue(AbstractRequestScheduler.completeFutureResult(fixture.delivery().publication(), publishSuccess));
        assertSame(success, fixture.requestContext().future().join());
        callback.join();
    }

    @Test
    void futureCancelCannotReplaceSelectedUnpublishedDelivery() {
        Fixture fixture = fixture();
        Response success = new Response();
        success.setSuccess(true);
        RequestContext.ResponseResult selected = AbstractRequestScheduler.selectPublication(fixture.requestContext(), fixture.delivery().publication(), RequestContext.ResponseCompletion.RESPONSE, success, null, false);

        assertFalse(fixture.requestContext().future().cancel(false));
        assertFalse(fixture.requestContext().future().isDone());
        assertEquals(RequestState.Phase.ACKNOWLEDGED, fixture.requestContext().snapshot().state());
        assertTrue(AbstractRequestScheduler.completeFutureResult(fixture.delivery().publication(), selected));
        assertSame(success, fixture.requestContext().future().join());
    }

    @ParameterizedTest
    @EnumSource(TerminalForm.class)
    void terminalSelectionInvalidatesAnUnpublishedAcknowledgementForEveryCompletionKind(TerminalForm form) {
        Fixture fixture = fixture();
        Response failure = new Response();
        failure.setSuccess(false);
        TerminalAction terminal;
        synchronized (fixture.requestContext()) {
            // ACKNOWLEDGED records the Engine fact, not a selected frontend result.
            terminal = RequestProtocolTestSupport.claimTerminal(fixture.scheduler(), fixture.requestContext(), TerminalOutcome.fail("worker failed before response publication"), failure, failure != null);
            assertNotNull(terminal.publication());
            fixture.scheduler().commitTerminalRecord(fixture.requestContext(), terminal);
        }
        Response success = new Response();
        success.setSuccess(true);
        assertFalse(AbstractRequestScheduler.completeFutureResult(fixture.delivery().publication(), AbstractRequestScheduler.selectPublication(fixture.requestContext(), fixture.delivery().publication(), RequestContext.ResponseCompletion.RESPONSE, success, null, false)));
        assertFalse(fixture.requestContext().future().isDone());
        CompletableFuture<Void> callback = fixture.requestContext().future().handle((response, error) -> {
            assertFalse(Thread.holdsLock(fixture.requestContext()));
            return null;
        });
        RequestContext.ResponseResult publication = switch (form) {
            case RESPONSE -> AbstractRequestScheduler.selectPublication(fixture.requestContext(), terminal.publication(), RequestContext.ResponseCompletion.RESPONSE, failure, null, false);
            case FAILURE -> AbstractRequestScheduler.selectPublication(fixture.requestContext(), terminal.publication(), RequestContext.ResponseCompletion.FAILURE, null, new IllegalStateException("worker failed"), false);
            case CANCELLATION -> AbstractRequestScheduler.selectPublication(fixture.requestContext(), terminal.publication(), RequestContext.ResponseCompletion.CANCELLATION, null, null, false);
        };
        assertTrue(AbstractRequestScheduler.completeFutureResult(terminal.publication(), publication));
        callback.join();
        assertTrue(fixture.requestContext().future().isDone());
        if (form == TerminalForm.RESPONSE) {
            assertSame(failure, fixture.requestContext().future().join());
        } else {
            assertTrue(fixture.requestContext().future().isCompletedExceptionally());
            assertEquals(form == TerminalForm.CANCELLATION, fixture.requestContext().future().isCancelled());
        }
    }

    @ParameterizedTest
    @EnumSource(TerminalForm.class)
    void externalFutureOperationUnderContextLockLeavesRequestUnchanged(TerminalForm form) {
        RequestContext requestContext = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 702L);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(mock(ResponseCompletionExecutor.class), requestContext, mock(ExpirationTimer.class));
        synchronized (requestContext) {
            assertThrows(IllegalStateException.class, () -> {
                switch (form) {
                    case RESPONSE ->
                        requestContext.future().complete(new Response());
                    case FAILURE ->
                        requestContext.future().completeExceptionally(new IllegalStateException("failure"));
                    case CANCELLATION ->
                        requestContext.future().cancel(false);
                }
            });
            assertTrue(requestContext.isOpen());
            assertEquals(RequestState.Phase.QUEUED, requestContext.snapshot().state());
            assertFalse(requestContext.future().isDone());
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void rejectedPublicationRegistrationDoesNotClaimTerminalOwnership(boolean deliveryFailure) {
        try (var publisher = spy(new ResponseCompletionExecutor(1))) {
            Fixture fixture = fixture(publisher);
            RequestContext context = fixture.requestContext();
            RequestRoute exact = context.route();
            var stage = context.stage();
            org.mockito.Mockito.doReturn(null).when(publisher).tryRegister();
            try {
                synchronized (context) {
                    assertThrows(IllegalStateException.class, () -> {
                        if (deliveryFailure) {
                            fixture.scheduler().selectDeliveryFailureLocked(context, exact, DeliveryResult.Status.PREFILL_REJECTED, "rejected");
                        } else {
                            var decision = context.decideRequestEndLocked(DeferredTerminal.worker(WorkerTerminalSource.DECODE_ENDPOINT, true, 0L));
                            assertNotNull(decision);
                            fixture.scheduler().claimFinalizationLocked(context, decision);
                        }
                    });
                    assertSame(exact, context.route());
                    assertEquals(stage, context.stage());
                    assertFalse(context.hasTerminalAction());
                    assertFalse(context.hasCleanup());
                    assertNull(context.selectedResponse());
                    assertFalse(context.future().isDone());
                }
            } finally {
                fixture.delivery().publication().abandonIfUnused();
                fixture.scheduler().runtime.timer().close();
                fixture.scheduler().runtime.closeRequestExecutors();
            }
        }
    }

    @Test
    void obsoleteTerminalDoesNotAcquirePublicationAfterExecutorClose() {
        try (var publisher = spy(new ResponseCompletionExecutor(1))) {
            Fixture fixture = fixture(publisher);
            fixture.delivery().publication().abandonIfUnused();
            publisher.close();
            org.mockito.Mockito.clearInvocations(publisher);
            try {
                synchronized (fixture.requestContext()) {
                    var decision = fixture.requestContext().acceptRequestEndLocked(null,
                            DeferredTerminal.worker(WorkerTerminalSource.PREFILL_ENDPOINT, true, 0L));
                    assertNull(fixture.scheduler().claimFinalizationLocked(fixture.requestContext(), decision));
                    assertEquals(RequestContext.RequestStage.DELIVERING, fixture.requestContext().stage());
                    assertFalse(fixture.requestContext().hasCleanup());
                }
                verify(publisher, org.mockito.Mockito.never()).tryRegister();
            } finally {
                fixture.delivery().publication().abandonIfUnused();
                fixture.scheduler().runtime.timer().close();
                fixture.scheduler().runtime.closeRequestExecutors();
            }
        }
    }

    private static Fixture fixture() {
        var executor = mock(ResponseCompletionExecutor.class);
        when(executor.tryRegister()).thenAnswer(invocation -> new ResponseCompletionExecutor.CompletionRegistration(executor));
        return fixture(executor);
    }

    private static Fixture fixture(ResponseCompletionExecutor responseExecutor) {
        var config = SchedulingTestConfig.batchConfig();
        RequestContext context = RequestProtocolTestSupport.context(config, 701L);
        AbstractRequestScheduler requestOwner = RequestProtocolTestSupport.initialize(responseExecutor, context, mock(ExpirationTimer.class));
        RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), new Response(), null, null, null, null, null, context.createdAtMs());
        RequestContext.DeliveryPublication delivery;
        AdmissionHandle admission;
        synchronized (context) {
            admission = RequestProtocolTestSupport.beginAdmission(requestOwner, context);
            assertNotNull(admission);
        }
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, requestOwner.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        admission.finish();
        synchronized (context) {
            RequestProtocolTestSupport.startBatchDelivery(requestOwner, context, 801L);
            var permit = requestOwner.requirePublicationPermitLocked(context, RequestContext.PublicationKind.DELIVERY);
            delivery = context.acknowledgeDelivery(permit, System.currentTimeMillis());
        }
        assertNotNull(delivery);
        return new Fixture(requestOwner, context, delivery);
    }

    private enum TerminalForm { RESPONSE, FAILURE, CANCELLATION }

    private record Fixture(AbstractRequestScheduler scheduler, RequestContext requestContext, RequestContext.DeliveryPublication delivery) { }
}
