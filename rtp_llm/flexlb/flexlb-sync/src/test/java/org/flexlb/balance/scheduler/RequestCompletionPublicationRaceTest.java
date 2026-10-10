package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
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
    void externalCloseWaitsForNestedCompletionCallbacks(boolean asyncOuter) throws Exception {
        var responseExecutor = new ResponseCompletionExecutor(1);
        var outerRegistration = responseExecutor.tryRegister();
        var innerRegistration = responseExecutor.tryRegister();
        var outer = new RequestFuture((completion, response, failure) -> false);
        var inner = new RequestFuture((completion, response, failure) -> false);
        var response = new Response();
        var callbackEntered = new CountDownLatch(1);
        var resumeCallback = new CountDownLatch(1);
        var outerCallback = outer.thenRun(() -> {
            assertTrue(innerRegistration.completeNow(() -> inner.completeOwned(response)));
            callbackEntered.countDown();
            await(resumeCallback);
        });
        var closingThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
        var closer = Executors.newSingleThreadExecutor();
        try {
            var execution = CompletableFuture.runAsync(() -> {
                if (asyncOuter) {
                    outerRegistration.submit(() -> outer.completeOwned(response));
                } else {
                    assertTrue(outerRegistration.completeNow(() -> outer.completeOwned(response)));
                }
            });
            assertTrue(callbackEntered.await(3, TimeUnit.SECONDS));
            var close = closer.submit(() -> {
                closingThread.set(Thread.currentThread());
                responseExecutor.close();
            });
            RequestProtocolTestSupport.awaitCondition(() -> closingThread.get() != null
                    && closingThread.get().getState() == Thread.State.WAITING);
            assertNull(responseExecutor.tryRegister(), "close must stop accepting new registrations");
            assertTrue(outer.isDone());
            assertTrue(inner.isDone());
            assertFalse(outerCallback.isDone(), "the source result does not mean its callback has finished");
            assertFalse(close.isDone());
            var workers = (ExecutorService) org.springframework.test.util.ReflectionTestUtils
                    .getField(responseExecutor, "completionWorkers");
            assertFalse(workers.isShutdown(), "workers must stay alive until accepted callbacks return");
            resumeCallback.countDown();
            outerCallback.get(3, TimeUnit.SECONDS);
            execution.get(3, TimeUnit.SECONDS);
            close.get(3, TimeUnit.SECONDS);
            assertTrue(workers.isTerminated());
            assertTrue(((ExecutorService) org.springframework.test.util.ReflectionTestUtils
                    .getField(responseExecutor, "rejectedTaskWorker")).isTerminated());
        } finally {
            resumeCallback.countDown();
            innerRegistration.close();
            outerRegistration.close();
            closer.shutdownNow();
            assertTrue(closer.awaitTermination(5, TimeUnit.SECONDS));
            responseExecutor.close();
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
            CompletableFuture<Response> future = registry.register(RequestProtocolTestSupport.context(config, 503L), () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            var responseExecutor = (ResponseCompletionExecutor) org.springframework.test.util.ReflectionTestUtils
                    .getField(registry.runtime, "responseCompletions");
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
                SchedulerTestSupport.timer(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)).close();
                SchedulerTestSupport.closeRequestExecutors(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
            }
        }
    }

    @Test
    @Timeout(15)
    void callbackCompletionsDrainBeforeExternalExecutorClose() throws Exception {
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
                .getField(scheduler.runtime, "responseCompletions");
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
                        if (!last) {
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
            SchedulerTestSupport.timer(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler)).close();
            SchedulerTestSupport.closeRequestExecutors(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
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
            RequestContext requestContext = registry.requests.findActive(501L);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            when(prefill.getIp()).thenReturn("prefill");
            DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
            var reservation = new DecodeResources.ReservationHandle(1L, 501L, 1L);
            context.data().setFuture(future);
            RequestRoute item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, new Response(), null, null,
                    prefill, decode, reservation, requestContext.snapshot().createdAtMs());
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
                return DecodeResources.ResourceChange.of(DecodeResources.ReservationReleaseResult.RELEASED, true);
            }).when(decode).release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            when(SchedulerTestSupport.cancelChannel(registry).cancel(any(), anyLong(), any(), anyLong()))
                    .thenReturn(CompletableFuture.completedFuture(org.flexlb.balance.eviction.EngineCancelChannel.CancelAck.REQUEST_CLEANED));
            assertTrue(claim.owner().data().getScheduler().tryStartSend(claim));
            var completions = new java.util.concurrent.atomic.AtomicInteger();
            CompletableFuture<Void> callback = future.thenAccept(response -> {
                assertFalse(Thread.holdsLock(requestContext), "frontend callbacks must not hold the context lock");
                completions.incrementAndGet();
            });

            Future<?> acknowledgement = operations.submit(() -> claim.owner().data().getScheduler().onDeliveryEvidence(claim, new ContextResult.TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis())));
            assertTrue(reportingEntered.await(2L, TimeUnit.SECONDS));
            assertEquals(RequestState.Phase.ACKNOWLEDGED, RequestState.from(requestContext.snapshot()).state());
            assertFalse(future.isDone());

            long handoffAtMs = requestContext.snapshot().delivery().startedAtMs();
            Future<?> expiry = operations.submit(() ->
                    RequestProtocolTestSupport.expireInactiveRequest(registry, requestContext, handoffAtMs + timeoutMs));
            assertTrue(cleanupEntered.await(2L, TimeUnit.SECONDS));
            assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(requestContext.snapshot()).state(),
                    "selected terminal is visible before endpoint cleanup completes");
            assertEquals(SchedulerTestSupport.Stage.FINALIZING, SchedulerTestSupport.stage(requestContext));
            assertSame(requestContext, registry.requests.findActive(501L));
            assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                    registry.register(RequestProtocolTestSupport.context(config, 501L), () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED).join().getCode());
            resumeReporting.countDown();
            acknowledgement.get(2L, TimeUnit.SECONDS);

            // A second response proves the obsolete ACK has run. The terminal response
            // may finish while cleanup is pending, but cannot archive the live identity.
            var barrier = registry.register(RequestProtocolTestSupport.context(config, 502L), () -> { }, StrategyErrorType.BATCH_SLO_EXPIRED);
            registry.cancel(502L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(barrier.get(2L, TimeUnit.SECONDS).isSuccess());
            assertFalse(future.get(2L, TimeUnit.SECONDS).isSuccess(),
                    "an obsolete success permit cannot win after TTL claims cleanup");
            assertSame(requestContext, registry.requests.findActive(501L));
            assertEquals(SchedulerTestSupport.Stage.FINALIZING, SchedulerTestSupport.stage(requestContext));

            resumeCleanup.countDown();
            expiry.get(2L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(registry.runtime);
            Response expired = future.get(2L, TimeUnit.SECONDS);
            callback.get(2L, TimeUnit.SECONDS);
            assertFalse(expired.isSuccess());
            assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), expired.getCode());
            assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    expired.getAdmissionRejectReason());
            assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(requestContext.snapshot()).state());
            assertEquals(SchedulerTestSupport.Stage.FINISHED, SchedulerTestSupport.stage(requestContext));
            assertNull(registry.requests.findActive(501L));
            assertEquals(RequestState.Phase.TIMED_OUT, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(registry).getRequestState(501L, 0L).state());
            assertEquals(1, completions.get());
            verify(decode).release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
            verify(prefill).release(null, item);
        } finally {
            resumeReporting.countDown();
            resumeCleanup.countDown();
            operations.shutdownNow();
            operations.awaitTermination(5L, TimeUnit.SECONDS);
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(registry)) {
                registry.closeOutstandingAndTerminalize();
                SchedulerTestSupport.timer(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry)).close();
                SchedulerTestSupport.closeRequestExecutors(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(registry));
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
            var blockedFuture = new RequestFuture((kind, response, failure) -> false);
            var blockedCallback = blockedFuture.thenRun(() -> { entered.countDown(); await(resume); });
            completions.tryRegister().submit(() -> blockedFuture.completeOwned(new Response()));
            try {
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                org.springframework.test.util.ReflectionTestUtils.invokeMethod(fixture.scheduler(), "publishSelected",
                        fixture.publication(), fixture.candidate(), RequestFuture.Completion.RESPONSE,
                        fixture.response(), null, false);
                assertNull(fixture.requestContext().snapshot().selectedResponse(), "queued ACK must remain unselected");
                assertFalse(fixture.requestContext().data().getFuture().isDone());
                Response failure = new Response();
                failure.setSuccess(false);
                var terminalPublication = new ResponsePublication(fixture.scheduler().runtime, fixture.requestContext(), false);
                ContextResult.EndToken terminal;
                SchedulerTestSupport.FixtureSettlement settled;
                synchronized (fixture.requestContext()) {
                    terminal = RequestProtocolTestSupport.claimTerminal(fixture.scheduler(), fixture.requestContext(),
                            TerminalOutcome.fail("worker failed while response executor was busy"), failure, true);
                    assertNotNull(terminalPublication);
                    settled = SchedulerTestSupport.settleTerminalFixture(fixture.requestContext(), terminal);
                }
                settled.releaseAndArchive(fixture.scheduler(), fixture.requestContext());
                var candidate = new ContextResult.ResponseCandidate(fixture.requestContext(), ContextResult.ResponseKind.TERMINAL,
                        fixture.requestContext().snapshot().outcome(), System.currentTimeMillis());
                terminalPublication.submit(() -> fixture.requestContext().selectResponse(candidate) instanceof ContextResult.Granted<?>
                        && terminalPublication.complete(RequestFuture.Completion.RESPONSE, failure, null));
                resume.countDown();
                blockedCallback.get(2, TimeUnit.SECONDS);
                assertSame(failure, fixture.requestContext().data().getFuture().get(2, TimeUnit.SECONDS));
            } finally {
                resume.countDown();
                completions.close();
                assertTrue(((java.util.Set<?>) org.springframework.test.util.ReflectionTestUtils
                        .getField(completions, "registrations")).isEmpty());
                SchedulerTestSupport.timer(fixture.scheduler().runtime).close();
                SchedulerTestSupport.closeRequestExecutors(fixture.scheduler().runtime);
            }
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @Timeout(15)
    void rejectedLockHeldCompletionReturnsExecutionRegistration(boolean asynchronous) {
        try (var responseExecutor = new ResponseCompletionExecutor(1)) {
            Fixture fixture = fixture(responseExecutor);
            assertTrue(fixture.requestContext().selectResponse(fixture.candidate()) instanceof ContextResult.Granted<?>);
            try {
                synchronized (fixture.requestContext()) {
                    assertThrows(IllegalStateException.class, () -> {
                        if (asynchronous) { fixture.publication().submit(() -> true); }
                        else { fixture.publication().completeNow(() -> true); }
                    });
                }
                assertFalse(fixture.requestContext().data().getFuture().isDone(), "invalid lock-held calls must not run callbacks");
                assertTrue(((java.util.Set<?>) org.springframework.test.util.ReflectionTestUtils
                        .getField(responseExecutor, "registrations")).isEmpty());
            } finally {
                SchedulerTestSupport.timer(fixture.scheduler().runtime).close();
                SchedulerTestSupport.closeRequestExecutors(fixture.scheduler().runtime);
            }
        }
    }

    @Test
    void deliverySelectedBeforeExpiryKeepsItsResponseEvenBeforeTheFutureIsCompleted() {
        Fixture fixture = fixture();
        Response success = new Response();
        success.setSuccess(true);
        assertTrue(fixture.requestContext().selectResponse(fixture.candidate()) instanceof ContextResult.Granted<?>);
        assertFalse(fixture.requestContext().data().getFuture().isDone());
        CompletableFuture<Void> callback = fixture.requestContext().data().getFuture().thenAccept(response ->
                assertFalse(Thread.holdsLock(fixture.requestContext())));
        SchedulerTestSupport.FixtureSettlement settled;
        synchronized (fixture.requestContext()) {
            var deadline = fixture.requestContext().snapshot().deadlines().inactivity();
            fixture.requestContext().recordDeadline(deadline, deadline.atMs(), null);
            var terminal = RequestProtocolTestSupport.claimTerminal(fixture.scheduler(), fixture.requestContext(),
                    TerminalOutcome.timeout("request inactive"), new Response(), true);
            assertNotNull(terminal);
            assertSame(fixture.candidate(), fixture.requestContext().snapshot().selectedResponse().candidate(),
                    "an already selected delivery owns the frontend result");
            settled = SchedulerTestSupport.settleTerminalFixture(fixture.requestContext(), terminal);
            assertEquals(RequestState.Phase.TIMED_OUT, RequestState.from(fixture.requestContext().snapshot()).state());
        }
        settled.releaseAndArchive(fixture.scheduler(), fixture.requestContext());
        assertTrue(fixture.publication().complete(RequestFuture.Completion.RESPONSE, success, null));
        assertSame(success, fixture.requestContext().data().getFuture().join());
        callback.join();
        fixture.publication().close();
    }

    @Test
    void futureCancelCannotReplaceSelectedUnpublishedDelivery() {
        Fixture fixture = fixture();
        Response success = new Response();
        success.setSuccess(true);
        assertTrue(fixture.requestContext().selectResponse(fixture.candidate()) instanceof ContextResult.Granted<?>);
        assertFalse(fixture.requestContext().data().getFuture().cancel(false));
        assertFalse(fixture.requestContext().data().getFuture().isDone());
        assertEquals(RequestState.Phase.ACKNOWLEDGED, RequestState.from(fixture.requestContext().snapshot()).state());
        assertTrue(fixture.publication().complete(RequestFuture.Completion.RESPONSE, success, null));
        assertSame(success, fixture.requestContext().data().getFuture().join());
        fixture.publication().close();
    }

    @ParameterizedTest
    @EnumSource(TerminalForm.class)
    void terminalSelectionInvalidatesAnUnpublishedAcknowledgementForEveryCompletionKind(TerminalForm form) {
        Fixture fixture = fixture();
        Response failure = new Response();
        failure.setSuccess(false);
        var publication = new ResponsePublication(fixture.scheduler().runtime, fixture.requestContext(), false);
        SchedulerTestSupport.FixtureSettlement settled;
        synchronized (fixture.requestContext()) {
            var terminal = RequestProtocolTestSupport.claimTerminal(fixture.scheduler(), fixture.requestContext(),
                    TerminalOutcome.fail("worker failed before response publication"), failure, true);
            assertNotNull(publication);
            settled = SchedulerTestSupport.settleTerminalFixture(fixture.requestContext(), terminal);
        }
        settled.releaseAndArchive(fixture.scheduler(), fixture.requestContext());
        Response success = new Response();
        success.setSuccess(true);
        assertFalse(fixture.requestContext().selectResponse(fixture.candidate()) instanceof ContextResult.Granted<?>);
        assertFalse(fixture.requestContext().data().getFuture().isDone());
        CompletableFuture<Void> callback = fixture.requestContext().data().getFuture().handle((response, error) -> {
            assertFalse(Thread.holdsLock(fixture.requestContext())); return null;
        });
        var candidate = new ContextResult.ResponseCandidate(fixture.requestContext(), ContextResult.ResponseKind.TERMINAL,
                fixture.requestContext().snapshot().outcome(), System.currentTimeMillis());
        assertTrue(fixture.requestContext().selectResponse(candidate) instanceof ContextResult.Granted<?>);
        boolean completed = switch (form) {
            case RESPONSE -> publication.complete(RequestFuture.Completion.RESPONSE, failure, null);
            case FAILURE -> publication.complete(RequestFuture.Completion.FAILURE, null, new IllegalStateException("worker failed"));
            case CANCELLATION -> publication.complete(RequestFuture.Completion.CANCELLATION, null, null);
        };
        assertTrue(completed);
        callback.join();
        assertTrue(fixture.requestContext().data().getFuture().isDone());
        if (form == TerminalForm.RESPONSE) { assertSame(failure, fixture.requestContext().data().getFuture().join()); }
        else {
            assertTrue(fixture.requestContext().data().getFuture().isCompletedExceptionally());
            assertEquals(form == TerminalForm.CANCELLATION, fixture.requestContext().data().getFuture().isCancelled());
        }
        fixture.publication().close(); publication.close();
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
                        requestContext.data().getFuture().complete(new Response());
                    case FAILURE ->
                        requestContext.data().getFuture().completeExceptionally(new IllegalStateException("failure"));
                    case CANCELLATION ->
                        requestContext.data().getFuture().cancel(false);
                }
            });
            assertTrue(requestContext.snapshot().phase() == ContextResult.Phase.OPEN && requestContext.snapshot().cancellation() == null);
            assertEquals(RequestState.Phase.QUEUED, RequestState.from(requestContext.snapshot()).state());
            assertFalse(requestContext.data().getFuture().isDone());
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void rejectedPublicationRegistrationDoesNotClaimTerminalOwnership(boolean deliveryFailure) {
        try (var publisher = spy(new ResponseCompletionExecutor(1))) {
            Fixture fixture = fixture(publisher, !deliveryFailure);
            RequestContext context = fixture.requestContext();
            RequestRoute exact = context.snapshot().route();
            var phase = context.snapshot().phase();
            org.mockito.Mockito.doReturn(null).when(publisher).tryRegister();
            try {
                assertThrows(IllegalStateException.class, () -> {
                    if (deliveryFailure) {
                        fixture.scheduler().onDeliveryEvidence(fixture.claim(), new ContextResult.TransportResult(fixture.claim(),
                                DeliveryResult.prefillRejected(new IllegalStateException("rejected")), System.currentTimeMillis()));
                    } else {
                        fixture.scheduler().closeRequest(context, new ContextResult.EndEvidence(exact, ContextResult.EndKind.WORKER,
                                RoleType.DECODE, true, 0L, StrategyErrorType.WORKER_EXECUTION_FAILED, null, false, true), null);
                    }
                });
                assertSame(exact, context.snapshot().route());
                assertEquals(phase, context.snapshot().phase());
                assertNull(context.snapshot().close().endOperation());
                assertNull(context.snapshot().outcome());
                assertNull(context.snapshot().selectedResponse());
                assertFalse(context.data().getFuture().isDone());
            } finally {
                fixture.publication().close();
                SchedulerTestSupport.timer(fixture.scheduler().runtime).close();
                SchedulerTestSupport.closeRequestExecutors(fixture.scheduler().runtime);
            }
        }
    }

    @Test
    void obsoleteTerminalDoesNotAcquirePublicationAfterExecutorClose() {
        try (var publisher = spy(new ResponseCompletionExecutor(1))) {
            Fixture fixture = fixture(publisher);
            fixture.publication().close();
            publisher.close();
            org.mockito.Mockito.clearInvocations(publisher);
            try {
                fixture.scheduler().onEngineEvidence(fixture.requestContext(), new ContextResult.EngineEvidence(null, null,
                        RoleType.PREFILL, ContextResult.EngineKind.COMPLETED, 0L, null, System.currentTimeMillis()));
                assertEquals(ContextResult.Phase.OPEN, fixture.requestContext().snapshot().phase());
                assertTrue(fixture.requestContext().snapshot().delivery().transferred());
                assertNull(fixture.requestContext().snapshot().outcome());
                verify(publisher, org.mockito.Mockito.never()).tryRegister();
            } finally {
                fixture.publication().close();
                SchedulerTestSupport.timer(fixture.scheduler().runtime).close();
                SchedulerTestSupport.closeRequestExecutors(fixture.scheduler().runtime);
            }
        }
    }

    private static Fixture fixture() {
        var executor = mock(ResponseCompletionExecutor.class);
        when(executor.tryRegister()).thenAnswer(invocation -> mock(ResponseCompletionExecutor.CompletionRegistration.class));
        return fixture(executor);
    }

    private static Fixture fixture(ResponseCompletionExecutor responseExecutor) { return fixture(responseExecutor, true); }

    private static Fixture fixture(ResponseCompletionExecutor responseExecutor, boolean delivered) {
        var config = SchedulingTestConfig.batchConfig();
        RequestContext context = RequestProtocolTestSupport.context(config, 701L);
        AbstractRequestScheduler owner = RequestProtocolTestSupport.initialize(responseExecutor, context, mock(ExpirationTimer.class));
        RequestRoute item = SchedulingTestConfig.createRoute(context, new Response(), null, null,
                mock(PrefillEndpoint.class), null, null, context.snapshot().createdAtMs());
        var admission = RequestProtocolTestSupport.beginAdmission(owner, context);
        assertNotNull(admission);
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                owner.commitRoute(item, RequestProtocolTestSupport.publication(() -> true)));
        admission.finish();
        var claim = owner.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 801L, null);
        assertNotNull(claim);
        assertTrue(owner.tryStartSend(claim));
        if (delivered) { context.recordDeliveryResult(new ContextResult.TransportResult(claim, DeliveryResult.delivered(), System.currentTimeMillis())); }
        var publication = new ResponsePublication(owner.runtime, context, true);
        if (delivered) { assertTrue(context.selectResponse(new ContextResult.ResponseCandidate(context, ContextResult.ResponseKind.CONFIRMATION,
                null, System.currentTimeMillis())) instanceof ContextResult.Granted<?>); }
        var candidate = new ContextResult.ResponseCandidate(context, ContextResult.ResponseKind.ACK, null, System.currentTimeMillis());
        return new Fixture(owner, context, claim, publication, candidate, item.successResponse(true));
    }

    private enum TerminalForm { RESPONSE, FAILURE, CANCELLATION }

    private record Fixture(AbstractRequestScheduler scheduler, RequestContext requestContext, DeliveryClaim claim,
                           ResponsePublication publication, ContextResult.ResponseCandidate candidate, Response response) { }
}
